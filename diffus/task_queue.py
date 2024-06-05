import json
import logging
import os
import threading
import time

import aiohttp.web_routedef
from aiohttp import web
import requests
from pydantic import BaseModel, Field
import version
import diffus.redis_client

_logger = logging.getLogger(__name__)


class ServiceStatusResponse(BaseModel):
    status: str = Field(title="Status", default='startup')
    release_version: str = Field(default='')

    node_type: str = Field(title="NodeType", default='FOCUS')
    instance_name: str = Field(default='')

    accepted_tiers: list[str] = Field(default=[])
    accepted_type_types: list[str] = Field(default=[])

    current_task: str = Field(title="CurrentTask", default='')
    queued_tasks: list = Field(title='QueuedTasks', default=[])
    finished_task_count: int = Field(title='FinishedTaskCount', default=0)
    failed_task_count: int = Field(title='FailedTaskCount', default=0)


class ServiceStatusRequest(BaseModel):
    status: str = Field(title="Status", default='startup')
    node_type: str = Field(title="NodeType", default='FOCUS')
    accepted_tiers: list[str] = Field(default=[])
    accepted_type_types: list[str] = Field(default=[])


class _State:
    def __init__(self) -> None:
        self.host_ip = os.getenv('HOST_IP', '127.0.0.1')
        self.service_port = os.getenv('SERVER_PORT')

        self.node_name = os.getenv('NODE_NAME')
        self.node_type = os.getenv("NODE_TYPE")
        self.accepted_tiers: [str] = os.getenv('ACCEPTED_TIERS').split(',')
        self.accepted_type_types: [str] = os.getenv('ACCEPTED_TASK_TYPES').split(',')

        self.service_ready = False
        self._redis_client = None
        self.starting_flag = True
        self.finished_task_count = 0
        self.failed_task_count = 0
        self.last_error_message = ''
        self.service_interrupted = False

        self.task_thread = None

        self.service_status = 'startup'

        self.current_task = ''
        self.remaining_tasks = 0

    @property
    def redis_client(self):
        if self._redis_client is None:
            self._redis_client = diffus.redis_client.get_redis_client()
        return self._redis_client

    def reset_redis_client(self):
        self._redis_client = None


def _setup_daemon_api(_task_state: _State, routes: aiohttp.web_routedef.RouteTableDef):
    @routes.get("/daemon/v1/status")
    async def get_status(request):
        resp = ServiceStatusResponse(
            release_version=version.version,
            node_type=_task_state.node_type,
            instance_name=_task_state.node_name,
            accepted_tiers=_task_state.accepted_tiers,
            accepted_type_types=_task_state.accepted_type_types,

            status=_task_state.service_status,
            current_task=_task_state.current_task or '',
            finished_task_count=_task_state.finished_task_count,
            failed_task_count=_task_state.failed_task_count,
            pending_task_count=_task_state.remaining_tasks
        ).model_dump()

        return web.json_response(resp)

    @routes.put("/daemon/v1/status")
    async def update_status(request):
        request_data = await request.json()
        req = ServiceStatusRequest(**request_data)
        if req.status:
            _task_state.service_status = req.status
        if req.accepted_tiers:
            _task_state.accepted_tiers = req.accepted_tiers
        _logger.info(f'update_status: service status was set to {_task_state.service_status}')
        resp = await get_status(request)
        return resp


def _service_is_alive(_task_state: _State):
    try:
        request_url = f'http://localhost:{_task_state.service_port}/daemon/v1/status'
        resp = requests.get(request_url, json={})
        if resp.status_code == 200:
            return True
        else:
            _logger.warning(f'failed to check service status on {request_url}: {resp.status_code} {resp.text}')
            return False
    except Exception as e:
        _logger.warning(f'failed to check service status: {e.__str__()}')
        return False


def _post_task(_task_state: _State, request_obj, retry=1):
    task_id = request_obj['task_id']
    timeout = request_obj.get('timeout', 15 * 60)
    headers = request_obj['headers']
    path = request_obj['path']


    try:
        encoded_headers = {}
        for k, v in headers.items():
            encoded_headers[k] = ';'.join(v)
        encoded_headers['X-Predict-Timeout'] = f'{timeout}'
        encoded_headers['X-Task-Timeout'] = f'{timeout}'
        encoded_headers['X-Task-Id'] = task_id
        # pack request headers to extra_data
        request_data = json.loads(request_obj['body'])
        extra_data = request_data.get('extra_data', {})
        extra_data['diffus-request-headers'] = encoded_headers
        request_data['extra_data'] = extra_data

        request_url = f'http://localhost:{_task_state.service_port}{path}'
        request_timeout = timeout + 3

        resp = requests.post(request_url,
                             headers=encoded_headers,
                             json=request_data,
                             timeout=request_timeout)
        if not (199 < resp.status_code < 300):
            return False
        else:
            return True
    except Exception as e:
        _logger.exception(f'failed to post task status to redis: {e}')
        _task_state.reset_redis_client()
        if retry > 0:
            return _post_task(_task_state, request_obj, retry=retry - 1)
        else:
            raise


def _fetch_task(_task_state: _State, fetch_task_timeout=5):
    if not _task_state.current_task and _task_state.remaining_tasks == 0:
        queue_name_list = [f"SD-COMFY-TASKS-{tier}" for tier in _task_state.accepted_tiers]
        _logger.info(
            f"begin to fetch pending requests from {queue_name_list}, current task: '{_task_state.current_task}'")
        queued_task = _task_state.redis_client.blpop(queue_name_list, timeout=fetch_task_timeout)
        if not queued_task:
            _logger.debug(f'not get any pending requests in {fetch_task_timeout} seconds from {queue_name_list}')
            # no task get, check service status, and fetch task again
            return

        queue_name, packed_request = queued_task[0], queued_task[1]
        if not packed_request or not queue_name:
            # no task get, check service status, and fetch task again
            return

        _logger.info(f'popped a task from {queue_name}: {packed_request}')
        return json.loads(packed_request.decode('utf-8'))
    else:
        time.sleep(fetch_task_timeout)


class TaskDispatcher:
    def __init__(self, prompt_queue, routes: aiohttp.web_routedef.RouteTableDef):
        self._task_state = _State()
        self._prompt_queue = prompt_queue
        self._t = threading.Thread(target=self._task_loop, name='comfy-task-dispatcher-thread')
        _setup_daemon_api(self._task_state, routes)

    def start(self):
        self._task_state.service_status = 'up'
        self._task_state.service_ready = True
        self._t.start()

    def stop(self):
        self._task_state.service_interrupted = True
        self._t.join()

    def _task_loop(self):
        while not self._task_state.service_interrupted \
                and not _service_is_alive(self._task_state):
            time.sleep(1)

        while not self._task_state.service_interrupted \
                and self._task_state.service_ready \
                and self._task_state.service_status == 'up':
            try:
                # 1. update current state
                self._task_state.remaining_tasks = self._prompt_queue.get_tasks_remaining()
                # 2. fetch a task from remote queue
                task = _fetch_task(self._task_state, fetch_task_timeout=5)
                # 3. post task to service
                if task:
                    _post_task(self._task_state, task)
            except Exception as e:
                _logger.exception(f'failed to fetch task from queue: {e.__str__()}')
                time.sleep(3)
        _logger.info(f'daemon_main: service is {self._task_state.service_status}, exit task loop now')

    def on_task_started(self, task_id):
        _logger.info(f'on_task_started, {task_id}')
        self._task_state.current_task = task_id

    def on_task_finished(self, task_id, success, messages):
        _logger.info(f'on_task_finished, {task_id} {success} {messages}')
        if self._task_state.current_task != task_id:
            _logger.warning(
                f'on_task_finished, task_state.current_task({self._task_state.current_task}) and task_id{task_id} are mismatched')
        self._task_state.current_task = ''
        if success:
            self._task_state.finished_task_count += 1
        else:
            self._task_state.failed_task_count += 1
            self._task_state.last_error_message = messages
