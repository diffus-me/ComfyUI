import asyncio
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager

import aiohttp.web_routedef
from aiohttp import web
import requests
from pydantic import BaseModel, Field
import version
import diffus.redis_client
import diffus.constant

_logger = logging.getLogger(__name__)


class _InstalledModels(BaseModel):
    checkpoints: list[str]
    loras: list[str]
    embeddings: list[str]
    diffusion_models: list[str]


_installed_models: _InstalledModels | None = None


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
    pending_task_count: int = Field(title='PendingTaskCount', default=0)
    consecutive_failed_task_count: int = Field(title='ConsecutiveFailedTaskCount', default=0)
    gpu_utilization: float = Field(title='GpuUtilization', default=0)

    last_error_message: str = Field(title='LastErrorMessage', default='')


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
        self.accepted_tiers: list[str] = os.getenv('ACCEPTED_TIERS').split(',')
        self.accepted_type_types: list[str] = os.getenv('ACCEPTED_TASK_TYPES').split(',')

        self.service_ready = False
        self._redis_client = None
        self.starting_flag = True
        self.finished_task_count = 0
        self.failed_task_count = 0
        self.consecutive_failed_task_count = 0
        self.last_error_message = ''
        self.busy_time = 0

        self.service_interrupted = False

        self.task_thread = None

        self.service_status = 'startup'

        self.current_task = ''
        self.current_task_start_time = 0

    @property
    def redis_client(self):
        if self._redis_client is None:
            self._redis_client = diffus.redis_client.get_redis_client()
        return self._redis_client

    def reset_redis_client(self):
        self._redis_client = None


def _get_model_name_list(model_type: str) -> list[str]:
    if diffus.constant.FAVORITE_MODEL_TYPES:
        return []
    else:
        import folder_paths
        return folder_paths.get_filename_list_(model_type)[0]


def _setup_daemon_api(_server_instance, _task_state: _State, routes: aiohttp.web_routedef.RouteTableDef):
    service_started_at = time.time()

    @routes.get("/daemon/v1/status")
    async def get_status(request):
        if _task_state.current_task_start_time > 0:
            busy_time = _task_state.busy_time + time.time() - _task_state.current_task_start_time
        else:
            busy_time = _task_state.busy_time
        live_time = time.time() - service_started_at
        current_task = ""
        for i in range(3):
            current_task = _task_state.current_task or _server_instance.current_prompt_id or ''
            if current_task:
                break
            await asyncio.sleep(0.05)

        _prompt_queue = _server_instance.prompt_queue
        resp = ServiceStatusResponse(
            release_version=version.version,
            node_type=_task_state.node_type,
            instance_name=_task_state.node_name,
            accepted_tiers=_task_state.accepted_tiers,
            accepted_type_types=_task_state.accepted_type_types,

            status=_task_state.service_status,
            current_task=current_task,
            finished_task_count=_task_state.finished_task_count,
            failed_task_count=_task_state.failed_task_count,
            pending_task_count=_prompt_queue.get_tasks_remaining(),
            consecutive_failed_task_count=_task_state.consecutive_failed_task_count,
            gpu_utilization=busy_time / live_time,
            last_error_message=_task_state.last_error_message
        )

        return web.json_response(resp.model_dump())

    @routes.put("/daemon/v1/status")
    async def update_status(request):
        request_data = await request.json()
        req = ServiceStatusRequest(**request_data)
        if req.status:
            if _task_state.service_status != req.status:
                _logger.info(f'update_status: service status was set to {_task_state.service_status}')
            _task_state.service_status = req.status
        if req.accepted_tiers:
            _task_state.accepted_tiers = req.accepted_tiers

        resp = await get_status(request)
        return resp

    @routes.get("/daemon/v1/models")
    async def get_installed_models(request):
        global _installed_models

        if _installed_models is None:
            _installed_models = _InstalledModels(
                checkpoints=_get_model_name_list("checkpoints"),
                loras=_get_model_name_list("loras"),
                embeddings=_get_model_name_list("embeddings"),
                diffusion_models=_get_model_name_list("diffusion_models"),
            )
        return web.json_response(_installed_models.model_dump())


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
    if not task_id:
        task_id = str(uuid.uuid4())
    timeout = request_obj.get('timeout', 15 * 60)
    headers = request_obj['headers']
    path = request_obj['path']

    encoded_headers = {}
    for k, v in headers.items():
        encoded_headers[k] = ';'.join(v)
    encoded_headers['X-Predict-Timeout'] = f'{timeout}'
    encoded_headers['X-Task-Timeout'] = f'{timeout}'
    encoded_headers['X-Task-Id'] = task_id

    # read body from file if body_path is set
    request_data = {}
    if request_obj.get('body_path', None):
        body_path = request_obj['body_path']
        try:
            with open(body_path, 'r') as f:
                request_data = json.load(f)
            os.remove(body_path)
        except Exception as e:
            _logger.warning(f'failed to remove temporary body file {body_path}: {e}')

    if not request_data:
        request_data = json.loads(request_obj.get('body', "{}"))

    # pack request headers to extra_data
    extra_data = request_data.get('extra_data', {})
    extra_data['diffus-request-headers'] = encoded_headers
    request_data['extra_data'] = extra_data

    # post body to service
    try:
        request_url = f'http://localhost:{_task_state.service_port}{path}'
        request_timeout = timeout + 3

        resp = requests.post(request_url,
                             headers=encoded_headers,
                             json=request_data,
                             timeout=request_timeout)
        if not (199 < resp.status_code < 300):
            return True
        else:
            _logger.error(f"failed to post task to server: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        _logger.exception(f'failed to post task status to redis: {e}')
        _task_state.reset_redis_client()
        if retry > 0:
            return _post_task(_task_state, request_obj, retry=retry - 1)
        else:
            raise


def _fetch_task(_task_state: _State, remaining_tasks: int, fetch_task_timeout=5):
    if not _task_state.current_task and remaining_tasks == 0:
        queue_name_list = []
        for task_type in _task_state.accepted_type_types:
            queue_name_list += [f"SD-{task_type}-TASKS-{tier}" for tier in _task_state.accepted_tiers]
        _logger.debug(
            f"begin to fetch pending requests from {queue_name_list}, current task: '{_task_state.current_task}'")
        queued_task = _task_state.redis_client.blpop(queue_name_list, timeout=fetch_task_timeout)
        if not queued_task:
            _logger.debug(f'not get any pending requests in {fetch_task_timeout} seconds from {queue_name_list}')
            # no task get, check service status, and fetch task again
            return None

        queue_name, packed_request = queued_task[0], queued_task[1]
        if not packed_request or not queue_name:
            # no task get, check service status, and fetch task again
            return None

        _logger.info(f'popped a task from {queue_name}: {packed_request}')
        return json.loads(packed_request.decode('utf-8'))
    else:
        time.sleep(fetch_task_timeout)
        return None


class TaskDispatcher:
    def __init__(self, service_instance, routes: aiohttp.web_routedef.RouteTableDef):
        self._task_state = _State()
        self._server_instance = service_instance
        self._prompt_queue = service_instance.prompt_queue
        self._disable_embedded_task_dispatcher = os.getenv(
            'DISABLE_EMBEDDED_TASK_DISPATCHER', "false"
        ).lower() in ('true', '1')
        if not self._disable_embedded_task_dispatcher:
            self._t = threading.Thread(target=self._task_loop, name='comfy-task-dispatcher-thread')
        else:
            self._t = None
        _setup_daemon_api(self._server_instance, self._task_state, routes)

    def start(self):
        self._task_state.service_status = 'up'
        self._task_state.service_ready = True
        if self._t:
            self._t.start()

    def stop(self):
        self._task_state.service_interrupted = True
        if self._t:
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
                remaining_tasks = self._prompt_queue.get_tasks_remaining()
                # 2. fetch a task from remote queue
                task = _fetch_task(self._task_state, remaining_tasks, fetch_task_timeout=5)
                # 3. post task to service
                if task:
                    _post_task(self._task_state, task)

                    # sleep 3 seconds, let current_task of self._task_state to get updated
                    time.sleep(3)
            except Exception as e:
                _logger.exception(f'failed to fetch task from queue: {e.__str__()}')
                time.sleep(3)
        _logger.info(f'daemon_main: service is {self._task_state.service_status}, exit task loop now')

    def _make_current_key_in_redis(self):
        return f"diffus:comfyui:node-current-task:{self._task_state.host_ip}:{self._task_state.service_port}"

    def _publish_current_task_prams_to_redis(self, task_item):
        context = task_item[-1]
        user_id = context.user_id
        self._task_state.redis_client.set(
            self._make_current_key_in_redis(),
            json.dumps({
                "user_id": user_id,
                "prompt": [
                    0,
                    task_item[1],
                    task_item[2],
                    {},
                    task_item[4],
                ],
            }),
            ex=3600,
        )

    def _remove_current_task_prams_from_redis(self):
        self._task_state.redis_client.delete(self._make_current_key_in_redis())

    @contextmanager
    def dispatch(self, task_item):
        prompt_id = task_item[1]

        _logger.info(f'on_task_started, {prompt_id}')
        self._publish_current_task_prams_to_redis(task_item)

        self._task_state.current_task = prompt_id
        self._task_state.current_task_start_time = time.time()

        yield self._on_task_finished

        if self._task_state.current_task != prompt_id:
            _logger.warning(
                f'on_task_finished, task_state.current_task({self._task_state.current_task}) and task_id{prompt_id} are mismatched'
            )
        self._task_state.busy_time += time.time() - self._task_state.current_task_start_time
        self._task_state.current_task_start_time = 0
        self._task_state.current_task = ''

        self._remove_current_task_prams_from_redis()
        _logger.info(f'on_task_finished, {prompt_id}')

    def _on_task_finished(self, task_id, success, messages, monitor_error=None):
        if success or monitor_error is not None:
            self._task_state.finished_task_count += 1
            self._task_state.consecutive_failed_task_count = 0
            self._task_state.last_error_message = ""
        else:
            self._task_state.failed_task_count += 1
            self._task_state.consecutive_failed_task_count += 1
            if len(messages) > 0:
                self._task_state.last_error_message = str(messages[-1])
