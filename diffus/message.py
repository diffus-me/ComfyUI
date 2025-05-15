import json
import logging
from enum import Enum

import redis
import requests
from pydantic import BaseModel

import diffus.redis_client

logger = logging.getLogger(__name__)


class ImageResult(BaseModel):
    filename: str
    subfolder: str
    type: str


class PromptOutput(BaseModel):
    images: list[ImageResult]


class SubscriptionConsumption(BaseModel):
    discount: float
    credit_consumption: int
    machine_time_consumption: int
    gpu_acceleration_time_consumption: int


class MonitorInfo(BaseModel):
    monitor_addr: str
    system_monitor_api_secret: str


class PromptState(str, Enum):
    waiting = "waiting"
    started = "started"
    executing = "executing"
    success = "success"
    error = "error"
    execution_interrupted = "execution_interrupted"
    monitor_error = "monitor_error"
    finished = "finished"


class MsgProgress(BaseModel):
    prompt_id: str
    value: int
    max: int
    node: str


class MsgExecuting(BaseModel):
    prompt_id: str
    node: str | None
    display_node: str | None = None


class MsgExecuted(BaseModel):
    prompt_id: str
    node: str
    display_node: str
    output: PromptOutput


class MsgBase(BaseModel):
    prompt_id: str
    timestamp: int


class MsgExecutionError(BaseModel):
    prompt_id: str
    node_id: str
    node_type: str
    executed: str
    exception_message: str
    exception_type: str
    traceback: str


class MsgFinished(BaseModel):
    prompt_id: str
    node: str | None
    used_time: float
    subscription_consumption: SubscriptionConsumption | None = None
    monitor_info: MonitorInfo | None = None
    message: dict | None = None


class MsgExecutionInterrupted(BaseModel):
    prompt_id: str
    node_id: str
    node_type: str
    executed: str


class PromptMessages(BaseModel):
    type: str
    data: MsgProgress \
          | MsgExecuted \
          | MsgExecuting \
          | MsgExecutionError \
          | MsgExecutionInterrupted \
          | MsgFinished \
          | MsgBase \
          | None = None


class PromptResult(BaseModel):
    prompt_id: str
    client_id: str
    progress: MsgProgress | None = None
    outputs: list[PromptOutput] | None = None
    result: MsgFinished | None = None
    state: PromptState = PromptState.waiting  # started, executing, success, error, interrupted
    success: bool | None = None
    last_msg: PromptMessages | None = None

    @property
    def finished(self):
        return self.state in (PromptState.finished, PromptState.monitor_error)


def _make_prompt_result_key(prompt_id: str) -> str:
    return f'diffus:comfyui:prompt:{prompt_id}'


def _update_prompt_result(
        redis_client: redis.Redis,
        prompt_id: str,
        prompt_messages: PromptMessages,
        prompt_result: PromptResult,
):
    redis_client.set(
        name=_make_prompt_result_key(prompt_id),
        value=prompt_result.json(),
        ex=60 * 60
    )

    if prompt_result.finished \
            and prompt_messages.type in ("monitor_error", "finished") \
            and prompt_messages.data.monitor_info is not None:
        msg: MsgFinished = prompt_messages.data
        monitor_addr = msg.monitor_info.monitor_addr
        system_monitor_api_secret = msg.monitor_info.system_monitor_api_secret
        logger.info(f'** monitor addr: {monitor_addr}')
        logger.info(f'** monitor api secret: {system_monitor_api_secret}')
        logger.info(f'** monitor result: {prompt_result.json()}')
        resp = requests.patch(
            url=f"{monitor_addr}/{prompt_id}",
            headers={
                "Content-Type": "application/json",
                'Api-Secret': system_monitor_api_secret,
            },
            json={
                "update_type": "result",
                "result": prompt_result.json(
                    exclude={"prompt_id", "result", "last_msg"},
                    exclude_none=True
                ),
            }
        )
        if resp.status_code != 200:
            logger.error(f'*** monitor error: {resp.text}')


def _fetch_prompt_result(
        redis_client,
        prompt_id: str,
):
    prompt_status_str = redis_client.get(
        _make_prompt_result_key(prompt_id)
    )
    if prompt_status_str:
        return PromptResult.validate(json.loads(prompt_status_str))
    else:
        return None


def _process_prompt_message(
        client_id: str,
        prompt_id: str,
        prompt_result: PromptResult,
        msg: PromptMessages
):
    # fetch existing prompt from redis, or create a new if not exists
    if not prompt_result:
        result = PromptResult(
            prompt_id=prompt_id,
            client_id=client_id,
            outputs=[],
        )
    else:
        result = prompt_result

    # update status
    if msg.type == "status":
        pass
    elif msg.type == "execution_start":
        result.state = PromptState.started
    elif msg.type == "execution_cached":
        pass
    elif msg.type == "execution_success":
        result.success = True
    elif msg.type == "execution_error":
        result.success = False
    elif msg.type == "execution_interrupted":
        result.success = False
        result.state = PromptState.execution_interrupted
    elif msg.type == "executing":
        result.state = PromptState.executing
    elif msg.type == "progress":
        result.progress = msg.data
    elif msg.type == "executed":
        executed: MsgExecuted = msg.data
        result.outputs.append(executed.output)
    elif msg.type == "monitor_error":
        result.success = False
        result.state = PromptState.monitor_error
    elif msg.type == "finished":
        result.state = PromptState.finished
        result.result = msg.data
    else:
        logger.warning(f"unknown comfyui prompt message type: {msg.type}")
    result.last_msg = msg
    return result


class MessageQueue:
    def __init__(self):
        self._redis_client = diffus.redis_client.get_redis_client()

    def _publish_prompt_message(self, sid, message):
        if sid:
            keys = [
                f'COMFYUI_MESSAGE_{sid}',
                f'diffus:comfyui:message:{sid}'
            ]
        else:
            keys = [
                f'COMFYUI_MESSAGE_anonymous',
                f'diffus:comfyui:message:anonymous'
            ]
        for key in keys:
            if "monitor_info" in message:
                msg_dict = json.loads(message)
                del msg_dict["data"]["monitor_info"]
                msg = json.dumps(msg_dict)
            else:
                msg = message
            self._redis_client.rpush(key, msg)
            self._redis_client.expire(key, 60 * 60)

    def _update_prompt_result(self, sid, message):
        if "prompt_id" not in message:
            return

        try:
            msg_dict = json.loads(message)
            msg = PromptMessages.validate(msg_dict)
            prompt_id = msg.data.prompt_id
            if not prompt_id:
                return
            prompt_result = _process_prompt_message(
                client_id=sid,
                prompt_id=prompt_id,
                prompt_result=_fetch_prompt_result(self._redis_client, prompt_id),
                msg=msg
            )
            _update_prompt_result(
                redis_client=self._redis_client,
                prompt_id=prompt_id,
                prompt_messages=msg,
                prompt_result=prompt_result,
            )
        except Exception as e:
            logger.exception(f"failed to validate prompt message '{message}': {e}")

    def send_message(self, sid: str, message: bytes | str, retry=1):
        if not self._redis_client:
            self._redis_client = diffus.redis_client.get_redis_client()
        try:
            self._publish_prompt_message(sid, message)
            self._update_prompt_result(sid, message)
        except Exception as e:
            logger.exception(e)
            if retry > 0:
                self._redis_client = None
                self.send_message(sid, message, retry=retry - 1)


def fetch_prompt_result(prompt_id: str) -> PromptResult:
    redis_client = diffus.redis_client.get_redis_client()
    return _fetch_prompt_result(
        redis_client=redis_client,
        prompt_id=prompt_id
    )
