import json
import logging
from enum import Enum

import redis
import requests
from pydantic import BaseModel

import diffus.redis_client

logger = logging.getLogger(__name__)


class ImageResult(BaseModel):
    filename: str | None = None
    subfolder: str | None = None
    type: str | None = None
    presigned_url: str | None = None
    user_hash: str | None = None


class GifResult(BaseModel):
    filename: str | None = None
    subfolder: str | None = None
    type: str | None = None
    format: str | None = None
    frame_rate: float | None = None
    user_hash: str | None = None


class PromptOutput(BaseModel):
    images: list[ImageResult] | None = None
    gifs: list[ImageResult] | None = None
    video: list[ImageResult] | None = None


class SubscriptionConsumption(BaseModel):
    discount: float | None = None
    credit_consumption: int | None = None
    machine_time_consumption: int | None = None
    gpu_acceleration_time_consumption: int | None = None


class MonitorInfo(BaseModel):
    monitor_addr: str
    system_monitor_api_secret: str


class PromptState(str, Enum):
    waiting = "waiting"
    queued = "queued"
    started = "started"
    executing = "executing"
    success = "success"
    error = "error"
    execution_interrupted = "execution_interrupted"
    monitor_error = "monitor_error"
    finished = "finished"


class MsgData(BaseModel):
    prompt_id: str
    node: str | None = None
    node_id: str | None = None
    node_type: str | None = None

    value: int | None = None
    max: int | None = None
    display_node: str | None = None
    output: PromptOutput | None = None
    timestamp: int | None = None
    exception_message: str | None = None
    exception_type: str | None = None
    traceback: list[str] | None = None
    preview_img: str | None = None
    used_time: float | None = None
    subscription_consumption: SubscriptionConsumption | None = None
    monitor_info: MonitorInfo | None = None
    message: dict | None = None
    executed: str | list[str] | None = None


class PromptMessages(BaseModel):
    type: str
    data: MsgData | None = None


class ProgressProgress(BaseModel):
    value: int
    max: int
    node: str


class ProgressResult(BaseModel):
    used_time: float | None = None
    subscription_consumption: SubscriptionConsumption | None = None
    monitor_info: MonitorInfo | None = None
    message: dict | None = None


class PromptStatus(BaseModel):
    prompt_id: str
    client_id: str | None = None
    progress: ProgressProgress | None = None
    outputs: list[PromptOutput] | None = None
    result: ProgressResult | None = None
    state: PromptState = PromptState.waiting  # started, executing, success, error, interrupted
    success: bool | None = None
    preview_img: str | None = None
    last_msg: PromptMessages | None = None

    @property
    def finished(self):
        return self.state in (PromptState.finished, PromptState.monitor_error)

    def to_json_str(self):
        return self.model_dump_json(
            include={"outputs", "state", "success"},
            exclude_none=True, exclude_unset=True,
        )


def _make_prompt_result_key(prompt_id: str) -> str:
    return f'diffus:comfyui:prompt:status:{prompt_id}'


def _update_prompt_result(
        redis_client: redis.Redis,
        prompt_id: str,
        prompt_messages: PromptMessages,
        prompt_result: PromptStatus,
):
    if not prompt_result:
        return

    redis_client.set(
        name=_make_prompt_result_key(prompt_id),
        value=prompt_result.model_dump_json(),
        ex=60 * 60
    )

    if prompt_result.finished and prompt_messages.data.monitor_info:
        monitor_addr = prompt_messages.data.monitor_info.monitor_addr
        system_monitor_api_secret = prompt_messages.data.monitor_info.system_monitor_api_secret
        resp = requests.patch(
            url=f"{monitor_addr}/{prompt_id}",
            headers={
                "Content-Type": "application/json",
                'Api-Secret': system_monitor_api_secret,
            },
            json={
                "update_type": "result",
                "result": prompt_result.to_json_str(),
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
    try:
        if prompt_status_str:
            return PromptStatus.model_validate_json(prompt_status_str)
    except Exception as e:
        logger.exception(f"failed to validate '{prompt_status_str}' to PromptStatus: {e}")
    return None


def _process_prompt_message(
        client_id: str,
        prompt_id: str,
        prompt_result: PromptStatus,
        msg: PromptMessages
) -> PromptStatus:
    # fetch existing prompt from redis, or create a new if not exists
    if not prompt_result:
        result = PromptStatus(
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
        result.result = ProgressResult(
            message={
                "reason": msg.data.exception_type,
                "detail": msg.data.exception_message
            },
            used_time=msg.data.used_time
        )
    elif msg.type == "execution_interrupted":
        result.success = False
        result.state = PromptState.execution_interrupted
    elif msg.type == "executing":
        result.state = PromptState.executing
    elif msg.type == "progress":
        result.progress = msg.data
    elif msg.type == "executed":
        if result.outputs is None:
            result.outputs = []
        result.outputs.append(msg.data.output)
    elif msg.type == "monitor_error":
        result.success = False
        result.state = PromptState.monitor_error
        result.result = ProgressResult(message=msg.data.message, used_time=msg.data.used_time)
    elif msg.type == "finished":
        result.state = PromptState.finished
        result.result = msg.data
    elif msg.type == "preview":
        result.preview_img = msg.data.preview_img
    else:
        logger.warning(f"unknown comfyui prompt message type: {msg.type}")
    result.last_msg = msg
    return result


class MessageQueue:
    def __init__(self):
        self._redis_client = diffus.redis_client.get_redis_client()

    def _publish_prompt_message(self, sid, message):
        if not sid:
            sid = "anonymous"
        keys = [
            f'COMFYUI_MESSAGE_{sid}',
            f'diffus:comfyui:message:{sid}'
        ]

        for key in keys:
            if isinstance(message, str) and "monitor_info" in message:
                msg_dict = json.loads(message)
                del msg_dict["data"]["monitor_info"]
                msg = json.dumps(msg_dict)
            else:
                msg = message
            self._redis_client.rpush(key, msg)
            self._redis_client.expire(key, 60 * 60)

    def _update_prompt_result(self, sid, message):
        if not isinstance(message, str) or "prompt_id" not in message:
            return

        try:
            msg = PromptMessages.model_validate_json(message)
            prompt_id = msg.data.prompt_id
            if not prompt_id:
                return
        except Exception as e:
            logger.exception(f"failed to validate prompt message '{message}': {e}")
            return

        try:
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
            logger.exception(f"failed to update prompt result '{message}': {e}")

    def send_message(self, sid: str, message: bytes | str, retry=1):
        if not self._redis_client:
            self._redis_client = diffus.redis_client.get_redis_client()
        try:
            self._publish_prompt_message(sid, message)
            self._update_prompt_result(sid, message)
        except Exception as e:
            logger.exception(f"failed to send message: {e}")
            if retry > 0:
                self._redis_client = None
                self.send_message(sid, message, retry=retry - 1)


def fetch_prompt_result(prompt_id: str) -> PromptStatus:
    redis_client = diffus.redis_client.get_redis_client()
    return _fetch_prompt_result(
        redis_client=redis_client,
        prompt_id=prompt_id
    )
