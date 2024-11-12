import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from typing import Optional, Tuple

import requests

import diffus.decoded_params
import diffus.task_queue
from diffus.image_gallery import post_output_to_image_gallery
from diffus.redis_client import get_redis_client

logger = logging.getLogger(__name__)


class MonitorException(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message

    def __repr__(self) -> str:
        return f"{self.status_code} {self.code} {self.message}"


class MonitorTierMismatchedException(Exception):
    def __init__(self, msg, current_tier, allowed_tiers):
        self._msg = msg
        self.current_tier = current_tier
        self.allowed_tiers = allowed_tiers

    def __repr__(self) -> str:
        return self._msg


def _get_system_monitor_config(headers: dict) -> Tuple[str, str]:
    # take per-task config as priority instead of global config
    monitor_addr = headers.get(
        'x-diffus-system-monitor-url', ""
    ) or headers.get(
        'X-Diffus-System-Monitor-Url', ""
    )
    system_monitor_api_secret = headers.get(
        'x-diffus-system-monitor-api-secret', ""
    ) or headers.get(
        'X-Diffus-System-Monitor-Api-Secret', ""
    )
    return monitor_addr, system_monitor_api_secret


def _make_headers(extra_data: dict):
    headers = extra_data.get('diffus-request-headers', {})
    result = {}
    for key, value in headers.items():
        key = key.lower()
        if isinstance(value, list):
            if len(value) > 0:
                result[key] = value[0]
            else:
                result[key] = ''
        else:
            result[key] = value
    return result


def before_task_started(
        header_dict: dict,
        api_name: str,
        function_name: str,
        job_id: Optional[str] = None,
        decoded_params: Optional[dict] = None,
        is_intermediate: bool = False,
        refund_if_task_failed: bool = True,
        only_available_for: Optional[list[str]] = None) -> Optional[str]:
    if decoded_params is None and is_intermediate:
        return ''

    if job_id is None:
        job_id = str(uuid.uuid4())
    monitor_addr, system_monitor_api_secret = _get_system_monitor_config(header_dict)
    if not monitor_addr or not system_monitor_api_secret:
        logger.error(f'{job_id}: system_monitor_addr or system_monitor_api_secret is not present')
        return None

    session_hash = header_dict.get('x-session-hash', "")
    if not session_hash:
        logger.error(f'{job_id}: x-session-hash does not presented in headers')
        return None
    task_id = header_dict.get('x-task-id', "")
    if not task_id:
        logger.error(f'{job_id}: x-task-id does not presented in headers')
        return None
    if not is_intermediate and task_id != job_id:
        logger.error(f'x-task-id ({task_id}) and job_id ({job_id}) are not equal')
    deduct_flag = header_dict.get('x-deduct-credits', "")
    deduct_flag = not (deduct_flag == 'false')
    if only_available_for:
        user_tier = header_dict.get('user-tire', '') or header_dict.get('user-tier', '')
        if not user_tier or user_tier.lower() not in [item.lower() for item in only_available_for]:
            raise MonitorTierMismatchedException(
                f'This feature is available for {only_available_for} only. The current user tier is {user_tier}.',
                user_tier,
                only_available_for)

    user_id = header_dict.get('user-id', None) or header_dict.get('user-id', None)
    request_data = {
        'api': api_name,
        'initiator': function_name,
        'user': user_id,
        'started_at': time.time(),
        'session_hash': session_hash,
        'skip_charge': not deduct_flag,
        'refund_if_task_failed': refund_if_task_failed,
        'node': os.getenv('HOST_IP', default=''),
    }
    if is_intermediate:
        request_data['step_id'] = job_id
        request_data['task_id'] = task_id
    else:
        request_data['task_id'] = job_id
    request_data['decoded_params'] = decoded_params if decoded_params is not None else {}
    resp = requests.post(
        monitor_addr,
        headers={
            'Api-Secret': system_monitor_api_secret,
        },
        json=request_data
    )
    logger.info(json.dumps(request_data, ensure_ascii=False, sort_keys=True))

    # check response, raise exception if status code is not 2xx
    if 199 < resp.status_code < 300:
        return job_id

    content = resp.json()
    # log the response if request failed
    logger.error(f'create monitor log failed, status: {resp.status_code}, content: {content}')
    raise MonitorException(resp.status_code, content["code"], content["message"])


def after_task_finished(
        header_dict: dict,
        job_id: Optional[str],
        status: str,
        message: Optional[str] = None,
        is_intermediate: bool = False,
        refund_if_failed: bool = False,
        decoded_params=None, ):
    if decoded_params is None and is_intermediate:
        return {}
    if job_id is None:
        logger.error(
            'task_id is not present in after_task_finished, there might be error occured in before_task_started.')
        return {}
    monitor_addr, system_monitor_api_secret = _get_system_monitor_config(header_dict)
    if not monitor_addr or not system_monitor_api_secret:
        logger.error(f'{job_id}: system_monitor_addr or system_monitor_api_secret is not present')
        return {}

    session_hash = header_dict.get('x-session-hash', "")
    if not session_hash:
        logger.error(f'{job_id}: x-session-hash does not presented in headers')
        return {}
    task_id = header_dict.get('x-task-id', "")
    if not task_id:
        logger.error(f'{job_id}: x-task-id does not presented in headers')
        return {}

    request_url = f'{monitor_addr}/{job_id}'
    request_body = {
        'status': status,
        'result': message if message else "{}",
        'finished_at': time.time(),
        'session_hash': session_hash,
        'refund_if_failed': refund_if_failed,
    }
    if is_intermediate:
        request_body['step_id'] = job_id
        request_body['task_id'] = task_id
    else:
        request_body['task_id'] = job_id
    resp = requests.post(
        request_url,
        headers={
            'Api-Secret': system_monitor_api_secret,
        },
        json=request_body
    )

    # log the response if request failed
    if resp.status_code < 200 or resp.status_code > 299:
        logger.error((f'update monitor log failed, status: monitor_log_id: {job_id}, {resp.status_code}, '
                      f'message: {resp.text[:1000]}'))
    return resp.json()


@contextmanager
def monitor_call_context(
        queue_dispatcher: diffus.task_queue.TaskDispatcher | None,
        extra_data: dict,
        api_name: str,
        function_name: str,
        task_id: Optional[str] = None,
        decoded_params: Optional[dict] = None,
        is_intermediate: bool = True,
        refund_if_task_failed: bool = True,
        refund_if_failed: bool = False,
        only_available_for: Optional[list[str]] = None):
    status = 'unknown'
    message = ''
    task_is_failed = False
    header_dict = _make_headers(extra_data)

    def result_encoder(success, result):
        try:
            nonlocal message
            nonlocal task_is_failed
            message = json.dumps(result, ensure_ascii=False, sort_keys=True)
            task_is_failed = not success
        except Exception as ex:
            logger.error(f'{task_id}: Json encode result failed {ex}.')

    try:
        if not is_intermediate and queue_dispatcher:
            queue_dispatcher.on_task_started(task_id)
        task_id = before_task_started(
            header_dict,
            api_name,
            function_name,
            task_id,
            decoded_params,
            is_intermediate,
            refund_if_task_failed,
            only_available_for)
        yield result_encoder
        if task_is_failed:
            status = 'failed'
        else:
            status = 'finished'
    except Exception as e:
        status = 'failed'
        message = f'{type(e).__name__}: {str(e)}'
        raise e
    finally:
        monitor_result = after_task_finished(
            header_dict,
            task_id,
            status,
            message,
            is_intermediate,
            refund_if_failed,
            decoded_params,
        )
        if not is_intermediate:
            logger.info(f'monitor_result: {monitor_result}')
            extra_data['subscription_consumption'] = monitor_result.get('consumptions', {})
            if queue_dispatcher:
                queue_dispatcher.on_task_finished(task_id, not task_is_failed, message)


def node_execution_monitor(get_output_data):
    import nodes

    redis_client = get_redis_client()

    def wrapper(obj, input_data_all, extra_data, execution_block_cb=None, pre_execute_cb=None):
        node_class_name = type(obj).__name__
        for k, v in nodes.NODE_CLASS_MAPPINGS.items():
            if type(obj) is v:
                node_class_name = k
                break

        with monitor_call_context(
                None,
                extra_data,
                f'comfy.{node_class_name}',
                'comfyui',
                decoded_params=diffus.decoded_params.get_monitor_params(obj, node_class_name, input_data_all),
                is_intermediate=True,
        ) as result_encoder:
            try:
                output_data = get_output_data(obj, input_data_all, extra_data, execution_block_cb, pre_execute_cb)
                result_encoder(True, None)
                post_output_to_image_gallery(redis_client, obj, _make_headers(extra_data), input_data_all, output_data)
                return output_data
            except Exception as ex:
                result_encoder(False, ex)
                raise

    return wrapper


def make_monitor_error_message(ex):
    if isinstance(ex, MonitorException):
        match (ex.status_code, ex.code):
            case (402, "WEBUIFE-01010001"):
                upgrade_info = {
                    "need_upgrade": True,
                    "reason": "INSUFFICIENT_CREDITS",
                }
            case (402, "WEBUIFE-01010003"):
                upgrade_info = {
                    "need_upgrade": True,
                    "reason": "INSUFFICIENT_DAILY_CREDITS",
                }
            case (429, "WEBUIFE-01010004"):
                upgrade_info = {
                    "need_upgrade": True,
                    "reason": "REACH_CONCURRENCY_LIMIT",
                }
            case _:
                logger.error(f"mismatched status_code({ex.status_code}) and code({ex.code}) in 'MonitorException'")
                upgrade_info = {"need_upgrade": False}
    elif isinstance(ex, MonitorTierMismatchedException):
        upgrade_info = {
            "need_upgrade": True,
            "reason": "TIER_MISSMATCH",
        }
    else:
        upgrade_info = {"need_upgrade": False}
    return upgrade_info
