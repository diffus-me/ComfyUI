import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from typing import Optional, Iterable

import requests

import diffus.decoded_params
import diffus.task_queue
import execution_context
import folder_paths

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


def _get_system_monitor_config(headers: dict):
    # take per-task config as priority instead of global config
    monitor_addr = headers.get('x-diffus-system-monitor-url', None) or headers.get('X-Diffus-System-Monitor-Url', None)
    system_monitor_api_secret = headers.get('x-diffus-system-monitor-api-secret', None) or headers.get(
        'X-Diffus-System-Monitor-Api-Secret', None)
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

    session_hash = header_dict.get('x-session-hash', None)
    if not session_hash:
        logger.error(f'{job_id}: x-session-hash does not presented in headers')
        return None
    task_id = header_dict.get('x-task-id', None)
    if not task_id:
        logger.error(f'{job_id}: x-task-id does not presented in headers')
        return None
    if not is_intermediate and task_id != job_id:
        logger.error(f'x-task-id ({task_id}) and job_id ({job_id}) are not equal')
    deduct_flag = header_dict.get('x-deduct-credits', None)
    deduct_flag = not (deduct_flag == 'false')
    if only_available_for:
        user_tier = header_dict.get('user-tire', None) or header_dict.get('user-tier', None)
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
    resp = requests.post(monitor_addr,
                         headers={
                             'Api-Secret': system_monitor_api_secret,
                         },
                         json=request_data)
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

    session_hash = header_dict.get('x-session-hash', None)
    if not session_hash:
        logger.error(f'{job_id}: x-session-hash does not presented in headers')
        return {}
    task_id = header_dict.get('x-task-id', None)
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
    resp = requests.post(request_url,
                         headers={
                             'Api-Secret': system_monitor_api_secret,
                         },
                         json=request_body)

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
        except Exception as e:
            logger.error(f'{task_id}: Json encode result failed {str(e)}.')

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
                post_output_to_image_gallery(obj, _make_headers(extra_data), input_data_all, output_data)
                return output_data
            except Exception as ex:
                result_encoder(False, ex)
                raise

    return wrapper


def _find_user_hash_from_input_data(input_data):
    if not isinstance(input_data, dict):
        return ""
    for key, value in input_data.items():
        if key == "user_hash":
            return value[0]
        elif key == "context":
            return value[0].user_hash
    return ""


def _find_extra_pnginfo_from_input_data(input_data):
    if not isinstance(input_data, dict):
        return ""
    for key, value in input_data.items():
        if key == "extra_pnginfo":
            return json.dumps(value)
    return ""


def _do_post_image_to_gallery(task_id, user_id, user_hash, image_type, image_subfolder, image_filename, pnginfo):
    if image_type != "output":
        return
    post_url = "http://10.10.24.46:7070/gallery-api/v1/images"
    post_json = {
        "task_id": task_id,
        "path": os.path.join(
            "comfyui",
            folder_paths.get_relative_output_directory(user_hash), image_subfolder, image_filename,
        ),
        "feature": "COMFYUI",
        "pnginfo": pnginfo,
        "created_by": user_id,
        "base": None,
        "prompt": None,
        "model_ids": [],
        "is_public": False,
    }
    resp = requests.post(
        url=post_url,
        json=post_json
    )

    if 199 < resp.status_code < 300:
        logger.debug(
            f"succeeded to post image to gallery, {resp.status_code} {resp.text}, url={post_url}, post_json={post_json}"
        )
    else:
        logger.error(
            f"failed to post image to gallery, {resp.status_code} {resp.text}, url={post_url}, post_json={post_json}"
        )


def post_output_to_image_gallery(node_obj, header_dict, input_data, output_data):
    if not output_data:
        return

    user_hash = _find_user_hash_from_input_data(input_data)
    if not user_hash:
        return

    user_id = header_dict.get('user-id', None) or header_dict.get('user-id', None)
    if not user_id:
        return

    result_data, ui_data, _ = output_data

    if not isinstance(ui_data, dict):
        return

    proceeded_files = set()

    task_id = header_dict.get('x-task-id', str(uuid.uuid4()))
    for images_key in ("images", "gifs", "video"):
        if images_key not in ui_data:
            continue

        if not isinstance(ui_data[images_key], Iterable):
            continue

        for image in ui_data[images_key]:
            if not isinstance(image, dict):
                logger.error("image is not a dict, do nothing")
                continue
            image_type = image["type"]
            image_subfolder = image["subfolder"]
            image_filename = image["filename"]
            pnginfo = _find_extra_pnginfo_from_input_data(input_data=input_data)

            if image_filename in proceeded_files:
                continue
            _do_post_image_to_gallery(task_id, user_id, user_hash, image_type, image_subfolder, image_filename, pnginfo)
            proceeded_files.add(image_filename)

    if hasattr(node_obj, "RETURN_TYPES") and "VHS_FILENAMES" in node_obj.RETURN_TYPES:
        for node_result in result_data[node_obj.RETURN_TYPES.index("VHS_FILENAMES")]:
            if node_result[0]:
                for filepath in node_result[1]:
                    output_directory_len = len(folder_paths.get_output_directory(user_hash))
                    filename = filepath[output_directory_len+1:]
                    if filename in proceeded_files:
                        continue
                    _do_post_image_to_gallery(
                        task_id,
                        user_id,
                        user_hash,
                        "output",
                        "",
                        filename,
                        ""
                    )
                    proceeded_files.add(filename)


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
