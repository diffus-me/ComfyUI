import json
import logging
import os
import pathlib
import uuid
from typing import Iterable

import requests

import execution_context
from diffus.service_registrar import get_service_node
import folder_paths

logger = logging.getLogger(__name__)


def _do_post_image_to_gallery(
        post_url,
        task_id,
        user_id,
        user_hash,
        image_type,
        image_subfolder,
        image_filename,
        positive_prompt: str,
        pnginfo,
        model_base: str,
        model_ids: list[int],
):
    if image_type != "output":
        return
    post_json = {
        "task_id": task_id,
        "path": os.path.join(
            folder_paths.get_relative_output_directory(user_hash),
            image_subfolder,
            image_filename,
        ),
        "feature": "COMFYUI",
        "pnginfo": json.dumps(pnginfo),
        "created_by": user_id,
        "base": model_base,
        "prompt": positive_prompt,  # positive prompt
        "model_ids": model_ids,  # checkpoint, loras
        "is_public": False,
    }
    try:
        resp = requests.post(
            url=post_url,
            timeout=5,
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
        result = resp.json()
        return result.get("url", None)
    except Exception as e:
        logger.error(
            f"failed to post image to gallery, {e}, url={post_url}, post_json={post_json}"
        )
        return None


def post_output_to_image_gallery(redis_client, node_obj, header_dict, input_data, output_data):
    if not output_data:
        return

    user_hash = _find_user_hash_from_input_data(input_data)
    if not user_hash:
        return

    user_id = header_dict.get('user-id', None) or header_dict.get('User-Id', None)
    if not user_id:
        return

    disable_post = header_dict.get('x-disable-gallery-post', None) or header_dict.get('X-Disable-Gallery-Post', None)
    if disable_post and disable_post.lower() == "true":
        logger.warning(f"post result to gallery is disabled")
        return

    result_data, ui_data, _ = output_data

    if not isinstance(ui_data, dict):
        return

    proceeded_files = set()

    task_id = header_dict.get('x-task-id', str(uuid.uuid4()))
    gallery_service_node = get_service_node(redis_client, "gallery")
    if not gallery_service_node:
        logger.warning("no gallery service node is found")
        return
    image_server_endpoint = f"{gallery_service_node.host_url}/gallery-api/v1/images"

    exec_context = _find_execution_context_from_input_data(input_data)
    for images_key in ("images", "gifs", "video", "3d"):
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
            pnginfo = _find_extra_pnginfo_from_input_data(exec_context, input_data=input_data)

            if image_filename in proceeded_files:
                continue
            presigned_url = _do_post_image_to_gallery(
                image_server_endpoint,
                task_id,
                user_id,
                user_hash,
                image_type,
                image_subfolder,
                image_filename,
                exec_context.positive_prompt,
                pnginfo,
                exec_context.checkpoints_model_base,
                exec_context.loaded_model_ids
            )
            image["presigned_url"] = presigned_url
            proceeded_files.add(image_filename)

    if hasattr(node_obj, "RETURN_TYPES") and "VHS_FILENAMES" in node_obj.RETURN_TYPES:
        for node_result in result_data[node_obj.RETURN_TYPES.index("VHS_FILENAMES")]:
            if node_result[0]:
                for filepath in node_result[1]:
                    filename = pathlib.Path(filepath).name
                    if filename in proceeded_files:
                        continue
                    _do_post_image_to_gallery(
                        image_server_endpoint,
                        task_id,
                        user_id,
                        user_hash,
                        "output",
                        "",
                        filename,
                        exec_context.positive_prompt,
                        {},
                        exec_context.checkpoints_model_base,
                        exec_context.loaded_model_ids
                    )
                    proceeded_files.add(filename)


def _find_user_hash_from_input_data(input_data):
    if not isinstance(input_data, dict):
        return ""
    for key, value in input_data.items():
        if key == "user_hash":
            return value[0]
        elif key == "context":
            return value[0].user_hash
    return ""


class _DummyRequest:
    def __init__(self):
        self.headers: dict = {}


def _find_execution_context_from_input_data(input_data):
    import execution_context
    if not isinstance(input_data, dict):
        return None
    for key, value in input_data.items():
        if key == "context":
            return value[0]
    return execution_context.ExecutionContext(_DummyRequest())


def _find_extra_pnginfo_from_input_data(context: execution_context.ExecutionContext, input_data):
    if not isinstance(input_data, dict):
        return ""
    for key, value in input_data.items():
        if key == "extra_pnginfo" and value:
            pnginfo = value[0]
            if not pnginfo or not isinstance(pnginfo, dict):
                return {}
            pnginfo["parameters"] = context.geninfo if context else {}
            return pnginfo
    return {}
