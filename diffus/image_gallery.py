import json
import logging
import os
import pathlib
import uuid
from typing import Iterable

import requests

import execution_context
import folder_paths

logger = logging.getLogger(__name__)

from json import JSONEncoder


class _MyEncoder(JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__json__'):
            return obj.__json__()
        else:
            return super().default(obj)


def _do_post_image_to_gallery(
        post_url,
        api_secret,
        task_id,
        user_id,
        user_tier,
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

    relative_file_path = str(os.path.join(
        folder_paths.get_relative_output_directory(user_hash),
        image_subfolder,
        image_filename,
    ))
    jfs_volume = os.getenv("JFS_VOLUME_NAME", "")
    jfs_token = os.getenv("JFS_TOKEN", "")
    if jfs_volume and jfs_token:
        import juicefs
        import shutil
        ak = os.getenv("JFS_ACCESS_KEY", "")
        sk = os.getenv("JFS_SECRET_KEY", "")
        jfs = juicefs.Client(
            name=jfs_volume,
            token=jfs_token,
            access_key=ak,
            secret_key=sk
        )
        fsrc_path = os.path.join(
            folder_paths.get_output_directory(user_hash),
            image_subfolder,
            image_filename,
        )
        fdst_path = pathlib.Path("/workdir", relative_file_path)
        fdst_path_parent = str(fdst_path.parent)
        if not jfs.exists(fdst_path_parent):
            jfs.makedirs(fdst_path_parent, exist_ok=True)
        with open(fsrc_path, "rb") as fsrc:
            with jfs.open(
                    path=str(fdst_path),
                    mode="wb"
            ) as fdst:
                shutil.copyfileobj(fsrc, fdst)

    post_json = {
        "task_id": task_id,
        "path": relative_file_path,
        "feature": "COMFYUI",
        "pnginfo": json.dumps(pnginfo, cls=_MyEncoder),
        "base": model_base,
        "prompt": positive_prompt,  # positive prompt
        "model_ids": model_ids,  # checkpoint, loras
    }
    try:
        resp = requests.post(
            url=post_url,
            timeout=5,
            headers={
                "user-id": user_id,
                "user-tier": user_tier,
                "Api-Secret": api_secret,
            },
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


def post_output_to_image_gallery(redis_client, node_obj, header_dict, input_data, output_data, hidden_inputs=None):
    if not output_data:
        return

    user_hash = header_dict.get('x-diffus-user-hash', None)
    if not user_hash:
        logger.warning("post_output_to_image_gallery, no user_hash, returning")
        return

    user_id = header_dict.get('user-id', None)
    if not user_id:
        logger.warning("post_output_to_image_gallery, no user_id, returning")
        return
    user_tier = header_dict.get('user-tier', None) or header_dict.get('User-Tier', None)

    disable_post = header_dict.get('x-disable-gallery-post', None)
    if disable_post and disable_post.lower() == "true":
        logger.warning(f"post result to gallery is disabled, returning")
        return

    result_data, ui_data, _, _ = output_data

    if not isinstance(ui_data, dict):
        logger.warning("post_output_to_image_gallery, no ui_data, returning")
        return

    proceeded_files = set()

    task_id = header_dict.get('x-task-id', str(uuid.uuid4()))

    gallery_endpoint = header_dict.get('x-diffus-gallery-url', None)
    gallery_secret = header_dict.get('x-diffus-gallery-secret', None)

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
                post_url=gallery_endpoint,
                api_secret=gallery_secret,
                task_id=task_id,
                user_id=user_id,
                user_tier=user_tier,
                user_hash=user_hash,
                image_type=image_type,
                image_subfolder=image_subfolder,
                image_filename=image_filename,
                positive_prompt=exec_context.positive_prompt,
                pnginfo=pnginfo,
                model_base=exec_context.checkpoints_model_base,
                model_ids=exec_context.loaded_model_ids
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
                        post_url=gallery_endpoint,
                        api_secret=gallery_secret,
                        task_id=task_id,
                        user_id=user_id,
                        user_hash=user_hash,
                        user_tier=user_tier,
                        image_type="output",
                        image_subfolder="",
                        image_filename=filename,
                        positive_prompt=exec_context.positive_prompt,
                        pnginfo={},
                        model_base=exec_context.checkpoints_model_base,
                        model_ids=exec_context.loaded_model_ids
                    )
                    proceeded_files.add(filename)


def _find_user_hash_from_input_data(input_data, hidden_inputs=None):
    if not isinstance(input_data, dict) and hidden_inputs is None:
        return ""

    if isinstance(input_data, dict):
        for key, value in input_data.items():
            if key == "user_hash":
                return value[0]
            elif key == "context":
                return value[0].user_hash
    if hidden_inputs is not None:
        from comfy_api.latest import io
        context = hidden_inputs.get(io.Hidden.exec_context, None)
        if context:
            return context.user_hash
        return hidden_inputs.get(io.Hidden.user_hash, None)

    return ""


class _DummyRequest:
    def __init__(self):
        self.headers: dict = {}


def _find_execution_context_from_input_data(input_data) -> execution_context.ExecutionContext | None:
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
