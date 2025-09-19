import logging
import os
import pathlib
from typing import Literal

from pydantic import BaseModel
from sqlalchemy.orm import Session, Query

from diffus import constant, models
from diffus.database import gallery, comfy

logger = logging.getLogger(__name__)

MODEL_BINARY_CONTAINER = os.getenv('MODEL_BINARY_CONTAINER')
MODEL_CONFIG_CONTAINER = os.getenv('MODEL_CONFIG_CONTAINER')


def get_binary_path(sha256: str):
    sha256 = sha256.lower()
    return pathlib.Path(MODEL_BINARY_CONTAINER, sha256[0:2], sha256[2:4], sha256[4:6], sha256)


def get_config_path(sha256: str) -> pathlib.Path:
    sha256 = sha256.lower()
    return pathlib.Path(MODEL_CONFIG_CONTAINER, sha256)


class ModelInfo(BaseModel):
    id: int
    model_type: Literal["checkpoint", "embedding", "hypernetwork", "lora", "lycoris"]
    base: str | None
    stem: str
    extension: str
    sha256: str
    config_sha256: str | None

    @property
    def name(self):
        return f'{self.stem}.{self.extension}'

    @property
    def is_safetensors(self) -> bool:
        return self.extension in ("safetensors", "sft")

    @property
    def filename(self) -> str:
        return str(get_binary_path(self.sha256))

    def __str__(self):
        return self.filename


def create_model_info(record: models.Model) -> ModelInfo:
    return ModelInfo(
        id=record.id,
        model_type=str(record.model_type).lower(),
        base=record.base,
        stem=record.stem,
        extension=record.extension,
        sha256=record.sha256,
        config_sha256=record.config_sha256,
    )


class TaskInfo(BaseModel):
    number: int
    task_id: str
    user_id: str
    params: dict


def create_task_record(number: int, record: models.ComfyTaskRecord) -> TaskInfo | None:
    if not record:
        return None
    return TaskInfo(
        number=number,
        task_id=record.task_id,
        user_id=record.user_id,
        params=record.params,
    )


def list_favorite_model_by_model_type(user_id: str, folder_name: str, **kwargs):
    if folder_name not in constant.FAVORITE_MODEL_TYPES:
        return []
    model_type = constant.FAVORITE_MODEL_TYPES[folder_name]
    with gallery.Database() as session:
        model_base = kwargs.get('model_base', None)
        query = _make_favorite_model_query(session)
        query = _filter_favorite_model_by_model_type(query, user_id, model_type, model_base)
        return [create_model_info(ckpt).name for ckpt in session.scalars(query)]


def get_favorite_model_full_path(user_id: str, folder_name: str, filename: str) -> ModelInfo | None:
    if folder_name not in constant.FAVORITE_MODEL_TYPES:
        return None
    model_type = constant.FAVORITE_MODEL_TYPES[folder_name]
    with gallery.Database() as session:
        query = _make_favorite_model_query(session)
        query = _filter_favorite_model_by_model_type(query, user_id, model_type, None)
        query = _filter_model_by_name(query, filename)
        record = session.scalar(query)
        if not record:
            raise Exception(f"model is not found, [{folder_name}]/{filename}")
        return create_model_info(record)


def _make_favorite_model_query(session: Session) -> Query:
    return session.query(
        models.Model
    ).join(
        models.FavoriteModel,
        models.Model.id == models.FavoriteModel.model_id,
        isouter=True
    )


def _filter_model_by_sha256(query: Query, sha256: str) -> Query:
    return query.filter(models.Model.sha256 == sha256)


def _filter_model_by_name(query: Query, name: str) -> Query:
    filename = pathlib.Path(name)
    suffix = filename.suffix
    if suffix:
        suffix = suffix[1:]
    return query.filter(
        models.Model.stem == filename.stem,
        models.Model.extension == suffix,
    )


def _filter_favorite_model_by_model_type(query: Query, user_id: str, model_type: str, model_base) -> Query:
    query = query.filter(
        models.FavoriteModel.favorited_by == user_id
    )
    if model_type in {"LORA", "LYCORIS"}:
        query = query.filter(models.Model.model_type.in_(["LORA", "LYCORIS"]))
    else:
        query = query.filter(models.Model.model_type == model_type)
    if model_base:
        query = query.filter(models.Model.base == model_base)

    return query


def insert_comfy_task_record(
        task_id: str,
        user_id: str,
        params: dict,
) -> models.ComfyTaskRecord:
    # simplify params:
    try:
        number, prompt_id, prompt_dict, extra_data, outputs_to_execute = params["prompt"]
        if "extra_pnginfo" in extra_data and "workflow" in extra_data["extra_pnginfo"]:
            del extra_data["extra_pnginfo"]["workflow"]
        for node_param in prompt_dict.values():
            if node_param["class_type"] == "easy loadImageBase64":
                node_param["inputs"]["base64_data"] = ""
    except Exception as e:
        logger.warning(f"failed to simplify params for comfy task record '{task_id}': {e}")

    with comfy.Database() as session:
        record = models.ComfyTaskRecord(
            task_id=task_id,
            user_id=user_id,
            node=os.getenv('HOST_IP', default=''),
            params=params,
        )
        session.add(record)
        session.commit()
        return record


def get_comfy_task_record(task_id: str) -> TaskInfo | None:
    with comfy.Database() as session:
        query = session.query(
            models.ComfyTaskRecord
        ).where(
            models.ComfyTaskRecord.task_id == task_id
        )
        return create_task_record(0, session.scalar(query).one_or_none())


def list_comfy_task_record(user_id: str) -> list[TaskInfo]:
    with comfy.Database() as session:
        query = session.query(
            models.ComfyTaskRecord
        ).where(
            models.ComfyTaskRecord.user_id == user_id
        ).order_by(
            models.ComfyTaskRecord.id.desc()
        ).limit(
            10
        )
        return [
            create_task_record(number, record) for number, record in enumerate(session.scalars(query).all())
        ]


def delete_comfy_task_record(
        user_id: str,
        task_ids: list[str]
) -> None:
    if not (task_ids and user_id):
        return
    with comfy.Database() as session:
        session.query(
            models.ComfyTaskRecord
        ).filter(
            models.ComfyTaskRecord.user_id == user_id,
            models.ComfyTaskRecord.task_id.in_(task_ids),
        ).delete()
        session.commit()


def clear_comfy_task_record_for_user(
        user_id: str,
) -> None:
    if not user_id:
        return
    with comfy.Database() as session:
        session.query(
            models.ComfyTaskRecord
        ).filter(
            models.ComfyTaskRecord.user_id == user_id,
        ).delete()
        session.commit()
