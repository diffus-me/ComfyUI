import os
import pathlib
from typing import Literal

from pydantic import BaseModel
from sqlalchemy.orm import Session, Query

from diffus import models, database

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


def list_favorite_model_by_model_type(user_id: str, folder_name: str):
    if folder_name not in models.FAVORITE_MODEL_TYPES:
        return []
    model_type = models.FAVORITE_MODEL_TYPES[folder_name]
    with database.Database() as session:
        query = _make_favorite_model_query(session)
        query = _filter_favorite_model_by_model_type(query, user_id, model_type)
        return [create_model_info(ckpt).name for ckpt in session.scalars(query)]


def get_favorite_model_full_path(user_id: str, folder_name: str, filename: str):
    if folder_name not in models.FAVORITE_MODEL_TYPES:
        return None
    model_type = models.FAVORITE_MODEL_TYPES[folder_name]
    with database.Database() as session:
        query = _make_favorite_model_query(session)
        query = _filter_favorite_model_by_model_type(query, user_id, model_type)
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


def _filter_favorite_model_by_model_type(query: Query, user_id: str, model_type: str) -> Query:
    query = query.filter(
        models.FavoriteModel.favorited_by == user_id
    )
    if model_type in {"LORA", "LYCORIS"}:
        query = query.filter(models.Model.model_type.in_(["LORA", "LYCORIS"]))
    else:
        query = query.filter(models.Model.model_type == model_type)

    return query