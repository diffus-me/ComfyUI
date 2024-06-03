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
    model_type: Literal["checkpoint", "embedding", "hypernetwork", "lora", "lycoris"]
    source: str | None
    name: str
    sha256: str
    config_sha256: str | None

    @property
    def name_for_extra(self) -> str:
        return os.path.splitext(self.name)[0]

    @property
    def model_name(self) -> str:
        return self.name_for_extra

    @property
    def title(self) -> str:
        return f"{self.name} [{self.shorthash}]"

    @property
    def short_title(self) -> str:
        return f"{self.name_for_extra} [{self.shorthash}]"

    @property
    def filename(self) -> str:
        return str(get_binary_path(self.sha256))

    @property
    def shorthash(self) -> str:
        return self.sha256[:10]

    @property
    def config_filename(self) -> str | None:
        if not self.config_sha256:
            return None

        return str(get_config_path(self.config_sha256))

    @property
    def is_safetensors(self) -> bool:
        return os.path.splitext(self.name)[-1].lower() == ".safetensors"

    def calculate_shorthash(self) -> str:
        return self.shorthash

    def check_file_existence(self) -> None:
        assert os.path.exists(self.filename), f"Model '{self.title}' does not exist"
        assert self.config_filename is None or os.path.exists(
            self.config_filename
        ), f"Config '{self.config_sha256}' for model '{self.title}' does not exist"

    def __str__(self):
        return self.name


def create_model_info(record: models.Model) -> ModelInfo:
    return ModelInfo(
        model_type=record.model_type,
        source=None,
        name=record.name,
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


def get_favorite_model_full_path(user_id: str, folder_name: str, name: str):
    if folder_name not in models.FAVORITE_MODEL_TYPES:
        return []
    model_type = models.FAVORITE_MODEL_TYPES[folder_name]
    with database.Database() as session:
        query = _make_favorite_model_query(session)
        query = _filter_favorite_model_by_model_type(query, user_id, model_type)
        query = _filter_model_by_name(query, name)
        return create_model_info(session.scalar(query))


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
    return query.filter(models.Model.name == name)


def _filter_favorite_model_by_model_type(query: Query, user_id: str, model_type: str) -> Query:
    query = query.filter(models.FavoriteModel.user_id == user_id)
    if model_type in {"lora", "lycoris"}:
        query = query.filter(models.Model.model_type.in_(["lora", "lycoris"]))
    else:
        query = query.filter(models.Model.model_type == model_type)

    return query
