from sqlalchemy import Column, Integer, String, JSON

from diffus.database import gallery, comfy

FAVORITE_MODEL_TYPES = {
    'checkpoints': 'CHECKPOINT',
    'loras': 'LORA',
    'lycoris': 'LYCORIS',
}


class Model(gallery.Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String)
    base = Column(String)

    stem = Column(String)
    extension = Column(String, index=True)

    sha256 = Column(String, index=True)
    config_sha256 = Column(String)


class FavoriteModel(gallery.Base):
    __tablename__ = "favorite_models"

    id = Column(Integer, primary_key=True)
    favorited_by = Column(String, index=True)
    model_id = Column(Integer, index=True)


class ComfyTaskRecord(comfy.Base):
    __tablename__ = "comfy_task_records"

    id = Column(Integer, primary_key=True, nullable=False, unique=True, autoincrement=True)
    task_id = Column(String(64), index=True, nullable=False)
    user_id = Column(String(64), index=True, nullable=False)
    params = Column(JSON, nullable=False)
