from sqlalchemy import Column, Integer, String

from diffus import database

FAVORITE_MODEL_TYPES = {
    'checkpoints': 'CHECKPOINT',
    'loras': 'LORA',
    'lycoris': 'LYCORIS',
}


class Model(database.Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String)
    base = Column(String)

    stem = Column(String)
    extension = Column(String, index=True)

    sha256 = Column(String, index=True)
    config_sha256 = Column(String)


class FavoriteModel(database.Base):
    __tablename__ = "favorite_models"

    id = Column(Integer, primary_key=True)
    favorited_by = Column(String, index=True)
    model_id = Column(Integer, index=True)
