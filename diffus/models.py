from sqlalchemy import Column, Integer, String

from diffus import database

FAVORITE_MODEL_TYPES = {
    'loras': 'lora',
    'checkpoints': 'checkpoint'
}


class Model(database.Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String)

    name = Column(String)
    filename = Column(String)
    name = Column(String, index=True)
    model_name = Column(String, index=True)
    name_for_extra = Column(String)

    hash = Column(String, index=True)
    sha256 = Column(String, index=True)
    config_sha256 = Column(String)


class FavoriteModel(database.Base):
    __tablename__ = "favorite_models"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    model_id = Column(Integer, index=True)


