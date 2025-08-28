import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

SQLALCHEMY_DATABASE_URL = os.getenv('SQL_DATABASE_URL_COMFY')

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={
    },
    pool_pre_ping=True,
    pool_recycle=3600
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Database:

    def __init__(self):
        self._db: Session | None = None

    def __enter__(self) -> Session:
        self._db = SessionLocal()
        return self._db

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._db.commit()
        self._db.close()
        self._db = None


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.commit()
        db.close()
