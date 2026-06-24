import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import logging

SQLALCHEMY_DATABASE_URL = os.getenv('SQL_DATABASE_URL')
logger = logging.getLogger(__name__)
if SQLALCHEMY_DATABASE_URL:
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
else:
    class Base:
        pass

    class Database:

        def __init__(self):
            raise NotImplementedError(f"SQL_DATABASE_URL is not defined")

        def __enter__(self) -> Session:
            raise NotImplementedError(f"SQL_DATABASE_URL is not defined")

        def __exit__(self, exc_type, exc_val, exc_tb):
            raise NotImplementedError(f"SQL_DATABASE_URL is not defined")
