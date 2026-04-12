from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection

from app.config import settings


@contextmanager
def get_db() -> Generator[Connection, None, None]:
    engine = create_engine(settings.DATABASE_URL)
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()
        engine.dispose()
