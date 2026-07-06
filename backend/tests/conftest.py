# Shared test fixtures. Tests run against an in-memory SQLite database
# (not the real Postgres) so they're fast and need no external services —
# same philosophy as test_citation_validation.py's "no external services"
# comment, just extended to cover the API layer via FastAPI's TestClient.

import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import get_db
from app.models import Base

# A real (temp) file rather than sqlite:///:memory: — avoids in-memory
# SQLite's "each new connection is its own separate database" behavior,
# which otherwise requires pool tuning to work around. TestClient's async
# support runs requests on a different thread than the test body, so a
# shared file is the simplest way to guarantee every connection sees the
# same schema and data.
_TEST_DB_PATH = tempfile.mktemp(suffix=".db")
engine = create_engine(
    f"sqlite:///{_TEST_DB_PATH}",
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db


@pytest.fixture
def _fresh_database():
    """Every test gets a clean schema — no leftover users/papers between tests."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="session", autouse=True)
def _cleanup_test_db_file():
    yield
    engine.dispose()
    if os.path.exists(_TEST_DB_PATH):
        os.remove(_TEST_DB_PATH)


@pytest.fixture
def client(_fresh_database):
    # Explicitly depends on _fresh_database (rather than relying on
    # autouse ordering) so the schema is guaranteed to exist before
    # TestClient's lifespan/requests ever touch the database.
    with TestClient(app) as test_client:
        yield test_client
