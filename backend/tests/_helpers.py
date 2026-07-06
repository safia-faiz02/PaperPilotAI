# Plain helper functions for the API tests. Deliberately NOT named
# conftest.py and not a fixture module — pytest auto-loads every
# conftest.py it finds as a plugin under its own internal module name,
# so a test file doing `from tests.conftest import ...` would trigger a
# SECOND, separate import of conftest.py under the "tests.conftest"
# name — re-running its top-level code (including creating a whole
# second test database engine) and silently overwriting the dependency
# override with one pointed at an empty, never-migrated database. Kept
# here instead, as a normal one-way import, to avoid that trap.

from fastapi.testclient import TestClient


def register_and_login(client: TestClient, email: str, password: str = "testpass123") -> dict:
    """Registers a user and returns the login response body ({access_token, refresh_token, ...})."""
    client.post("/auth/register", json={"email": email, "password": password})
    response = client.post("/auth/login", data={"username": email, "password": password})
    return response.json()


def auth_headers_for(client: TestClient, email: str, password: str = "testpass123") -> dict:
    tokens = register_and_login(client, email, password)
    return {"Authorization": f"Bearer {tokens['access_token']}"}
