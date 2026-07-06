# Tests for the auth flow: register, login, /auth/me, and the refresh
# token lifecycle (issue, rotate-on-use, reject-after-rotation, revoke).

from tests._helpers import register_and_login, auth_headers_for


def test_register_creates_user_without_exposing_password(client):
    response = client.post(
        "/auth/register", json={"email": "alice@example.com", "password": "pass1234"}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "alice@example.com"
    assert "password" not in body
    assert "hashed_password" not in body


def test_register_rejects_duplicate_email(client):
    client.post("/auth/register", json={"email": "alice@example.com", "password": "pass1234"})
    response = client.post(
        "/auth/register", json={"email": "alice@example.com", "password": "different"}
    )
    assert response.status_code == 400


def test_login_returns_access_and_refresh_token(client):
    client.post("/auth/register", json={"email": "alice@example.com", "password": "pass1234"})
    response = client.post(
        "/auth/login", data={"username": "alice@example.com", "password": "pass1234"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["token_type"] == "bearer"


def test_login_rejects_wrong_password(client):
    client.post("/auth/register", json={"email": "alice@example.com", "password": "pass1234"})
    response = client.post(
        "/auth/login", data={"username": "alice@example.com", "password": "wrong"}
    )
    assert response.status_code == 401


def test_me_returns_current_user_with_valid_token(client):
    headers = auth_headers_for(client, "alice@example.com")
    response = client.get("/auth/me", headers=headers)
    assert response.status_code == 200
    assert response.json()["email"] == "alice@example.com"


def test_me_rejects_missing_token(client):
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_me_rejects_garbage_token(client):
    response = client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401


def test_refresh_issues_new_access_and_refresh_token(client):
    tokens = register_and_login(client, "bob@example.com")
    response = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert response.status_code == 200
    new_tokens = response.json()
    assert new_tokens["access_token"]
    # The refresh token is random per issuance, so it's guaranteed to differ
    # (rotation). The access token is a deterministic JWT of {sub, exp} with
    # second-level precision, so back-to-back calls within the same second
    # can legitimately produce an identical string — not asserted here.
    assert new_tokens["refresh_token"] != tokens["refresh_token"]


def test_refresh_token_rejected_after_rotation(client):
    """A refresh token can only be used once — reusing it (e.g. after theft) fails."""
    tokens = register_and_login(client, "carol@example.com")
    client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    response = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert response.status_code == 401


def test_refresh_rejects_unknown_token(client):
    response = client.post("/auth/refresh", json={"refresh_token": "not-a-real-refresh-token"})
    assert response.status_code == 401


def test_logout_revokes_refresh_token(client):
    tokens = register_and_login(client, "dave@example.com")
    logout_response = client.post("/auth/logout", json={"refresh_token": tokens["refresh_token"]})
    assert logout_response.status_code == 204

    refresh_response = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert refresh_response.status_code == 401
