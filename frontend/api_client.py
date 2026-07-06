# A single thin wrapper around the PaperPilot backend's REST API. Every
# page imports functions from here rather than calling `requests` itself,
# so there's exactly one place that knows about base URLs, auth headers,
# and how to turn a failed response into a readable error message.
#
# Streamlit apps run entirely server-side (the whole script re-runs on
# every interaction), so these are plain Python-to-Python HTTP calls —
# there's no browser involved, unlike a JS-based frontend.

import requests
import streamlit as st

BASE_URL = st.secrets.get("api_base_url", "http://localhost:8000")


class ApiError(Exception):
    """Raised for any non-2xx response. `message` is the backend's own
    human-readable `detail` string, ready to show directly to the user."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


class AuthExpiredError(ApiError):
    """Raised when the access token is invalid/expired AND the refresh
    token couldn't get a new one either — the user needs to log in again."""


def _extract_detail(response: requests.Response) -> str:
    try:
        body = response.json()
        return body.get("detail", response.reason)
    except ValueError:
        return response.reason


def _auth_header() -> dict:
    token = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _try_refresh() -> bool:
    """Attempts to trade the stored refresh token for a new access token.
    Returns True on success (session_state is updated in place)."""
    refresh_token = st.session_state.get("refresh_token")
    if not refresh_token:
        return False

    response = requests.post(
        f"{BASE_URL}/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    if not response.ok:
        return False

    body = response.json()
    st.session_state.access_token = body["access_token"]
    st.session_state.refresh_token = body["refresh_token"]
    return True


def _request(method: str, path: str, retry_on_401: bool = True, **kwargs) -> dict:
    """
    Makes an authenticated request. On a 401 (expired access token), tries
    exactly once to silently refresh and retry — only raising
    AuthExpiredError (which pages treat as "show the login screen again")
    if that refresh attempt also fails.
    """
    response = requests.request(
        method,
        f"{BASE_URL}{path}",
        headers=_auth_header(),
        **kwargs,
    )

    if response.status_code == 401:
        if retry_on_401 and _try_refresh():
            return _request(method, path, retry_on_401=False, **kwargs)
        raise AuthExpiredError(401, "Your session expired. Please log in again.")

    if not response.ok:
        raise ApiError(response.status_code, _extract_detail(response))

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


# ── Auth ──────────────────────────────────────────────────────────────────────

def register(email: str, password: str) -> dict:
    response = requests.post(
        f"{BASE_URL}/auth/register",
        json={"email": email, "password": password},
    )
    if not response.ok:
        raise ApiError(response.status_code, _extract_detail(response))
    return response.json()


def login(email: str, password: str) -> None:
    """On success, stores both tokens in session_state directly (there's
    no caller-visible return value — callers should follow up with get_me())."""
    response = requests.post(
        f"{BASE_URL}/auth/login",
        data={"username": email, "password": password},
    )
    if not response.ok:
        raise ApiError(response.status_code, _extract_detail(response))

    body = response.json()
    st.session_state.access_token = body["access_token"]
    st.session_state.refresh_token = body["refresh_token"]


def logout() -> None:
    refresh_token = st.session_state.get("refresh_token")
    if refresh_token:
        try:
            requests.post(f"{BASE_URL}/auth/logout", json={"refresh_token": refresh_token})
        except requests.RequestException:
            pass  # best-effort — clearing local session state is what actually matters
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.user = None


def get_me() -> dict:
    return _request("GET", "/auth/me")


# ── Papers ────────────────────────────────────────────────────────────────────

def search_papers(query: str, max_results: int = 10) -> dict:
    return _request(
        "POST", "/papers/search",
        json={"query": query, "max_results": max_results},
    )


def list_papers(skip: int = 0, limit: int = 20) -> list:
    return _request("GET", "/papers/", params={"skip": skip, "limit": limit})


def embed_paper(paper_id: int) -> dict:
    return _request("POST", f"/papers/{paper_id}/embed")


def remove_paper(paper_id: int) -> None:
    """Removes a paper from your library only — the shared paper row and
    its Qdrant vector (if any) stay intact for other users."""
    _request("DELETE", f"/papers/{paper_id}")


def embed_all() -> dict:
    return _request("POST", "/papers/embed-all")


def semantic_search(query: str, limit: int = 5) -> dict:
    return _request("POST", "/papers/semantic-search", json={"query": query, "limit": limit})


def synthesize(query: str, limit: int = 5) -> dict:
    return _request("POST", "/papers/synthesize", json={"query": query, "limit": limit})


def synthesize_graph(query: str, limit: int = 5) -> dict:
    return _request("POST", "/papers/synthesize-graph", json={"query": query, "limit": limit})
