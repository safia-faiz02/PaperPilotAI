# Session-state helpers shared by every page. Streamlit has no built-in
# concept of "protected routes" — every page in pages/ is always reachable
# from the sidebar — so each page calls require_auth() at the top as the
# equivalent of a route guard.

import streamlit as st

import api_client
from api_client import ApiError, AuthExpiredError


def init_session_state() -> None:
    for key in ("access_token", "refresh_token", "user"):
        if key not in st.session_state:
            st.session_state[key] = None


def is_logged_in() -> bool:
    return bool(st.session_state.get("access_token") and st.session_state.get("user"))


def require_auth() -> dict:
    """
    Call at the top of every page under pages/. Renders a message and
    halts the script (st.stop()) if there's no logged-in user, so nothing
    below it ever runs. Returns the current user dict when logged in.
    """
    init_session_state()

    if not st.session_state.get("access_token"):
        st.info("Please log in from the Home page to use PaperPilot.")
        st.stop()

    if st.session_state.get("user") is None:
        # We have a token but haven't validated it yet this session (e.g.
        # the app was just restarted) — confirm it's still good.
        try:
            st.session_state.user = api_client.get_me()
        except AuthExpiredError:
            st.session_state.access_token = None
            st.session_state.refresh_token = None
            st.warning("Your session expired. Please log in again from the Home page.")
            st.stop()
        except ApiError as e:
            st.error(f"Couldn't verify your login: {e.message}")
            st.stop()

    return st.session_state.user
