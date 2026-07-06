# Entry point. Shows Login/Register forms when logged out, and a small
# dashboard with quick links when logged in. Every other page (under
# pages/) assumes auth.require_auth() has already gated access.

import streamlit as st

import api_client
from api_client import ApiError
from auth import init_session_state, is_logged_in

st.set_page_config(page_title="PaperPilot", page_icon="📄", layout="wide")
init_session_state()


def _login_form():
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", use_container_width=True)

    if submitted:
        try:
            api_client.login(email, password)
            st.session_state.user = api_client.get_me()
            st.rerun()
        except ApiError as e:
            st.error(e.message)


def _register_form():
    with st.form("register_form"):
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Password", type="password", key="register_password")
        submitted = st.form_submit_button("Create account", use_container_width=True)

    if submitted:
        try:
            api_client.register(email, password)
            st.success("Account created — log in from the Login tab.")
        except ApiError as e:
            st.error(e.message)


def _logged_out_view():
    st.title("📄 PaperPilot")
    st.caption("Multi-agent research literature review platform")

    login_tab, register_tab = st.tabs(["Log in", "Register"])
    with login_tab:
        _login_form()
    with register_tab:
        _register_form()


def _logged_in_view():
    st.title("📄 PaperPilot")
    st.write(f"Logged in as **{st.session_state.user['email']}**")

    st.markdown(
        """
        Use the sidebar to:
        - **Search** — find papers on arXiv and add them to your library
        - **Library** — browse your saved papers, embed them for semantic search
        - **Semantic Search** — find papers by meaning, not keywords
        - **Synthesis** — get a cited literature-review paragraph from your embedded papers
        """
    )

    if st.button("Log out"):
        api_client.logout()
        st.rerun()


if is_logged_in():
    _logged_in_view()
else:
    _logged_out_view()
