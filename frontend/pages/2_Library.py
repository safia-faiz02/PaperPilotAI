# Browse your saved papers (papers you've searched for), paginated the
# same way the backend does it (skip/limit), with a bulk "Embed All"
# action alongside per-paper embedding.

import streamlit as st

import api_client
from api_client import ApiError, AuthExpiredError
from auth import require_auth

st.set_page_config(page_title="Library · PaperPilot", page_icon="📚", layout="wide")
require_auth()

st.title("📚 Your Library")

PAGE_SIZE = 20
if "library_skip" not in st.session_state:
    st.session_state.library_skip = 0

top_col1, top_col2 = st.columns([1, 5])
if top_col1.button("Embed All"):
    try:
        with st.spinner("Embedding every un-embedded paper in your library..."):
            summary = api_client.embed_all()
        st.success(
            f"Embedded {summary['embedded_count']} papers "
            f"({summary['failed_count']} failed, "
            f"{summary['skipped_no_abstract_count']} skipped — no abstract)."
        )
    except AuthExpiredError:
        st.warning("Your session expired. Please log in again from the Home page.")
        st.stop()
    except ApiError as e:
        st.error(e.message)

try:
    with st.spinner("Loading your library..."):
        papers = api_client.list_papers(skip=st.session_state.library_skip, limit=PAGE_SIZE)
except AuthExpiredError:
    st.warning("Your session expired. Please log in again from the Home page.")
    st.stop()
except ApiError as e:
    st.error(e.message)
    papers = []

if not papers and st.session_state.library_skip == 0:
    st.info("Your library is empty. Go to Search to find and save some papers.")
else:
    for paper in papers:
        with st.container(border=True):
            status = "✓ Embedded" if paper.get("is_embedded") else "Not embedded"
            st.markdown(f"**{paper['title']}** — _{status}_")
            authors = ", ".join(paper["authors"]) if paper.get("authors") else "Unknown authors"
            st.caption(f"{authors} · {paper.get('year', 'n.d.')} · {paper.get('venue') or paper['source']}")
            with st.expander("Abstract"):
                st.write(paper.get("abstract") or "No abstract available.")

            action_col1, action_col2 = st.columns([1, 1])
            if not paper.get("is_embedded"):
                if action_col1.button("Embed", key=f"lib_embed_{paper['id']}"):
                    try:
                        with st.spinner("Embedding..."):
                            api_client.embed_paper(paper["id"])
                        st.rerun()
                    except AuthExpiredError:
                        st.warning("Your session expired. Please log in again from the Home page.")
                        st.stop()
                    except ApiError as e:
                        st.error(e.message)

            if action_col2.button("Remove from library", key=f"lib_remove_{paper['id']}"):
                try:
                    api_client.remove_paper(paper["id"])
                    st.rerun()
                except AuthExpiredError:
                    st.warning("Your session expired. Please log in again from the Home page.")
                    st.stop()
                except ApiError as e:
                    st.error(e.message)

    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 4])
    if nav_col1.button("← Previous", disabled=st.session_state.library_skip == 0):
        st.session_state.library_skip = max(0, st.session_state.library_skip - PAGE_SIZE)
        st.rerun()
    if nav_col2.button("Next →", disabled=len(papers) < PAGE_SIZE):
        st.session_state.library_skip += PAGE_SIZE
        st.rerun()
