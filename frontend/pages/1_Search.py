# Search arXiv, save results to your library, and embed papers one at a
# time (or use the Library page's "Embed All" for everything at once).

import streamlit as st

import api_client
from api_client import ApiError, AuthExpiredError
from auth import require_auth

st.set_page_config(page_title="Search · PaperPilot", page_icon="🔍", layout="wide")
require_auth()

st.title("🔍 Search arXiv")

with st.form("search_form"):
    col1, col2 = st.columns([4, 1])
    query = col1.text_input("Search query", placeholder="e.g. transformer attention mechanisms")
    max_results = col2.number_input("Max results", min_value=1, max_value=50, value=10)
    submitted = st.form_submit_button("Search")

if submitted and query:
    try:
        with st.spinner("Searching arXiv..."):
            st.session_state.last_search = api_client.search_papers(query, max_results)
    except AuthExpiredError:
        st.warning("Your session expired. Please log in again from the Home page.")
        st.stop()
    except ApiError as e:
        st.error(e.message)
        st.session_state.last_search = None

result = st.session_state.get("last_search")

if result:
    st.caption(
        f"Found {result['total_found']} papers — "
        f"{result['new_papers']} new, {result['already_existed']} already in your library."
    )

    for paper in result["papers"]:
        with st.container(border=True):
            st.markdown(f"**{paper['title']}**")
            authors = ", ".join(paper["authors"]) if paper.get("authors") else "Unknown authors"
            st.caption(f"{authors} · {paper.get('year', 'n.d.')}")
            with st.expander("Abstract"):
                st.write(paper.get("abstract") or "No abstract available.")

            if paper.get("is_embedded"):
                st.button("✓ Embedded", key=f"embedded_{paper['id']}", disabled=True)
            else:
                if st.button("Embed", key=f"embed_{paper['id']}"):
                    try:
                        with st.spinner("Embedding..."):
                            api_client.embed_paper(paper["id"])
                        paper["is_embedded"] = True
                        st.rerun()
                    except AuthExpiredError:
                        st.warning("Your session expired. Please log in again from the Home page.")
                        st.stop()
                    except ApiError as e:
                        st.error(e.message)
