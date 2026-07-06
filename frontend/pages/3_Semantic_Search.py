# Search your embedded papers by MEANING rather than keywords — a
# vector similarity search over Qdrant, scoped to papers in your library.

import streamlit as st

import api_client
from api_client import ApiError, AuthExpiredError
from auth import require_auth

st.set_page_config(page_title="Semantic Search · PaperPilot", page_icon="🧠", layout="wide")
require_auth()

st.title("🧠 Semantic Search")
st.caption("Finds papers by meaning, not exact keywords — only searches papers you've embedded.")

with st.form("semantic_search_form"):
    col1, col2 = st.columns([4, 1])
    query = col1.text_input("What are you looking for?", placeholder="e.g. how do machines understand language")
    limit = col2.number_input("Results", min_value=1, max_value=20, value=5)
    submitted = st.form_submit_button("Search")

if submitted and query:
    try:
        with st.spinner("Searching..."):
            result = api_client.semantic_search(query, limit)
    except AuthExpiredError:
        st.warning("Your session expired. Please log in again from the Home page.")
        st.stop()
    except ApiError as e:
        st.error(e.message)
        result = None

    if result:
        if not result["results"]:
            st.info(result.get("message", "No results found."))
        else:
            for r in result["results"]:
                with st.container(border=True):
                    st.markdown(f"**{r['title']}**")
                    authors = ", ".join(r["authors"]) if r.get("authors") else "Unknown authors"
                    st.caption(f"{authors} · {r.get('year', 'n.d.')}")
                    st.progress(min(max(r["similarity_score"], 0.0), 1.0), text=f"Similarity: {r['similarity_score']:.0%}")
                    st.write(r["abstract_preview"])
