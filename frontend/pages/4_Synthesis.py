# The main PaperPilot feature: retrieve your embedded papers relevant to
# a query, then have an LLM (Groq/Llama 3 via LangChain) write a cited
# literature-review paragraph. Two modes:
#   - Simple: one-shot generation.
#   - Reflection loop: a LangGraph pipeline that checks its own citations
#     against the real papers and retries if it finds a hallucinated one.

import streamlit as st

import api_client
from api_client import ApiError, AuthExpiredError
from auth import require_auth

st.set_page_config(page_title="Synthesis · PaperPilot", page_icon="✍️", layout="wide")
require_auth()

st.title("✍️ Synthesize")
st.caption("Generates a cited literature-review paragraph from your embedded papers.")

with st.form("synthesis_form"):
    query = st.text_input("Topic", placeholder="e.g. attention mechanisms in transformers")
    col1, col2 = st.columns(2)
    limit = col1.number_input("Papers to use", min_value=1, max_value=10, value=5)
    mode = col2.radio(
        "Mode",
        ["Simple", "Reflection loop (self-checks citations)"],
        horizontal=True,
    )
    submitted = st.form_submit_button("Synthesize")

if submitted and query:
    use_graph = mode.startswith("Reflection")
    try:
        with st.spinner("Retrieving papers and generating synthesis... this can take a few seconds"):
            if use_graph:
                result = api_client.synthesize_graph(query, limit)
            else:
                result = api_client.synthesize(query, limit)
    except AuthExpiredError:
        st.warning("Your session expired. Please log in again from the Home page.")
        st.stop()
    except ApiError as e:
        if e.status_code == 404:
            st.info(e.message + "  \nGo to **Search** to find papers, then embed them.")
        else:
            st.error(e.message)
        result = None

    if result:
        st.markdown("### Synthesis")
        st.write(result["synthesis"])

        if use_graph:
            m1, m2, m3 = st.columns(3)
            m1.metric("Attempts", result["attempts"])
            m2.metric("Valid citations", len(result["valid_citations"]))
            m3.metric("Invalid citations", len(result["invalid_citations"]))

            if result["citations_valid"]:
                st.success("All citations checked out against the source papers.")
            else:
                st.error("Some citations could not be matched to a real source paper.")

            if result["valid_citations"]:
                st.markdown("**✓ Valid citations:** " + ", ".join(result["valid_citations"]))
            if result["invalid_citations"]:
                st.markdown("**✗ Invalid citations:** " + ", ".join(result["invalid_citations"]))

        with st.expander(f"Source papers ({len(result['based_on'])})"):
            for source in result["based_on"]:
                score = source.get("similarity_score")
                score_text = f" · similarity {score:.0%}" if score is not None else ""
                st.markdown(f"- **{source['title']}** ({source.get('year', 'n.d.')}){score_text}")
