# Step 11 — LangGraph: turning the single synthesis chain into a small
# multi-agent GRAPH with a supervisor, a worker, and a reflection node.
#
# Why bother? The plain chain in synthesis_agent.py (prompt | llm | parser)
# is a straight line: it always runs once and always returns whatever the
# LLM said, even if the LLM hallucinated a citation to a paper that was
# never given to it. A graph lets us add a LOOP: generate -> check the
# work -> if something's wrong, go generate again with feedback about
# exactly what was wrong -> check again -> ... until it's correct or we
# give up after a few tries.
#
# LangGraph concepts introduced here:
#
# 1. State — a TypedDict that flows through every node. Each node reads
#    from it and returns a dict of fields to UPDATE (LangGraph merges
#    that dict into the running state — nodes don't need to return the
#    whole state, just what changed).
#
# 2. Nodes — plain Python functions (sync or async) that take the state
#    and return a partial state update. We have three:
#       - supervisor: decides what should happen next (routing only,
#         doesn't touch the content of the state)
#       - synthesize: calls the LLM to (re)write the synthesis paragraph
#       - reflect: a *validation* node — no LLM call, just Python code
#         that checks whether every citation the LLM produced actually
#         matches one of the real papers we gave it. This is the
#         "reflection" pattern: a cheap, deterministic check that catches
#         hallucinated citations before they reach the user.
#
# 3. Conditional edges — after the supervisor node runs, LangGraph calls
#    a router function that inspects the state and returns the NAME of
#    the next node to run. This is how we express "if the citations are
#    bad and we haven't retried too many times, go back and try again;
#    otherwise stop."
#
# The resulting shape is a loop:
#
#       START -> supervisor -> synthesize -> reflect -> supervisor -> ...
#                    |
#                    +-> END (once citations check out, or retries run out)

import re
import difflib
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from prometheus_client import Counter

from app.agents.synthesis_agent import (
    GROQ_API_KEY,
    _synthesis_chain,
    format_papers_for_prompt,
)

# Custom metrics (exposed at GET /metrics alongside the generic HTTP
# metrics from main.py) — these are specific to the AI pipeline itself,
# not just "how many requests came in". They make the reflect node's
# work observable: how often the LLM actually hallucinates a citation,
# and how often retrying fixes it versus running out of attempts.
SYNTHESIS_ATTEMPTS = Counter(
    "paperpilot_synthesis_attempts_total",
    "Number of times the synthesize node has called the LLM "
    "(one graph run uses 1 attempt normally, more if it retries)",
)
CITATIONS_CHECKED = Counter(
    "paperpilot_citations_checked_total",
    "Citations checked by the reflect node",
    ["result"],  # "valid" or "invalid"
)
GRAPH_RUNS_FINISHED = Counter(
    "paperpilot_synthesis_graph_runs_total",
    "Completed synthesis graph runs, by how they ended",
    ["outcome"],  # "citations_valid" or "retries_exhausted"
)


class SynthesisState(TypedDict):
    query: str
    papers: list[dict]
    synthesis: str
    citations: list[str]
    valid_citations: list[str]
    invalid_citations: list[str]
    is_valid: bool
    feedback: Optional[str]
    attempts: int
    max_attempts: int


# ---------------------------------------------------------------------
# Node: synthesize — the "worker". Same LLM chain as the simple version,
# but on a retry it appends the reflection node's feedback to the query
# so the LLM knows exactly what it got wrong last time.
# ---------------------------------------------------------------------
async def synthesize_node(state: SynthesisState) -> dict:
    papers_text = format_papers_for_prompt(state["papers"])

    query = state["query"]
    if state.get("feedback"):
        query = (
            f"{query}\n\n"
            f"IMPORTANT — fix this before answering: {state['feedback']} "
            f"Only cite papers using their exact title as shown below, "
            f"inside square brackets, e.g. [Exact Paper Title]."
        )

    result = await _synthesis_chain.ainvoke({
        "query": query,
        "papers_text": papers_text,
    })
    SYNTHESIS_ATTEMPTS.inc()

    return {
        "synthesis": result,
        "attempts": state["attempts"] + 1,
    }


# ---------------------------------------------------------------------
# Node: reflect — the "validator". Pulls every [bracketed] citation out
# of the synthesis text and checks it against the real paper titles we
# retrieved. No LLM call here — this is plain string matching, which is
# both free and 100% reliable for catching an invented title.
#
# A citation counts as valid if it:
#   - exactly matches a real paper title, or
#   - refers to it positionally ("Paper 2", matching how the papers
#     were numbered in the prompt), or
#   - is a close fuzzy match (handles the LLM slightly truncating or
#     paraphrasing a long title)
# Anything else is flagged as a hallucinated citation.
# ---------------------------------------------------------------------
CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")
PAPER_N_PATTERN = re.compile(r"^paper\s+(\d+)$", re.IGNORECASE)


def extract_citations(text: str) -> list[str]:
    return [c.strip() for c in CITATION_PATTERN.findall(text)]


def validate_citations(
    citations: list[str], papers: list[dict]
) -> tuple[list[str], list[str]]:
    titles = [p.get("title", "") for p in papers]

    valid, invalid = [], []
    for citation in citations:
        if citation in titles:
            valid.append(citation)
            continue

        paper_n = PAPER_N_PATTERN.match(citation)
        if paper_n and 0 <= int(paper_n.group(1)) - 1 < len(titles):
            valid.append(citation)
            continue

        if difflib.get_close_matches(citation, titles, n=1, cutoff=0.6):
            valid.append(citation)
            continue

        invalid.append(citation)

    return valid, invalid


def reflect_node(state: SynthesisState) -> dict:
    citations = extract_citations(state["synthesis"])
    valid, invalid = validate_citations(citations, state["papers"])

    if valid:
        CITATIONS_CHECKED.labels(result="valid").inc(len(valid))
    if invalid:
        CITATIONS_CHECKED.labels(result="invalid").inc(len(invalid))

    is_valid = len(citations) > 0 and len(invalid) == 0

    feedback = None
    if invalid:
        feedback = (
            f"These citations don't match any real paper: {invalid}. "
            f"The real titles available are: {[p.get('title') for p in state['papers']]}."
        )
    elif not citations:
        feedback = (
            "No citations were found at all. Cite at least one paper "
            "by its exact title in [square brackets]."
        )

    return {
        "citations": citations,
        "valid_citations": valid,
        "invalid_citations": invalid,
        "is_valid": is_valid,
        "feedback": feedback,
    }


# ---------------------------------------------------------------------
# Node: supervisor — a no-op node whose only job is to be the hub that
# the conditional routing logic (route_from_supervisor, below) hangs
# off of. It doesn't change any state; it exists so the graph's control
# flow is explicit and visible rather than buried inside synthesize/
# reflect themselves.
# ---------------------------------------------------------------------
def supervisor_node(state: SynthesisState) -> dict:
    return {}


def route_from_supervisor(state: SynthesisState) -> str:
    """
    Runs after every supervisor visit and decides what happens next:

    - Nothing generated yet -> go generate it.
    - Generated, but citations are bad AND we still have retries left
      -> go generate again (synthesize_node will see `feedback` and
      correct itself).
    - Generated and valid, OR we've used up our retries -> stop.
    """
    if not state.get("synthesis"):
        return "synthesize"

    if state["is_valid"]:
        return "end"

    if state["attempts"] >= state["max_attempts"]:
        return "end"

    return "synthesize"


def build_synthesis_graph():
    graph = StateGraph(SynthesisState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("reflect", reflect_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"synthesize": "synthesize", "end": END},
    )
    graph.add_edge("synthesize", "reflect")
    graph.add_edge("reflect", "supervisor")

    return graph.compile()


# Compiled once at module load, same reasoning as _synthesis_chain in
# synthesis_agent.py — building the graph is cheap but pointless to redo
# on every request.
_synthesis_graph = build_synthesis_graph()


async def run_synthesis_graph(
    query: str, papers: list[dict], max_attempts: int = 2
) -> dict:
    """
    Entry point called by the API. Runs the full supervisor/synthesize/
    reflect loop and returns the final synthesis plus a citation report
    so the API response can be transparent about how much of the text
    is actually grounded in the retrieved papers.
    """
    if not papers:
        return {
            "synthesis": "No papers provided for synthesis.",
            "citations_valid": False,
            "valid_citations": [],
            "invalid_citations": [],
            "attempts": 0,
        }

    if not GROQ_API_KEY:
        return {
            "synthesis": (
                "Synthesis unavailable: GROQ_API_KEY not set in .env. "
                "Get a free key at https://console.groq.com"
            ),
            "citations_valid": False,
            "valid_citations": [],
            "invalid_citations": [],
            "attempts": 0,
        }

    initial_state: SynthesisState = {
        "query": query,
        "papers": papers,
        "synthesis": "",
        "citations": [],
        "valid_citations": [],
        "invalid_citations": [],
        "is_valid": False,
        "feedback": None,
        "attempts": 0,
        "max_attempts": max_attempts,
    }

    final_state = await _synthesis_graph.ainvoke(initial_state)

    GRAPH_RUNS_FINISHED.labels(
        outcome="citations_valid" if final_state["is_valid"] else "retries_exhausted"
    ).inc()

    return {
        "synthesis": final_state["synthesis"],
        "citations_valid": final_state["is_valid"],
        "valid_citations": final_state["valid_citations"],
        "invalid_citations": final_state["invalid_citations"],
        "attempts": final_state["attempts"],
    }
