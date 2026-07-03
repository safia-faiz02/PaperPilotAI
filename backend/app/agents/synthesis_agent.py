# This is our first real AI agent — the synthesis agent.
#
# What it does:
# Takes a list of papers (title + abstract) that were retrieved via
# semantic search, and uses an LLM to write a coherent, cited literature
# review paragraph summarising what those papers collectively say.
#
# How it's built (LangChain concepts introduced here):
#
# 1. ChatPromptTemplate — a reusable, versioned prompt with variables
#    (like {query} and {papers}) that get filled in at runtime.
#
# 2. ChatGroq — LangChain's wrapper around the Groq API. Swap this one
#    line for ChatOllama or ChatOpenAI to use a different LLM provider.
#
# 3. StrOutputParser — extracts just the text string from the LLM's
#    response object (LLMs return a structured Message, not a plain string).
#
# 4. LCEL pipe syntax (prompt | llm | parser) — chains these three steps
#    together. When you call .invoke(), it runs left to right:
#    fill prompt → send to LLM → parse output → return string.
#    This is LangChain's core composition pattern.

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# The model we're using. llama-3.1-8b-instant is Groq's fastest free
# model — good quality for summarization tasks, ~0.5s response time.
# Other free options on Groq: llama-3.3-70b-versatile (smarter, slower)
GROQ_MODEL = "llama-3.1-8b-instant"


def build_synthesis_chain():
    """
    Builds and returns a LangChain chain for literature synthesis.

    A "chain" in LangChain is a sequence of steps that process data
    together. This one has three steps chained with the | pipe operator:

        prompt | llm | parser

    - prompt: fills in the template with our actual papers and query
    - llm: sends the filled prompt to Groq and gets a response
    - parser: extracts the plain text string from the response

    We build the chain once and reuse it for every synthesis request —
    creating an LLM client on every request would be wasteful.
    """
    # The prompt template. {query} and {papers_text} are placeholders
    # that get filled in when we call chain.invoke({...}).
    # The system message sets the LLM's role and constraints.
    # The human message is the actual task.
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert research assistant helping scientists \
understand academic literature. Your job is to synthesize multiple paper \
abstracts into a coherent, well-structured paragraph that:
- Identifies the main themes and findings across the papers
- Notes where papers agree, build on each other, or differ
- Cites papers by their title in square brackets like [Title of Paper]
- Uses formal academic language
- Is 150-250 words long

Do not invent facts. Only use information from the provided abstracts."""),

        ("human", """Research query: {query}

Here are the relevant papers found:

{papers_text}

Please write a synthesis paragraph covering what these papers collectively \
say about the research query."""),
    ])

    llm = ChatGroq(
        # ChatGroq treats an empty string as "no key" and falls back to
        # checking the GROQ_API_KEY env var itself — if that's ALSO
        # unset, the underlying groq client raises immediately here,
        # crashing the whole app at import time. "not-set" is a harmless
        # placeholder that satisfies the constructor; synthesize_papers()
        # below already checks GROQ_API_KEY before ever calling this
        # chain, so a placeholder key is never actually used to call Groq.
        api_key=GROQ_API_KEY or "not-set",
        model=GROQ_MODEL,
        temperature=0.3,
        # temperature=0.3 means "mostly deterministic" — we want
        # factual synthesis, not creative writing. Lower = more
        # consistent/predictable, higher = more varied/creative.
    )

    parser = StrOutputParser()
    # StrOutputParser just extracts .content from the AIMessage the LLM
    # returns — giving us a plain Python string we can return in the API.

    # The chain: fill prompt → call LLM → parse to string
    return prompt | llm | parser


def format_papers_for_prompt(papers: list[dict]) -> str:
    """
    Formats a list of paper dicts into a readable text block for the prompt.
    Each paper gets a numbered entry with title, authors, year, and abstract.
    Clear formatting helps the LLM correctly attribute findings to papers.
    """
    sections = []
    for i, paper in enumerate(papers, 1):
        authors = ", ".join(paper.get("authors") or []) or "Unknown authors"
        year = paper.get("year") or "n.d."
        title = paper.get("title", "Untitled")
        abstract = paper.get("abstract") or paper.get("abstract_preview", "")

        sections.append(
            f"[Paper {i}] {title}\n"
            f"Authors: {authors} ({year})\n"
            f"Abstract: {abstract}\n"
        )

    return "\n---\n".join(sections)


# Build the chain once when the module loads — reused for every request.
# If GROQ_API_KEY is missing this will still build fine; it only fails
# when .invoke() is actually called, giving a clear error at that point.
_synthesis_chain = build_synthesis_chain()


async def synthesize_papers(query: str, papers: list[dict]) -> str:
    """
    Main function called by the API endpoint. Takes the search query
    and a list of paper dicts, returns a synthesized paragraph string.

    Uses .ainvoke() (async) instead of .invoke() (sync) so FastAPI can
    handle other requests while waiting for Groq to respond — the same
    reason we used `async def` for the arXiv search endpoint.
    """
    if not papers:
        return "No papers provided for synthesis."

    if not GROQ_API_KEY:
        return (
            "Synthesis unavailable: GROQ_API_KEY not set in .env. "
            "Get a free key at https://console.groq.com"
        )

    papers_text = format_papers_for_prompt(papers)

    # .ainvoke() runs the full chain asynchronously:
    # fill prompt → await Groq API response → parse to string
    result = await _synthesis_chain.ainvoke({
        "query": query,
        "papers_text": papers_text,
    })

    return result
