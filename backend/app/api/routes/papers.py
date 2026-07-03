from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.models import User, Paper
from app.schemas.paper import PaperSearchRequest, PaperSearchResponse, PaperOut
from app.api.routes.auth import get_current_user
from app.integrations.arxiv_client import search_arxiv
from app.rag.vector_store import embed_paper, search_similar_papers
from app.agents.synthesis_agent import synthesize_papers
from app.agents.synthesis_graph import run_synthesis_graph

router = APIRouter()


class SemanticSearchRequest(BaseModel):
    query: str
    limit: int = 5


class SynthesisRequest(BaseModel):
    query: str
    limit: int = 5
    # How many papers to retrieve and synthesise. Keep this low (3-7)
    # for best synthesis quality — too many papers overwhelm the LLM.


@router.post("/search", response_model=PaperSearchResponse)
async def search_papers(
    request: PaperSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Searches arXiv for papers matching the query, saves new ones to
    the database, and returns the full list.

    This endpoint is `async` (notice `async def` instead of just `def`)
    because it calls `await search_arxiv(...)` — an async function that
    makes a real network request. Any endpoint that calls async functions
    must itself be async. Our auth and DB endpoints above were regular
    `def` because they didn't need async; this one does.

    Protected: requires a valid JWT token (via get_current_user).
    """

    # Step 1: hit the arXiv API and get raw results
    try:
        arxiv_results = await search_arxiv(
            query=request.query,
            max_results=request.max_results,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"arXiv API is currently unavailable: {str(e)}",
        )

    if not arxiv_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No papers found for query: '{request.query}'",
        )

    # Step 2: save new papers to the database, skip ones we already have.
    # This is called "upsert" logic (update-or-insert). We check by
    # external_id — if a paper with that arXiv ID is already in our DB,
    # we skip it rather than saving a duplicate.
    saved_papers = []
    new_count = 0
    existing_count = 0

    for paper_data in arxiv_results:
        existing = (
            db.query(Paper)
            .filter(Paper.external_id == paper_data["external_id"])
            .first()
        )

        if existing:
            # Already in the database — just collect it for the response
            saved_papers.append(existing)
            existing_count += 1
        else:
            # Brand new paper — create a database row for it
            new_paper = Paper(**paper_data)
            db.add(new_paper)
            db.flush()
            # flush() sends the INSERT to Postgres immediately so
            # new_paper.id gets populated, but doesn't commit yet.
            # We commit once after the loop (one transaction for all
            # inserts is faster than committing inside the loop).
            saved_papers.append(new_paper)
            new_count += 1

    db.commit()

    # refresh each new paper so SQLAlchemy fills in server-generated
    # fields like `ingested_at` from the database.
    for paper in saved_papers:
        db.refresh(paper)

    return PaperSearchResponse(
        query=request.query,
        total_found=len(saved_papers),
        new_papers=new_count,
        already_existed=existing_count,
        papers=saved_papers,
    )


@router.get("/", response_model=list[PaperOut])
def list_papers(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns all papers saved in the database so far, with pagination.

    `skip` and `limit` are query parameters — the client calls:
    GET /papers/?skip=0&limit=20   (first page)
    GET /papers/?skip=20&limit=20  (second page)

    This pattern is called "offset pagination" and is the simplest
    kind — we'll improve it later if we need cursor-based pagination
    for very large datasets.
    """
    papers = db.query(Paper).offset(skip).limit(limit).all()
    return papers


@router.get("/{paper_id}", response_model=PaperOut)
def get_paper(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns a single paper by its internal database ID.
    The {paper_id} in the URL is a "path parameter" — FastAPI
    extracts it automatically and passes it to the function.
    """
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper with id {paper_id} not found.",
        )
    return paper


@router.post("/{paper_id}/embed")
def embed_single_paper(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generates and stores a vector embedding for a single paper.

    Call this after searching — it takes a paper already in Postgres,
    sends its title + abstract to OpenAI's embedding API, and stores
    the resulting vector in Qdrant. Once embedded, the paper becomes
    findable via POST /papers/semantic-search.

    In a later step, Celery will do this automatically in the background
    whenever a new paper is saved — for now we call it manually.
    """
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper with id {paper_id} not found.",
        )

    if not paper.abstract:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This paper has no abstract to embed.",
        )

    success = embed_paper(
        paper_id=paper.id,
        external_id=paper.external_id,
        title=paper.title,
        abstract=paper.abstract,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding failed. Check your OPENAI_API_KEY in .env.",
        )

    return {
        "message": f"Paper '{paper.title[:60]}...' successfully embedded.",
        "paper_id": paper.id,
    }


@router.post("/semantic-search")
def semantic_search(
    request: SemanticSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Searches for papers by MEANING, not keywords.

    This is fundamentally different from POST /papers/search (which
    hits the arXiv API with a keyword query). This endpoint:
    1. Embeds your query using OpenAI
    2. Asks Qdrant "what paper vectors are closest to this query vector?"
    3. Returns those papers from our Postgres database, ranked by
       semantic similarity score (0-1, higher = more similar)

    Example: searching "how do machines understand language" will find
    papers about NLP even if they never use those exact words.
    """
    vector_results = search_similar_papers(
        query=request.query,
        limit=request.limit,
    )

    if not vector_results:
        return {
            "query": request.query,
            "message": "No embedded papers found. Try embedding some papers first via POST /papers/{id}/embed.",
            "results": [],
        }

    # Look up the full paper details from Postgres using the IDs
    # Qdrant returned. Qdrant only stores vectors + minimal metadata;
    # the full paper data (authors, year, etc.) lives in Postgres.
    paper_ids = [r["paper_id"] for r in vector_results]
    papers_map = {
        p.id: p
        for p in db.query(Paper).filter(Paper.id.in_(paper_ids)).all()
    }

    results = []
    for vr in vector_results:
        paper = papers_map.get(vr["paper_id"])
        if paper:
            results.append({
                "similarity_score": round(vr["score"], 4),
                "paper_id": paper.id,
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
                "abstract_preview": (paper.abstract or "")[:200] + "...",
            })

    return {
        "query": request.query,
        "results": results,
    }


@router.post("/synthesize")
async def synthesize(
    request: SynthesisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    The main PaperPilot feature — combines semantic search + LLM synthesis
    into a single endpoint.

    What happens under the hood:
    1. Embeds the query locally (fastembed, no API call)
    2. Searches Qdrant for the most similar paper vectors
    3. Looks up full paper details from Postgres
    4. Sends papers + query to Groq (Llama 3) via LangChain
    5. Returns a cited synthesis paragraph + the source papers

    This is the complete RAG loop:
    Retrieve (steps 1-3) → Augment prompt (step 4) → Generate (step 5)
    """
    # Step 1 & 2: semantic search over embedded papers
    vector_results = search_similar_papers(
        query=request.query,
        limit=request.limit,
    )

    if not vector_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No embedded papers found. "
                "Search for papers first (POST /papers/search), "
                "then embed them (POST /papers/{id}/embed)."
            ),
        )

    # Step 3: fetch full paper details from Postgres
    paper_ids = [r["paper_id"] for r in vector_results]
    papers = db.query(Paper).filter(Paper.id.in_(paper_ids)).all()

    # Build the list of paper dicts the synthesis agent expects
    papers_for_synthesis = [
        {
            "title": p.title,
            "authors": p.authors,
            "year": p.year,
            "abstract": p.abstract,
        }
        for p in papers
    ]

    # Steps 4 & 5: LangChain synthesis via Groq
    synthesis = await synthesize_papers(
        query=request.query,
        papers=papers_for_synthesis,
    )

    return {
        "query": request.query,
        "synthesis": synthesis,
        "based_on": [
            {
                "paper_id": p.id,
                "title": p.title,
                "year": p.year,
                "similarity_score": next(
                    (r["score"] for r in vector_results if r["paper_id"] == p.id),
                    None,
                ),
            }
            for p in papers
        ],
    }


@router.post("/synthesize-graph")
async def synthesize_with_graph(
    request: SynthesisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Same retrieval as /synthesize, but generation runs through the
    LangGraph pipeline (app/agents/synthesis_graph.py) instead of the
    single-shot chain.

    That pipeline is a small loop of three nodes:
      1. supervisor  — decides what happens next (generate vs. stop)
      2. synthesize  — writes the cited synthesis paragraph
      3. reflect     — checks every [citation] against the real papers;
                        no LLM call, just string matching

    If reflect finds a citation that doesn't match any real paper title,
    the supervisor routes back to synthesize with feedback describing
    exactly what was wrong, and it tries again (up to 2 attempts total)
    before giving up and returning whatever it has.

    The response includes `citations_valid` and the valid/invalid
    citation lists so the caller can see whether the graph's self-check
    actually passed, rather than just trusting the LLM's output blindly.
    """
    vector_results = search_similar_papers(
        query=request.query,
        limit=request.limit,
    )

    if not vector_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No embedded papers found. "
                "Search for papers first (POST /papers/search), "
                "then embed them (POST /papers/{id}/embed)."
            ),
        )

    paper_ids = [r["paper_id"] for r in vector_results]
    papers = db.query(Paper).filter(Paper.id.in_(paper_ids)).all()

    papers_for_synthesis = [
        {
            "title": p.title,
            "authors": p.authors,
            "year": p.year,
            "abstract": p.abstract,
        }
        for p in papers
    ]

    result = await run_synthesis_graph(
        query=request.query,
        papers=papers_for_synthesis,
    )

    return {
        "query": request.query,
        "synthesis": result["synthesis"],
        "citations_valid": result["citations_valid"],
        "valid_citations": result["valid_citations"],
        "invalid_citations": result["invalid_citations"],
        "attempts": result["attempts"],
        "based_on": [
            {
                "paper_id": p.id,
                "title": p.title,
                "year": p.year,
                "similarity_score": next(
                    (r["score"] for r in vector_results if r["paper_id"] == p.id),
                    None,
                ),
            }
            for p in papers
        ],
    }
