from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.models import User, Paper, LibraryEntry
from app.schemas.paper import PaperSearchRequest, PaperSearchResponse, PaperOut
from app.api.routes.auth import get_current_user
from app.integrations.arxiv_client import search_arxiv
from app.rag.vector_store import embed_paper, search_similar_papers
from app.agents.synthesis_agent import synthesize_papers
from app.agents.synthesis_graph import run_synthesis_graph

router = APIRouter()

# How many extra candidates to pull from Qdrant before filtering down to
# "papers in this user's library." Qdrant's vectors are shared/global (one
# per paper, not duplicated per user), so a query can surface papers other
# users embedded too — over-fetching and filtering in Python is a simple,
# pragmatic way to scope results without needing a native Qdrant payload
# filter. Fine at personal-tool scale (dozens-hundreds of papers); a much
# larger shared corpus would eventually want real Qdrant-side filtering.
LIBRARY_FILTER_OVERSAMPLE = 4


class SemanticSearchRequest(BaseModel):
    query: str
    limit: int = 5


class SynthesisRequest(BaseModel):
    query: str
    limit: int = 5
    # How many papers to retrieve and synthesise. Keep this low (3-7)
    # for best synthesis quality — too many papers overwhelm the LLM.


def _user_library_paper_ids(db: Session, user_id: int) -> set[int]:
    """All paper IDs currently in this user's library."""
    rows = db.query(LibraryEntry.paper_id).filter(LibraryEntry.user_id == user_id).all()
    return {row[0] for row in rows}


def _add_to_library(db: Session, user_id: int, paper_id: int) -> None:
    """Get-or-create a LibraryEntry — a paper can be in many users' libraries."""
    existing = (
        db.query(LibraryEntry)
        .filter(LibraryEntry.user_id == user_id, LibraryEntry.paper_id == paper_id)
        .first()
    )
    if not existing:
        db.add(LibraryEntry(user_id=user_id, paper_id=paper_id))


@router.post("/search", response_model=PaperSearchResponse)
async def search_papers(
    request: PaperSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Searches arXiv for papers matching the query, saves new ones to the
    shared papers table, adds them to the current user's library, and
    returns the full list.

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

    # Step 2: save new papers to the shared papers table, skip ones we
    # already have (by external_id — this is a "upsert" check, and it's
    # global, not per-user, so two users searching the same topic don't
    # create duplicate paper rows or duplicate embeddings).
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
            saved_papers.append(existing)
            existing_count += 1
        else:
            new_paper = Paper(**paper_data)
            db.add(new_paper)
            db.flush()
            # flush() sends the INSERT to Postgres immediately so
            # new_paper.id gets populated, but doesn't commit yet.
            saved_papers.append(new_paper)
            new_count += 1

    # Step 3: make sure every one of these papers is in THIS user's
    # library, regardless of whether the paper row itself was new or
    # already existed (someone else may have found it first).
    for paper in saved_papers:
        _add_to_library(db, current_user.id, paper.id)

    db.commit()

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
    Returns papers in the CURRENT USER's library, with pagination.
    Joins through library_entries so each user only sees what they've
    personally saved, even though the underlying papers table is shared.
    """
    papers = (
        db.query(Paper)
        .join(LibraryEntry, LibraryEntry.paper_id == Paper.id)
        .filter(LibraryEntry.user_id == current_user.id)
        .order_by(Paper.id)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return papers


@router.get("/{paper_id}", response_model=PaperOut)
def get_paper(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns a single paper by its internal database ID — but only if it's
    in the current user's library. A paper that exists but isn't in your
    library 404s the same as one that doesn't exist at all, so this
    endpoint never reveals what other users have saved.
    """
    paper = (
        db.query(Paper)
        .join(LibraryEntry, LibraryEntry.paper_id == Paper.id)
        .filter(Paper.id == paper_id, LibraryEntry.user_id == current_user.id)
        .first()
    )
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
    Generates and stores a vector embedding for a single paper in the
    current user's library. Once embedded, the paper becomes findable via
    POST /papers/semantic-search (for any user who has it in their library).
    """
    paper = (
        db.query(Paper)
        .join(LibraryEntry, LibraryEntry.paper_id == Paper.id)
        .filter(Paper.id == paper_id, LibraryEntry.user_id == current_user.id)
        .first()
    )
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
            detail="Embedding failed. Check the API logs for details.",
        )

    paper.is_embedded = True
    db.commit()

    return {
        "message": f"Paper '{paper.title[:60]}...' successfully embedded.",
        "paper_id": paper.id,
    }


@router.delete("/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_paper_from_library(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Removes a paper from the CURRENT USER's library only. The shared
    Paper row (and its Qdrant vector, if embedded) is left untouched —
    other users may still have it in their own library, and re-adding it
    later shouldn't require re-embedding.
    """
    entry = (
        db.query(LibraryEntry)
        .filter(LibraryEntry.user_id == current_user.id, LibraryEntry.paper_id == paper_id)
        .first()
    )
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper with id {paper_id} not found in your library.",
        )
    db.delete(entry)
    db.commit()
    return None


@router.post("/embed-all")
def embed_all_papers(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Embeds every not-yet-embedded paper in the current user's library that
    has an abstract. Lets the frontend offer a single "Embed All" action
    instead of clicking Embed on every paper one at a time.
    """
    unembedded = (
        db.query(Paper)
        .join(LibraryEntry, LibraryEntry.paper_id == Paper.id)
        .filter(LibraryEntry.user_id == current_user.id, Paper.is_embedded == False)  # noqa: E712
        .all()
    )

    to_embed = [p for p in unembedded if p.abstract]
    skipped_no_abstract_count = len(unembedded) - len(to_embed)

    embedded_count = 0
    failed_count = 0
    for paper in to_embed:
        success = embed_paper(
            paper_id=paper.id,
            external_id=paper.external_id,
            title=paper.title,
            abstract=paper.abstract,
        )
        if success:
            paper.is_embedded = True
            embedded_count += 1
        else:
            failed_count += 1

    db.commit()

    return {
        "embedded_count": embedded_count,
        "failed_count": failed_count,
        "skipped_no_abstract_count": skipped_no_abstract_count,
    }


@router.post("/semantic-search")
def semantic_search(
    request: SemanticSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Searches for papers by MEANING, not keywords, scoped to papers in the
    current user's library.

    1. Embeds your query using fastembed (local, no API call)
    2. Asks Qdrant "what paper vectors are closest to this query vector?"
       (over-fetching candidates, since Qdrant's vectors are shared across
       all users, not just this one)
    3. Filters those candidates down to papers in THIS user's library
    4. Returns them from Postgres, ranked by semantic similarity score
    """
    vector_results = search_similar_papers(
        query=request.query,
        limit=request.limit * LIBRARY_FILTER_OVERSAMPLE,
    )

    if not vector_results:
        return {
            "query": request.query,
            "message": "No embedded papers found. Try embedding some papers first via POST /papers/{id}/embed.",
            "results": [],
        }

    user_paper_ids = _user_library_paper_ids(db, current_user.id)
    vector_results = [r for r in vector_results if r["paper_id"] in user_paper_ids][: request.limit]

    if not vector_results:
        return {
            "query": request.query,
            "message": "No embedded papers found in your library. Try embedding some papers first via POST /papers/{id}/embed.",
            "results": [],
        }

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
    The main PaperPilot feature — combines semantic search (scoped to the
    current user's library) + LLM synthesis into a single endpoint.

    1. Embeds the query locally (fastembed, no API call)
    2. Searches Qdrant for the most similar paper vectors, over-fetching
       since vectors are shared across users
    3. Filters candidates down to papers in THIS user's library, then
       looks up full paper details from Postgres
    4. Sends papers + query to Groq (Llama 3) via LangChain
    5. Returns a cited synthesis paragraph + the source papers
    """
    vector_results = search_similar_papers(
        query=request.query,
        limit=request.limit * LIBRARY_FILTER_OVERSAMPLE,
    )

    user_paper_ids = _user_library_paper_ids(db, current_user.id)
    vector_results = [r for r in vector_results if r["paper_id"] in user_paper_ids][: request.limit]

    if not vector_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No embedded papers found in your library. "
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
    Same retrieval as /synthesize (scoped to the current user's library),
    but generation runs through the LangGraph pipeline
    (app/agents/synthesis_graph.py) instead of the single-shot chain.

    That pipeline is a small loop of three nodes:
      1. supervisor  — decides what happens next (generate vs. stop)
      2. synthesize  — writes the cited synthesis paragraph
      3. reflect     — checks every [citation] against the real papers;
                        no LLM call, just string matching

    If reflect finds a citation that doesn't match any real paper title,
    the supervisor routes back to synthesize with feedback describing
    exactly what was wrong, and it tries again (up to 2 attempts total)
    before giving up and returning whatever it has.
    """
    vector_results = search_similar_papers(
        query=request.query,
        limit=request.limit * LIBRARY_FILTER_OVERSAMPLE,
    )

    user_paper_ids = _user_library_paper_ids(db, current_user.id)
    vector_results = [r for r in vector_results if r["paper_id"] in user_paper_ids][: request.limit]

    if not vector_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No embedded papers found in your library. "
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
