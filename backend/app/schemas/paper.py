from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class PaperSearchRequest(BaseModel):
    """
    What the client sends when they want to search for papers.
    We validate this automatically — if `query` is missing or
    `max_results` is not an integer, FastAPI rejects the request
    before our code runs.
    """
    query: str
    max_results: int = 10
    # Default of 10 means the client can omit this field and get
    # a sensible result without having to specify it every time.


class PaperOut(BaseModel):
    """
    The shape of a single paper in our API responses. Matches exactly
    what we store in the database, minus internal fields we don't
    need to expose (nothing sensitive here, unlike User).
    """
    id: int
    external_id: str
    source: str
    title: str
    authors: Optional[list] = None
    abstract: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    citation_count: Optional[int] = None
    is_embedded: bool = False
    ingested_at: datetime

    class Config:
        from_attributes = True


class PaperSearchResponse(BaseModel):
    """
    The full response from a search — a list of papers plus a count
    of how many were newly saved vs. already in the database. This
    is useful feedback: if you search the same topic twice, you'll
    see `new_papers: 0` the second time, telling you they're all
    already in your library.
    """
    query: str
    total_found: int
    new_papers: int
    already_existed: int
    papers: list[PaperOut]
