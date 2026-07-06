from sqlalchemy import Column, Integer, String, DateTime, JSON, Boolean
from sqlalchemy.sql import func
from sqlalchemy.sql.expression import false

from app.models.base import Base


class Paper(Base):
    """
    One row per research paper we've ingested from arXiv or Semantic
    Scholar. This is the table the Discovery and Ingestion agents (built
    in a later step) will read from and write to.
    """

    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)

    external_id = Column(String, unique=True, index=True, nullable=False)
    # external_id -> the paper's own ID from arXiv/Semantic Scholar (e.g.
    # "2301.12345"). unique=True means our database itself will refuse to
    # let us accidentally store the same paper twice — a real, useful
    # safety net once we have an agent fetching papers automatically.

    source = Column(String, nullable=False)
    # source -> "arxiv" or "semantic_scholar", so we always know where a
    # given paper's data came from.

    title = Column(String, nullable=False)

    authors = Column(JSON, nullable=True)
    # Stored as JSON (a list of names) rather than a separate "authors"
    # table — simpler for now since we don't yet need to query things
    # like "find every paper by this specific author". We can split this
    # into its own table later if that need comes up; that's a normal,
    # expected kind of schema change, which is exactly what Alembic
    # migrations make safe to do.

    year = Column(Integer, nullable=True)
    venue = Column(String, nullable=True)
    citation_count = Column(Integer, nullable=True)
    abstract = Column(String, nullable=True)

    is_embedded = Column(Boolean, nullable=False, server_default=false())
    # Tracks whether this paper already has a vector stored in Qdrant, so
    # the frontend doesn't have to guess or re-track this client-side.
    # Uses SQLAlchemy's false() construct (not the raw string "false") so
    # it compiles to the correct DDL per dialect — a literal string default
    # is stored as the TEXT "false" on SQLite, which is truthy, not falsy.

    ingested_at = Column(DateTime(timezone=True), server_default=func.now())
