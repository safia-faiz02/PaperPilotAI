# A LibraryEntry represents "this user has saved this paper to their
# library." Papers themselves stay in one shared, deduplicated table
# (keyed by external_id) so the same arXiv paper found by two different
# users doesn't get stored, or embedded into Qdrant, twice — only which
# papers show up in whose library differs per user.

from sqlalchemy import Column, Integer, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.sql import func

from app.models.base import Base


class LibraryEntry(Base):
    __tablename__ = "library_entries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False, index=True)
    added_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("user_id", "paper_id", name="uq_library_entry_user_paper"),
    )
