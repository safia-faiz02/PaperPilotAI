# A RefreshToken lets a client get a new access token without making the
# user log in again every time the short-lived access token (30 min by
# default) expires. We store a hash of the token (never the raw value)
# so a leaked database doesn't hand out usable refresh tokens directly.

from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.sql.expression import false

from app.models.base import Base


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    token_hash = Column(String, unique=True, index=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked = Column(Boolean, nullable=False, server_default=false())
    # false() (not the raw string "false") so it compiles correctly per
    # dialect — see the same note on Paper.is_embedded.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
