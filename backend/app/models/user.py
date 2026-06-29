# A "model" is a Python class that represents a database table. Each
# class attribute (id, email, etc.) becomes a column. SQLAlchemy handles
# translating between this Python class and actual SQL behind the scenes.

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func

from app.models.base import Base


class User(Base):
    """
    One row per registered user. We'll connect this to real authentication
    (password hashing, JWT login tokens) in a later step — for now we're
    just defining the SHAPE of the data so Alembic can create the table.
    """

    # __tablename__ is the literal name Postgres will use for this table.
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    # primary_key=True -> this column uniquely identifies each row.
    # index=True -> Postgres builds a fast lookup index on this column
    # (very cheap to add now, much more annoying to add to a large table
    # later, so we add indexes on anything we'll search/filter by often).

    email = Column(String, unique=True, index=True, nullable=False)
    # unique=True -> Postgres will reject two users with the same email.
    # nullable=False -> this field is required; can't be left empty.

    hashed_password = Column(String, nullable=False)
    # We will NEVER store a real password — only a hashed (scrambled,
    # one-way) version of it. We'll write that logic in the auth step.

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # server_default=func.now() -> Postgres itself fills this in
    # automatically when a row is created, so we never have to remember
    # to set it ourselves in our Python code.
