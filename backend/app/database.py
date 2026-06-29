# This file's only job is: set up a connection to PostgreSQL, and give the
# rest of our app a clean way to use it. Nothing else in the project
# should need to know HOW we connect — just import from here.

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Read connection settings from environment variables (which Docker
# Compose loads from our .env file — see docker-compose.yml).
# os.getenv("NAME", "fallback") reads the variable, or uses "fallback" if
# it's missing — useful so this file doesn't crash if .env is incomplete.
POSTGRES_USER = os.getenv("POSTGRES_USER", "paperpilot")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "paperpilot_dev_password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "paperpilot")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# Build the full connection string Postgres/SQLAlchemy expects, in the
# format: postgresql://username:password@host:port/database_name
DATABASE_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# The "engine" is SQLAlchemy's core connection manager — it knows how to
# open/reuse/close actual network connections to Postgres. We create it
# once, here, and the whole app shares it.
engine = create_engine(DATABASE_URL)

# A "session" is a single conversation with the database — used to run
# queries, add data, etc. SessionLocal is a factory that creates new
# sessions whenever we need one (we'll use this properly once we add
# real database models in a later step).
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def check_database_connection() -> bool:
    """
    Tries to actually talk to Postgres and run the simplest possible
    query (SELECT 1). Returns True if it works, False if anything goes
    wrong. This is what our /db-check endpoint calls.
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception as e:
        # In a real production app we'd log this properly (see the
        # monitoring step later). For now, printing is enough to see
        # what went wrong in the terminal running docker compose.
        print(f"Database connection failed: {e}")
        return False


def get_db():
    """
    This is a "dependency" — a function FastAPI calls automatically for
    any endpoint that asks for it (you'll see `db: Session = Depends(get_db)`
    in route files). It opens one database session, hands it to the
    endpoint to use, and then — critically — closes it afterwards no
    matter what happens (success or error), via the `finally` block.
    `yield` (instead of `return`) is what makes that "do this, run the
    endpoint, then do cleanup" pattern possible.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
