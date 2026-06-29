# This is our FastAPI app's entry point — the file Docker actually runs.
# It creates the `app` object, mounts grouped routers (like auth.py), and
# defines a few simple "is everything connected" check endpoints directly
# here. As the project grows, most NEW endpoints should go in their own
# router file under app/api/routes/ (like auth.py) rather than piling up
# in this file.

from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.database import check_database_connection, SessionLocal
from app.cache import check_redis_connection
from app.models import User
from app.api.routes import auth, papers
from app.rag.vector_store import ensure_collection_exists


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code in the `lifespan` function runs at startup (before the yield)
    and shutdown (after the yield). This is FastAPI's modern way of
    running setup code when the server starts — replacing the older
    @app.on_event("startup") pattern.

    We use it here to ensure our Qdrant collection exists before any
    request tries to use it. If the collection is missing, Qdrant would
    throw an error on the first embedding search — checking at startup
    gives a much cleaner failure mode and a clear log message.
    """
    print("--- Setting up Qdrant collection ---")
    try:
        ensure_collection_exists()
    except Exception as e:
        print(f"Warning: Could not connect to Qdrant at startup: {e}")
        print("Qdrant features will be unavailable until it's reachable.")
    yield
    # Nothing to clean up on shutdown for now


app = FastAPI(
    title="PaperPilot API",
    description="Multi-agent research literature review platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(papers.router, prefix="/papers", tags=["papers"])


# This is a "route". The decorator @app.get("/") means:
# "When someone visits the homepage (the `/` path) using a GET request
# (the normal kind of request your browser makes when you type a URL),
# run the function right below it."
@app.get("/")
def read_root():
    return {"message": "PaperPilot API is running!"}


# A second route, at /health. It's standard practice to have a "health
# check" endpoint — something monitoring tools or load balancers can ping
# to confirm the app is alive. We'll use this exact endpoint later when we
# set up Docker health checks and Prometheus monitoring.
@app.get("/health")
def health_check():
    return {"status": "ok"}


# Our THIRD route — this is new. It proves the api container can actually
# reach the postgres container, not just that FastAPI itself is running.
@app.get("/db-check")
def db_check():
    is_connected = check_database_connection()
    if is_connected:
        return {"database": "connected"}
    return {"database": "NOT connected — check docker compose logs"}


# Same pattern as db_check, but for Redis. Once both of these say
# "connected", we know all three containers (api, postgres, redis) are
# correctly talking to each other — the full foundation our AI agents
# will eventually run on top of.
@app.get("/redis-check")
def redis_check():
    is_connected = check_redis_connection()
    if is_connected:
        return {"redis": "connected"}
    return {"redis": "NOT connected — check docker compose logs"}


# This endpoint proves something different from db-check: not just "can
# we reach Postgres" but "does the users table that Alembic created
# actually exist and work". A SessionLocal() is a single conversation
# with the database — we open one, run a query, then always close it
# (the try/finally guarantees we close it even if the query fails).
@app.get("/users-check")
def users_check():
    db = SessionLocal()
    try:
        user_count = db.query(User).count()
        return {"users_table": "exists", "row_count": user_count}
    except Exception as e:
        return {"users_table": "NOT accessible", "error": str(e)}
    finally:
        db.close()
