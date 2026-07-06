# This is our FastAPI app's entry point — the file Docker actually runs.
# It creates the `app` object, mounts grouped routers (like auth.py), and
# defines a few simple "is everything connected" check endpoints directly
# here. As the project grows, most NEW endpoints should go in their own
# router file under app/api/routes/ (like auth.py) rather than piling up
# in this file.

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator

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

    # LangSmith tracing is entirely env-var driven — LangChain/LangGraph
    # check LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY themselves on every
    # call, so there's no client to construct here. This is just a
    # startup log line so it's obvious from `docker compose logs api`
    # whether traces are actually being sent anywhere.
    tracing_on = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    if tracing_on and os.getenv("LANGCHAIN_API_KEY"):
        print(f"--- LangSmith tracing ENABLED (project: {os.getenv('LANGCHAIN_PROJECT', 'default')}) ---")
    else:
        print("--- LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY in .env to enable) ---")

    yield
    # Nothing to clean up on shutdown for now


app = FastAPI(
    title="PaperPilot API",
    description="Multi-agent research literature review platform",
    version="0.1.0",
    lifespan=lifespan,
)

# Allows a frontend running on a different origin (e.g. Streamlit on
# :8501, or a browser-based SPA on :5173) to call this API. Streamlit
# itself calls the API server-side (via `requests`, not browser JS) so
# this isn't a hard blocker for Streamlit specifically, but it's needed
# for any browser-based client and costs nothing to have on.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8501").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(papers.router, prefix="/papers", tags=["papers"])

# Wires up GET /metrics with standard HTTP metrics (request count,
# latency histograms, in-progress requests) for every route above —
# the format a Prometheus server scrapes. Called after the routers are
# mounted so it can instrument all of them; .expose() is what actually
# adds the /metrics endpoint itself.
Instrumentator().instrument(app).expose(app)


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
