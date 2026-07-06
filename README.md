# PaperPilot

A multi-agent research literature review platform: search arXiv, build a personal paper library, generate local vector embeddings, run semantic search, and get an LLM-synthesized literature review paragraph — with a LangGraph reflection loop that checks its own citations against the source papers and retries if it hallucinates one.

Built end-to-end: FastAPI backend, Streamlit frontend, Postgres, Qdrant, Redis, and a Groq-backed LangChain/LangGraph agent pipeline, all containerized with Docker Compose.

## What it does

1. **Search** — query arXiv, save results into a personal library (per-user, backed by a shared/deduplicated paper pool so the same paper found by two users isn't stored or embedded twice).
2. **Embed** — generate a local vector embedding (fastembed, `BAAI/bge-small-en-v1.5`, no external API call) for any paper, individually or in bulk.
3. **Semantic search** — find papers by meaning, not keywords, via a Qdrant vector search scoped to your own library.
4. **Synthesize** — retrieve relevant papers and have an LLM (Groq/Llama 3) write a ~150–250 word cited synthesis paragraph. Two modes:
   - **Simple**: single-shot generation.
   - **Reflection loop**: a small LangGraph state machine (`supervisor → synthesize → reflect`) that extracts every `[citation]` from the generated text, checks it against the real paper titles (exact match, positional, or fuzzy), and — if it finds a hallucinated citation — routes back to regenerate with feedback describing exactly what was wrong, up to 2 attempts, before returning whatever it has.

## Architecture

```
Streamlit (frontend/)              FastAPI (backend/app/)
┌─────────────────────┐            ┌──────────────────────────────┐
│ Search               │  REST/JWT │ /auth      register/login/    │
│ Library              │ ────────► │            refresh (rotating) │
│ Semantic Search       │            │ /papers    search/library/    │
│ Synthesis            │            │            embed/semantic/    │
└─────────────────────┘            │            synthesize(-graph) │
                                    └───────┬─────────┬────────────┘
                                            │         │
                        ┌───────────────────┘         └───────────────┐
                        ▼                                             ▼
              Postgres (users, papers,                     Qdrant (paper vectors,
              library_entries,                             fastembed local embeddings)
              refresh_tokens)                                        │
                                                                      ▼
                                                        LangChain + LangGraph → Groq (Llama 3)
```

Auth is JWT-based with rotating refresh tokens (bcrypt password hashing, HS256-signed access tokens, hashed refresh tokens stored server-side so a stolen token is only usable once). Papers are a single shared, deduplicated pool keyed by arXiv ID; a `library_entries` join table tracks which papers are in which user's personal library, so two users searching the same topic never duplicate storage or embeddings, but each user's view stays private.

## Engineering highlights

- **Self-correcting LLM pipeline** — the LangGraph reflection loop (`backend/app/agents/synthesis_graph.py`) doesn't just trust the LLM's citations; it validates them with deterministic string matching (no second LLM call needed for the check itself) and retries with targeted feedback on failure. Both the citation-validation logic and the retry/attempt-counting are unit tested.
- **Per-user data isolation over a shared corpus** — papers and their vector embeddings are deduplicated globally, while library membership, semantic search, and synthesis are all correctly scoped per user (verified with dedicated multi-user isolation tests).
- **Rotating refresh tokens** — access tokens expire in 30 minutes; a refresh token (hashed at rest, single-use, rotated on each refresh) lets a client silently re-authenticate without a full login.
- **26 automated tests** — auth flow (register/login/refresh/logout/rotation), per-user library isolation (dedup, ownership 404s, bulk-embed scoping), and LLM-free citation validation — running against an isolated test database with external calls (arXiv, embeddings) mocked, so the suite needs no live services and runs in seconds.
- **CI on every push** — GitHub Actions runs the full test suite and a Docker build check on every push/PR to `main`.
- **Observability-ready** — Prometheus metrics (`/metrics`) including custom counters for synthesis attempts and citation-check outcomes, plus optional LangSmith tracing (env-var driven, zero code changes to enable) for full visibility into every prompt/LLM call/graph step.

## Screenshots

<!-- Add a few screenshots or a short screen recording here, e.g.:
![Search](docs/screenshots/search.png)
![Synthesis with citation validation](docs/screenshots/synthesis.png)
-->

## Tech stack

**Backend:** FastAPI · SQLAlchemy · Alembic · PostgreSQL · Qdrant · Redis · fastembed · LangChain · LangGraph · Groq (Llama 3) · JWT (python-jose) · Prometheus · pytest
**Frontend:** Streamlit
**Infra:** Docker Compose · GitHub Actions CI

## Running it locally

```bash
git clone https://github.com/safia-faiz02/PaperPilotAI.git
cd PaperPilotAI
cp .env.example .env
# edit .env — at minimum set GROQ_API_KEY (free at https://console.groq.com)

docker compose up      # api:8000, postgres:5432, redis:6379, qdrant:6333
```

In a second terminal:

```bash
cd frontend
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
pip install -r requirements.txt
streamlit run Home.py  # http://localhost:8501
```

API docs (Swagger) are available at `http://localhost:8000/docs` once the backend is running.

### Running the tests

```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt
pytest -v
```
