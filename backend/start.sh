#!/bin/bash
# This script runs every time the api container starts.
# It handles migrations automatically so you never have to run
# alembic commands manually.

echo "--- Running database migrations ---"

# Check if any migration files exist in alembic/versions/.
# If the folder is empty (e.g. fresh clone or fresh zip), generate
# the first migration automatically before upgrading.
if [ -z "$(ls -A alembic/versions/*.py 2>/dev/null)" ]; then
    echo "No migration files found — auto-generating initial migration..."
    alembic revision --autogenerate -m "create users and papers tables"
fi

# Apply all pending migrations to the database.
alembic upgrade head

echo "--- Starting FastAPI server ---"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
