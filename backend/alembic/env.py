# This file is what Alembic actually runs every time you create or apply
# a migration. Its two jobs: (1) tell Alembic which database to connect
# to, and (2) tell Alembic what our tables are SUPPOSED to look like
# (target_metadata), so it can compare that against what's actually in
# the database and generate the difference as a migration script.

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Make sure Python can find our "app" package (this file lives in
# backend/alembic/, our code lives in backend/app/ — this line adds
# backend/ to the search path so "from app... import" works).
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.database import DATABASE_URL
from app.models import Base  # noqa: F401 — importing registers all tables

config = context.config
config.set_main_option("sqlalchemy.url", DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# This is the "what SHOULD exist" reference Alembic compares the real
# database against when you run `alembic revision --autogenerate`.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generates SQL without actually connecting to a database. We won't
    use this mode in this project, but Alembic expects it to be defined."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """The mode we actually use: connects directly to Postgres and runs
    migrations for real."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
