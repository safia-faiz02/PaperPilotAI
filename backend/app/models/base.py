from sqlalchemy.orm import declarative_base

# Base is the parent class every database "model" (table) inherits from.
# SQLAlchemy uses it to keep track of every table we've defined — which
# is also how Alembic (our migrations tool, set up below) knows what
# tables SHOULD exist, so it can generate the SQL to create/update them.
Base = declarative_base()
