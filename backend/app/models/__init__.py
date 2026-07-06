# This lets other files write `from app.models import User, Paper`
# instead of needing to know exactly which file each model lives in.
# It also matters for Alembic: importing every model here guarantees
# they're all "registered" on Base.metadata before Alembic looks at it.

from app.models.base import Base
from app.models.user import User
from app.models.paper import Paper
from app.models.library_entry import LibraryEntry
from app.models.refresh_token import RefreshToken
