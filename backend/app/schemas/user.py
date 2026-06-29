# "Schemas" here are Pydantic models — different from our SQLAlchemy
# "models" (database tables) even though the naming sounds similar.
# A SQLAlchemy model = a database table. A Pydantic schema = the shape of
# data going IN or OUT of an API endpoint. Keeping them separate matters:
# we never want to accidentally expose a hashed_password field in an API
# response just because it happens to exist on the database row.

from datetime import datetime
from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    """
    The shape of data we REQUIRE when someone registers. FastAPI uses
    this to automatically validate incoming requests — if someone sends
    a request missing "password", or with an email that isn't actually a
    valid email format, FastAPI rejects it automatically, before our own
    code even runs. We don't have to write that validation by hand.
    """
    email: EmailStr
    password: str


class UserOut(BaseModel):
    """
    The shape of data we SEND BACK after registration. Notice there's no
    password field here at all — even hashed, we simply never include it
    in API responses. This is the safety mechanism mentioned above, in
    practice.
    """
    id: int
    email: EmailStr
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    """
    The shape of the response after a successful login. `access_token` is
    the JWT string the client must store (e.g. in memory or localStorage)
    and send back with every future request that needs authentication.
    `token_type` is always "bearer" — that's the standard HTTP convention
    for "this is a token you put in the Authorization header."
    """
    access_token: str
    token_type: str = "bearer"
