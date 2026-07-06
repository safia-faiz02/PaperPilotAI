# Until now, every endpoint lived directly in main.py — fine for 4
# endpoints, but the project will eventually have endpoints for auth,
# projects, papers, jobs, admin, and more. APIRouter lets us group
# related endpoints into their own file, then "plug" the whole group
# into the main app in one line (see main.py). This is the standard
# FastAPI project layout you'll see in real codebases.

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, RefreshToken
from app.schemas.user import UserCreate, UserOut, Token, RefreshRequest
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    generate_refresh_token,
    hash_refresh_token,
)

router = APIRouter()

# OAuth2PasswordBearer tells FastAPI: "tokens come in via the Authorization
# header as 'Bearer <token>'. The tokenUrl here is just for the /docs UI
# — it tells Swagger where to POST login credentials to get a token,
# which makes the "Authorize" button in /docs actually work.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    This is a DEPENDENCY — a reusable function that any endpoint can
    declare to require a logged-in user. Add `current_user: User =
    Depends(get_current_user)` to any endpoint and FastAPI will
    automatically: extract the token from the request header → validate
    it → look up the user → hand them to your endpoint. If the token is
    missing, expired, or invalid, FastAPI returns a 401 automatically
    before your endpoint code even runs.
    """
    email = decode_access_token(token)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please log in again.",
            # WWW-Authenticate header is required by the HTTP standard
            # whenever we return a 401 on a bearer-token endpoint.
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    """
    Creates a new user.

    - `user_in: UserCreate` -> FastAPI reads the incoming request body,
      validates it against our UserCreate schema, and gives us back a
      typed Python object. If validation fails, FastAPI auto-responds
      with a clear 422 error — we never reach this code at all in that case.
    - `db: Session = Depends(get_db)` -> FastAPI calls get_db() for us
      automatically and hands us a ready-to-use database session.
    - `response_model=UserOut` -> FastAPI automatically filters whatever
      we return down to exactly the fields UserOut defines — so even if
      we returned the full User object, the hashed_password field would
      never actually be sent back to the client.
    """
    existing_user = db.query(User).filter(User.email == user_in.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists.",
        )

    new_user = User(
        email=user_in.email,
        hashed_password=hash_password(user_in.password),
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    Logs in a user and returns a JWT access token.

    Note: we use OAuth2PasswordRequestForm here instead of our own
    schema. This is a FastAPI built-in that reads a standard HTML-style
    form body with "username" and "password" fields — it's what the
    /docs "Authorize" button sends. The field is called "username" by
    HTTP convention, but we treat it as an email address.
    """
    # Step 1: does this email exist at all?
    user = db.query(User).filter(User.email == form_data.username).first()

    # Step 2: does the password match?
    # Note: we check BOTH in sequence but give the same vague error
    # either way — "incorrect email or password". Never tell an attacker
    # specifically WHICH one was wrong (that leaks whether an email is
    # registered).
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Step 3: create the short-lived access token. We embed the user's
    # email as "sub" (subject) — this is what decode_access_token() will
    # extract later to identify who made a request.
    access_token = create_access_token(data={"sub": user.email})

    # Step 4: create a long-lived refresh token so the client can get a
    # new access token later without logging in again. Only the hash is
    # stored — the raw token goes to the client and is never seen again.
    raw_refresh_token, token_hash, expires_at = generate_refresh_token()
    db.add(RefreshToken(user_id=user.id, token_hash=token_hash, expires_at=expires_at))
    db.commit()

    return Token(access_token=access_token, refresh_token=raw_refresh_token)


@router.post("/refresh", response_model=Token)
def refresh(request: RefreshRequest, db: Session = Depends(get_db)):
    """
    Trades a valid, unexpired, unrevoked refresh token for a brand new
    access token AND a brand new refresh token (the old one is revoked in
    the same step — this "rotation" means a stolen refresh token can only
    be used once before the legitimate client's next refresh invalidates
    it).
    """
    token_hash = hash_refresh_token(request.refresh_token)
    stored = (
        db.query(RefreshToken)
        .filter(RefreshToken.token_hash == token_hash)
        .first()
    )

    # We always store expires_at as UTC, but not every DB backend round-trips
    # tzinfo the same way (SQLite drops it on read even though Postgres
    # keeps it) — normalize before comparing so this works on either.
    expires_at = stored.expires_at if stored else None
    if expires_at is not None and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if (
        stored is None
        or stored.revoked
        or expires_at < datetime.now(timezone.utc)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token. Please log in again.",
        )

    user = db.query(User).filter(User.id == stored.user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists.",
        )

    # Rotate: revoke the used token, issue a new one.
    stored.revoked = True
    new_access_token = create_access_token(data={"sub": user.email})
    raw_refresh_token, new_token_hash, expires_at = generate_refresh_token()
    db.add(RefreshToken(user_id=user.id, token_hash=new_token_hash, expires_at=expires_at))
    db.commit()

    return Token(access_token=new_access_token, refresh_token=raw_refresh_token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: RefreshRequest, db: Session = Depends(get_db)):
    """Revokes a refresh token so it can no longer be used to get new access tokens."""
    token_hash = hash_refresh_token(request.refresh_token)
    stored = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()
    if stored is not None:
        stored.revoked = True
        db.commit()
    return None


@router.get("/me", response_model=UserOut)
def get_me(current_user: User = Depends(get_current_user)):
    """
    Returns the currently logged-in user's profile. This is the simplest
    possible "protected" endpoint — it demonstrates the full auth flow:
    log in → get token → send token → get your own data back. Every
    future endpoint that needs a logged-in user follows this exact same
    pattern using Depends(get_current_user).
    """
    return current_user
