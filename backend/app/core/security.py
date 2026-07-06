# This file's only job: turn real passwords into safely-stored hashes,
# and check a typed-in password against a stored hash. Nothing else in
# the app should ever touch a raw password directly except through these
# two functions.
#
# Why we never store real passwords: if our database ever leaked, a
# stored real password is immediately usable by an attacker — and most
# people reuse passwords across sites, so it's not just OUR app at risk.
# A "hash" is a one-way scramble: easy to check "does this match?", but
# practically impossible to reverse back into the original password.

from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
import hashlib
import os
import secrets

# ── Password hashing ──────────────────────────────────────────────────────────

# bcrypt is a well-established, deliberately slow hashing algorithm
# (slow is a FEATURE here — it makes brute-force password guessing
# expensive for an attacker). CryptContext is passlib's wrapper that
# handles the hashing details for us.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    """Turns a real password into a hash, safe to store in the database."""
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Checks a freshly typed-in password against a previously stored hash.
    Used in the login endpoint below.
    """
    return pwd_context.verify(plain_password, hashed_password)


# ── JWT tokens ────────────────────────────────────────────────────────────────
# A JWT (JSON Web Token) is a signed string that looks like:
#   eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyQGV4LmNvbSJ9.SIGNATURE
# It has three parts separated by dots: header, payload, signature.
# The PAYLOAD contains data we choose to put in (like the user's email).
# The SIGNATURE is a cryptographic stamp made with SECRET_KEY — so anyone
# who receives this token can verify it hasn't been tampered with, without
# asking the database anything.

SECRET_KEY = os.getenv("SECRET_KEY", "fallback-dev-secret-not-for-production")
ALGORITHM = "HS256"
# HS256 = HMAC with SHA-256 — a standard, well-trusted signing algorithm.
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))


def create_access_token(data: dict) -> str:
    """
    Creates a signed JWT token containing whatever data we pass in.
    We'll pass in {"sub": user_email} — "sub" (subject) is the standard
    JWT field name for "who this token belongs to".
    """
    to_encode = data.copy()

    # Set expiry time — after this, the token is considered invalid even
    # if the signature is correct. This limits the damage if a token is
    # ever stolen: it stops working on its own after 30 minutes.
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> str | None:
    """
    Validates a token and extracts the user's email ("sub") from it.
    Returns None if the token is invalid, expired, or tampered with.
    This is what protected endpoints will call to know WHO is making
    the request, without touching the database.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        return email
    except JWTError:
        # JWTError covers: expired token, wrong signature, malformed token.
        # In all these cases we just return None — the caller then decides
        # to reject the request with a 401 Unauthorized response.
        return None


# ── Refresh tokens ────────────────────────────────────────────────────────────
# Unlike the access token (a self-contained, short-lived JWT), a refresh
# token is just a long random string we store server-side (hashed) so it
# can be looked up, checked for expiry/revocation, and rotated. This is
# what lets a client silently get a new access token instead of being
# forced to log in again every 30 minutes.

REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))


def generate_refresh_token() -> tuple[str, str, datetime]:
    """
    Returns (raw_token, token_hash, expires_at). The raw token is what
    goes to the client; only the hash is ever stored in the database, the
    same way we never store a plain password.
    """
    raw_token = secrets.token_urlsafe(32)
    token_hash = hash_refresh_token(raw_token)
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    return raw_token, token_hash, expires_at


def hash_refresh_token(raw_token: str) -> str:
    """
    A refresh token is a random string, not a password someone chose —
    there's no brute-force risk to defend against, so a fast, deterministic
    hash (unlike bcrypt's salted, slow one) is fine and lets us look it up
    by value directly.
    """
    return hashlib.sha256(raw_token.encode()).hexdigest()
