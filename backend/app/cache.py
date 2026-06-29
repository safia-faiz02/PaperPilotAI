# Same idea as database.py, but for Redis instead of Postgres. Keeping
# each connection's setup in its own small file makes it easy to find and
# easy to reason about — main.py will just import "check it works"
# functions from both, without needing to know the details.

import os
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Creates a reusable Redis client. decode_responses=True means Redis will
# hand us back normal Python strings instead of raw bytes — simpler to
# work with for now.
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def check_redis_connection() -> bool:
    """
    Redis has a built-in PING command specifically for this purpose —
    "are you alive?" — and replies PONG if so. This is what our
    /redis-check endpoint will call.
    """
    try:
        return redis_client.ping()
    except Exception as e:
        print(f"Redis connection failed: {e}")
        return False
