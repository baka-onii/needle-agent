"""Authentication helpers for the Fieldnotes example workspace."""

import hashlib
import hmac


def hash_password(password: str, salt: bytes) -> bytes:
    """Derive a password hash using PBKDF2-HMAC-SHA256."""
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000)


def authenticate_user(password: str, salt: bytes, expected_hash: bytes) -> bool:
    """Check authentication without a timing-dependent string comparison."""
    actual_hash = hash_password(password, salt)
    return hmac.compare_digest(actual_hash, expected_hash)


# Demo: Simple auth request → verify → "ok"

SALT = b"my-secret-salt-1234567890"
PASSWORD = "correct-horse-battery-staple"

# 1. Hash the password once (the server-side secret)
expected_hash = hash_password(PASSWORD, SALT)

# 2. Send an auth request with the password
result = authenticate_user(PASSWORD, SALT, expected_hash)

# 3. Function verifies and returns "ok"
if result:
    print("Auth verified -> ok")
else:
    print("Auth failed → error")
