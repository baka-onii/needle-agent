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
