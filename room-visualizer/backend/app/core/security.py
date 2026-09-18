"""
API key generation and hashing primitives.

Security model (see Authentication milestone plan for full rationale):

- Raw keys are 256-bit, cryptographically random tokens generated with
  `secrets.token_urlsafe`. They are never derived from anything
  tenant-specific, and no information is embedded in them.
- Keys are hashed with HMAC-SHA256, keyed by a server-side secret
  (`Settings.api_key_hash_secret`, the "pepper"). This is deliberately
  NOT bcrypt/argon2/scrypt: those algorithms exist to slow down
  brute-forcing of low-entropy, human-chosen secrets, at the cost of
  making O(1) hash lookup impossible (you must iterate every stored
  hash and verify each one). A 256-bit random token has no brute-force
  surface for a slow hash to defend against, so a fast, deterministic,
  keyed hash is the correct tool: it can be looked up directly via an
  indexed database column, and the server-side pepper means a
  database-only leak (without the pepper) is not enough to reconstruct
  or rainbow-table the hash space.
- The pepper NEVER touches the database. It lives only in the
  application's environment. Rotating it invalidates every previously
  issued key's ability to verify (there is no migration path for this
  in the MVP — rotation is an explicit, disruptive operation).

Never log, store, or return the raw key anywhere except the single
creation-time response that generates it.
"""

import hashlib
import hmac
import secrets

from app.core.config import get_settings

# Prefix makes keys recognizable in logs/UIs (e.g. "this looks like a
# room-visualizer key") without revealing anything about the tenant or
# key itself. "_live_" mirrors the Stripe-style convention, leaving room
# for a future "rv_test_" prefix for sandbox/test-mode keys without a
# breaking format change.
API_KEY_PREFIX = "rv_live_"

# 32 raw bytes -> 256 bits of entropy, URL-safe base64 encoded by
# token_urlsafe (~43 characters).
_RAW_TOKEN_BYTES = 32


def generate_api_key() -> str:
    """Generate a new raw API key.

    Returns the full, prefixed raw key (e.g. "rv_live_<43 random chars>").
    This value must be shown to the caller exactly once and never stored.
    """
    token = secrets.token_urlsafe(_RAW_TOKEN_BYTES)
    return f"{API_KEY_PREFIX}{token}"


def hash_api_key(raw_key: str) -> str:
    """Compute the HMAC-SHA256 hash of a raw API key, hex-encoded.

    This is the only form of the key that is ever persisted. The same
    raw key always hashes to the same value (deterministic), which is
    what allows O(1) lookup by hash rather than an O(n) verification
    loop — see module docstring for why that's the right trade-off here.
    """
    settings = get_settings()
    secret_bytes = settings.api_key_hash_secret.encode("utf-8")
    message_bytes = raw_key.encode("utf-8")
    digest = hmac.new(secret_bytes, message_bytes, hashlib.sha256)
    return digest.hexdigest()


def verify_api_key(raw_key: str, expected_hash: str) -> bool:
    """Constant-time comparison of a raw key's hash against a stored hash.

    Using `hmac.compare_digest` (rather than `==`) avoids leaking timing
    information about how many leading characters of the hash matched,
    which is standard practice when comparing secret-derived values.
    """
    computed_hash = hash_api_key(raw_key)
    return hmac.compare_digest(computed_hash, expected_hash)
