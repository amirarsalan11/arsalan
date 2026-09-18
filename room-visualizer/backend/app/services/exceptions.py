"""
Domain-level service exceptions.

Services raise these — never HTTPException directly, keeping the
service layer free of any HTTP-specific concepts. The API layer
translates them to proper status codes via exception handlers
registered on the FastAPI app (see app/main.py).
"""


class ServiceError(Exception):
    """Base class for all domain-level service errors."""


class NotFoundError(ServiceError):
    """Raised when a requested entity does not exist (or is not visible
    to the requesting tenant)."""


class ConflictError(ServiceError):
    """Raised when an operation would violate a uniqueness or business
    rule (e.g. duplicate tenant slug)."""


class AuthenticationError(ServiceError):
    """Raised when an API key is missing, malformed, unknown, revoked,
    or belongs to a deactivated tenant.

    Deliberately raised with the SAME generic message for all of those
    distinct causes — see AuthService — so the API layer's response
    never reveals which specific reason caused the failure. Maps to
    HTTP 401.
    """


class ForbiddenError(ServiceError):
    """Raised when an authenticated caller is not permitted to perform
    the requested action (e.g. acting on a tenant that isn't their own,
    or a request from a non-allowlisted domain). Maps to HTTP 403.
    """
