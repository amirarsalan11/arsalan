"""
API-layer dependency injection providers.

Authentication milestone change: the temporary, unauthenticated
`get_tenant_id` (X-Tenant-Id header) has been REMOVED entirely and
replaced by `get_current_tenant_id`, which requires and verifies a real
API key via `Authorization: Bearer <key>`.

SDK + Widget milestone change: `verify_allowed_domain` (previously
foundation-only, uncomposed) is now actually wired into the new public
router via `get_public_tenant_id`, defined below.

Dependencies defined here:

1. `get_db` — yields a SQLAlchemy Session per request, closing it
   afterward. Unchanged from Milestone 3.

2. `get_current_tenant_id` — extracts the bearer token from the
   Authorization header and verifies it via AuthService, returning the
   resolved, active tenant_id. Required by every existing
   tenant-scoped route (tenants, api-keys, materials, render-jobs,
   usage-records, allowed-domains). UNCHANGED by this milestone —
   the public router below uses a completely separate dependency, so
   nothing about the existing Bearer-auth flow is touched.

3. `verify_allowed_domain` — the underlying domain-matching check,
   given a resolved tenant_id and an origin/domain string. Raises
   ForbiddenError if not allowed. Used by `get_public_tenant_id` below;
   also usable directly by any future caller that already has a
   verified tenant_id and just needs the domain check.

4. `get_public_tenant_id` — NEW. The auth mechanism for the public,
   widget-facing router. Deliberately does NOT use
   `get_current_tenant_id`/Bearer auth at all — the public router must
   never require or accept a secret API key, since the browser-side
   SDK/widget never has one. Instead: takes `tenant_id` from the path,
   the `Origin` header from the request, and verifies that origin is
   on the tenant's active AllowedDomain whitelist. Raises
   NotFoundError (-> 404) if the tenant doesn't exist or is inactive
   — deliberately not AuthenticationError, since this route never uses
   Bearer auth and a 401 would misleadingly imply otherwise —
   ForbiddenError (-> 403) if the origin isn't allowlisted. Also sets
   `Access-Control-Allow-Origin` on the response (echoing back the
   validated Origin, never `*`) so the browser's CORS check succeeds
   — but only after the real server-side domain check has already
   passed. This ordering is the whole point: CORS headers here are a
   consequence of passing the check, not a substitute for it.
"""

import uuid
from collections.abc import Generator

from fastapi import Depends, Header, Response
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.repositories import TenantRepository
from app.services.allowed_domain_service import AllowedDomainService
from app.services.auth_service import AuthService
from app.services.exceptions import AuthenticationError, ForbiddenError, NotFoundError

_BEARER_PREFIX = "Bearer "


def get_db() -> Generator[Session, None, None]:
    """Provide a request-scoped SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_tenant_id(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> uuid.UUID:
    """Resolve and verify the acting tenant from the Authorization header.

    Expects `Authorization: Bearer <raw_api_key>`. Raises
    AuthenticationError with a generic message for every failure case
    — see AuthService.verify_api_key and its docstring for the full
    list of causes that all collapse to the same response.
    """
    if authorization is None or not authorization.startswith(_BEARER_PREFIX):
        raise AuthenticationError("Invalid or missing API key.")

    raw_key = authorization[len(_BEARER_PREFIX):].strip()

    auth_service = AuthService(db)
    return auth_service.verify_api_key(raw_key)


def verify_allowed_domain(tenant_id: uuid.UUID, origin_or_referer: str | None, db: Session) -> None:
    """Raise ForbiddenError unless `origin_or_referer` is on the
    tenant's active AllowedDomain whitelist. Pure domain-check logic,
    reusable by any dependency that already has a tenant_id in hand
    (see `get_public_tenant_id` below, the first real caller)."""
    if origin_or_referer is None:
        raise ForbiddenError("Request origin is not permitted for this tenant.")

    service = AllowedDomainService(db)
    if not service.is_domain_allowed(tenant_id, origin_or_referer):
        raise ForbiddenError("Request origin is not permitted for this tenant.")


def get_public_tenant_id(
    tenant_id: uuid.UUID,
    response: Response,
    origin: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> uuid.UUID:
    """Auth mechanism for the public router — NOT Bearer-token based.

    Deliberately separate from `get_current_tenant_id`: the public
    router exists precisely because the browser-side widget has no
    secret key to present. Instead of a bearer token, the caller's
    identity is the `tenant_id` path parameter itself, and the only
    "credential" is browsing from a domain that tenant has explicitly
    allowlisted.

    - 404 (NotFoundError) if the tenant doesn't exist or is inactive: a
      public endpoint must not confirm/deny tenant existence to an
      unlisted caller either, and — importantly — must NOT raise
      AuthenticationError here, since that maps to a 401 response with
      a `WWW-Authenticate: Bearer` header, which would be misleading on
      a route that never uses Bearer auth in the first place.
    - 403 (ForbiddenError) if the tenant exists and is active, but the
      request's Origin isn't on its whitelist.
    - On success: sets `Access-Control-Allow-Origin` to the exact,
      validated Origin value (never `*`) and `Vary: Origin` (so caches
      don't serve one tenant's CORS-approved response to a different
      origin), then returns the tenant_id for the route to use.
    """
    tenant_repository = TenantRepository(db)
    tenant = tenant_repository.get_by_id(tenant_id)

    if tenant is None or not tenant.is_active:
        raise NotFoundError(f"Tenant '{tenant_id}' not found.")

    verify_allowed_domain(tenant_id, origin, db)

    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Vary"] = "Origin"

    return tenant_id
