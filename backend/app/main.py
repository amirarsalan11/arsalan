"""
FastAPI application entry point.

Authentication milestone scope:
- Instantiate the FastAPI app.
- Configure logging.
- Expose the `/health` endpoint (unchanged from Milestone 1).
- Mount the v1 API router (Tenant, ApiKey, Material, RenderJob,
  UsageRecord, AllowedDomain endpoints — all Bearer-authenticated — plus
  the new Public router, which is NOT Bearer-authenticated and instead
  relies on Origin-based AllowedDomain validation; see
  app/api/v1/public.py and app/api/deps.py:get_public_tenant_id).
- Register exception handlers translating service-layer domain
  exceptions (NotFoundError, ConflictError, AuthenticationError,
  ForbiddenError) into HTTP responses, so routes never need to catch
  them individually.

Tenant identity is now resolved via real API-key authentication
(`Authorization: Bearer <key>`, verified by AuthService) instead of
Milestone 3's unauthenticated X-Tenant-Id header placeholder, which has
been removed entirely — see app/api/deps.py.

Explicitly OUT of scope (added in later milestones):
- AI pipeline invocation.
- Celery task dispatch.
- Storage/S3 integration.
- Billing.
- Enforcement of allowed-domain validation on any route (the
  AllowedDomain check exists as tested, composable infrastructure —
  see app/api/deps.py:verify_allowed_domain — but is not yet wired
  into any endpoint, per the approved scope).
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.services.exceptions import AuthenticationError, ConflictError, ForbiddenError, NotFoundError

configure_logging()
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Room Visualizer SaaS — API Gateway backend.",
)

app.include_router(api_router, prefix="/api/v1")


@app.exception_handler(NotFoundError)
def handle_not_found(request: Request, exc: NotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ConflictError)
def handle_conflict(request: Request, exc: ConflictError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(AuthenticationError)
def handle_authentication_error(request: Request, exc: AuthenticationError) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"detail": str(exc)},
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.exception_handler(ForbiddenError)
def handle_forbidden_error(request: Request, exc: ForbiddenError) -> JSONResponse:
    return JSONResponse(status_code=403, content={"detail": str(exc)})


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Liveness/readiness probe for the API container.

    Intentionally has no dependencies on the database, Redis, or any other
    service — it only confirms the FastAPI process itself is running.
    Deeper readiness checks (DB connectivity, etc.) belong to a later
    milestone once those dependencies actually exist.
    """
    return {"status": "ok", "service": settings.app_name, "environment": settings.environment}
