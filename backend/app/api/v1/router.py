"""
Aggregates all v1 sub-routers into a single router mounted by main.py.
"""

from fastapi import APIRouter

from app.api.v1 import allowed_domains, api_keys, materials, public, render_jobs, tenants, usage_records

api_router = APIRouter()

api_router.include_router(tenants.router)
api_router.include_router(api_keys.router)
api_router.include_router(materials.router)
api_router.include_router(render_jobs.router)
api_router.include_router(usage_records.router)
api_router.include_router(allowed_domains.router)
api_router.include_router(public.router)
