"""
API v1 route package.

Each resource has its own router module (tenants, api_keys, materials,
render_jobs, usage_records); `router.py` aggregates them into a single
`api_router` mounted by app.main.
"""
