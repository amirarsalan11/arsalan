# Room Visualizer SaaS

A multi-tenant B2B SaaS platform that lets businesses embed an AI-powered
room visualization tool into their websites (WordPress, WooCommerce,
Shopify, React, PHP, or plain HTML).

## Status

**Milestone 1 — Project Foundation.** This repository currently contains
only the architectural scaffold: folder structure, empty layered modules,
build tooling, and the Docker Compose service topology. No authentication,
database models, AI pipeline, or Celery tasks are implemented yet — see
`Known Limitations` below and the project's CLAUDE.md for the full MVP
development order.

## Architecture

```
Customer Website
  -> Universal Integration Layer
  -> Iframe Visualizer
  -> API Gateway
  -> FastAPI Backend
  -> Celery + Redis Queue
  -> GPU Worker
  -> AI Rendering Pipeline
  -> Object Storage
```

## Repository Layout

```
backend/           FastAPI backend (API -> Service -> Repository -> DB)
ai-engine/          Independent AI rendering pipeline (segmentation, geometry,
                    material, perspective, lighting, compositing stages)
frontend-widget/    React + TypeScript + Vite widget (runs inside an iframe)
sdk/                Framework-independent JS SDK (creates/manages the iframe)
infra/              Docker Compose service definitions
```

## Local Development

### Prerequisites

- Docker and Docker Compose
- Node.js 20+ (for frontend-widget / sdk local dev outside Docker)
- Python 3.11+ (for backend local dev outside Docker)

### Run the backend stack

```bash
cp .env.example .env
cd infra
docker compose up --build
```

This starts `postgres`, `redis`, `api`, and an idle `worker` placeholder.
The GPU worker is disabled by default; enable it explicitly once the AI
pipeline exists:

```bash
docker compose --profile gpu up --build
```

Verify the API is up:

```bash
curl http://localhost:8000/health
```

### Run the backend tests

```bash
cd backend
pip install -r requirements.txt
pytest
```

### Run the frontend widget

```bash
cd frontend-widget
npm install
npm run dev
```

### Run the frontend widget tests

```bash
cd frontend-widget
npm install
npm run test
```

### Build the SDK

```bash
cd sdk
npm install
npm run build
```

## Known Limitations (Milestone 1)

- No authentication, API key validation, or tenant/domain validation.
- No database models or migrations (Alembic is initialized but empty).
- No AI pipeline logic — all `ai_engine/*` stage modules are empty placeholders.
- No Celery task implementation — the `worker` container is idle.
- No SDK ↔ iframe `postMessage` wiring yet.
- No widget UI beyond an empty shell.
- No storage integration (S3/R2) — `app/storage` is an empty placeholder.
