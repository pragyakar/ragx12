# ragX12-app

Frontend React (Vite) application for interacting with the ragX12 FastAPI summarization backend.

## Features
- Paste raw X12 EDI content and request an AI-generated human-readable summary.
- Health check indicator for backend availability.
- Displays structured summary sections, possible actions, and code meanings.
- Glassmorphic light UI with Tailwind (CDN) + responsive layout.

## Quick Start

```bash
# Install dependencies
npm install

# Run dev server
npm run dev
```
The app opens (default http://localhost:5173). Configure backend URL via environment variable:

1. Copy `.env.example` to `.env` (or `.env.local`).
2. Set `VITE_API_BASE` to your FastAPI endpoint (e.g. `http://localhost:8000`).
3. Restart dev server if already running.

## Build
```bash
npm run build
npm run preview
```

## Tailwind Styling
Using the CDN version for zero-config; for production hardening you can migrate to a build-time Tailwind setup.

## Environment Configuration
`VITE_API_BASE` provides the base URL for `/health` and `/summarize` calls. Vite exposes variables prefixed with `VITE_` to the frontend at build time.

---
MIT License
