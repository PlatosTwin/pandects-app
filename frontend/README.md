# Frontend

## Purpose

`frontend/` contains the Vite + React application, the SSR/prerender server code, and shared frontend types.

## What outside contributors can do here

- normal UI and UX work
- tests and type checks
- accessibility improvements
- docs and copy improvements in the app

This is one of the most contributor-friendly parts of the repository.

## Required tools

- Node.js 24.x
- npm 10+

## Local setup

```bash
cd frontend
npm install
cp .env.example .env.local
```

## Local commands

Start dev server:

```bash
caffeinate -i npm run dev
```

Run tests:

```bash
caffeinate -i npm test
```

Run type checks:

```bash
caffeinate -i npm run typecheck
```

Run a production build:

```bash
caffeinate -i npm run build
```

## Environment variables

See:

- `frontend/.env.example`
- root `ENVIRONMENT.md`

Normal outside-contributor usage usually needs only:

- `VITE_API_BASE_URL=http://localhost:5113`
- optionally `VITE_AUTH_SESSION_TRANSPORT=bearer`

Zitadel-related `VITE_*` values are maintainer-oriented or only needed when actively working on those auth flows.

## Maintainer-only dependencies and quirks

- Real Zitadel sign-in flows require private client configuration.
- Production deployment is on Fly and is not part of normal contributor onboarding.
- The frontend may link to the docs site and API, but local development does not require deployment access.

## Related docs

- root [README.md](../README.md)
- root [ENVIRONMENT.md](../ENVIRONMENT.md)
- [docs/README.md](../docs/README.md)
