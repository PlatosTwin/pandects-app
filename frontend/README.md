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

## Dependency security overrides

`package.json` uses npm `overrides` only for targeted security patches where an upstream dependency has not yet relaxed its own transitive range. These overrides should not become permanent by default.

Current overrides:

- `qs`: keeps Express 4/body-parser on a patched parser version without forcing an Express 5 migration.
- `diff`: keeps `@flydotio/dockerfile` on a patched `diff` version without accepting npm audit's downgrade suggestion for `@flydotio/dockerfile`.

When reviewing npm vulnerabilities or refreshing dependencies:

1. Run `npm audit`.
2. Run `npm ls qs diff`.
3. Check whether `express`/`body-parser` and `@flydotio/dockerfile` now resolve patched versions without overrides.
4. Remove any override that is no longer needed, then run `npm install`, `npm audit`, `npm test`, `npm run typecheck`, and `npm run build`.

## Related docs

- root [README.md](../README.md)
- root [ENVIRONMENT.md](../ENVIRONMENT.md)
- [docs/README.md](../docs/README.md)
