# Frontend House Rules

## Scope and Priority
- This file supplements the root `AGENTS.md` for work inside `frontend/`.
- If this file conflicts with root instructions, follow root unless this file is more specific to `frontend/`.
- Default scope here is maintained frontend source: `client/`, `server/`, `shared/`, config, and build scripts.
- Do not edit generated outputs such as `dist/` or derived API artifacts unless the task is explicitly to regenerate or inspect them.

## Routing
- Define SPA routes in `client/main.tsx`.
- Put new route page components in `client/pages/`.
- Keep custom routes above the catch-all `*` route.

## API and Server Usage
- Prefer client-side changes first.
- Only add or change `server/` endpoints when logic must remain server-side (secrets, privileged operations, or server-owned integrations).
- Keep API paths under `/api/`.

## Shared Types and Imports
- Put shared request/response types in `shared/` when used by both client and server.
- Use existing path aliases:
  - `@/*` for `client/*`
  - `@shared/*` for `shared/*`

## UI Changes
- Preserve the existing visual language unless the user explicitly asks for a redesign.
- Prefer small, coherent refinements over one-off visual inventions that are not reused elsewhere.

## Validation Commands
- After material frontend changes, run:
  - `npm test`
  - `npm run typecheck`
- If you change prerendering, SSR, or frontend build configuration, also run `npm run build`.
