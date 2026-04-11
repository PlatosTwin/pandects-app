# Docs

## Purpose

`docs/` is the standalone Docusaurus site for Pandects guides, MCP docs, and API reference material.

## What outside contributors can do here

- docs content
- navigation and information architecture
- API reference presentation
- docs build fixes

This is fully contributor-friendly.

## Required tools

- Node.js 24.x
- npm 10+

## Local commands

Install dependencies:

```bash
cd docs
npm install
```

Run the local docs server:

```bash
caffeinate -i npm run dev:clean
```

Build the docs site:

```bash
caffeinate -i npm run build
```

Regenerate API docs when the backend OpenAPI changes:

```bash
caffeinate -i npm run clean-api
caffeinate -i npm run gen-api
```

## Environment variables

No dedicated docs `.env.example` is provided because the docs app does not currently require a tracked local env contract for standard contributor workflows.

## Maintainer-only dependencies and quirks

- deployment uses Fly and is maintainer-only
- API reference generation expects the tracked OpenAPI source that lives with the repo; deploy access is not required

## Related docs

- root [README.md](../README.md)
- root [CONTRIBUTING.md](../CONTRIBUTING.md)
- [backend/README.md](../backend/README.md)
