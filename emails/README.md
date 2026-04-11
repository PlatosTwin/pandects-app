# Emails

## Purpose

`emails/` contains the React Email templates and sync tooling used for Pandects auth and account emails.

## What outside contributors can do here

- inspect the existing email templates
- improve markup, copy, or documentation in reviewed changes
- work on preview-only local changes

Rendering and previewing the existing templates can be done locally. Adding or publishing templates that depend on real delivery is maintainer-only.

## Required tools

- Node.js 24.x
- npm 10+

## Local commands

Install dependencies:

```bash
cd emails
npm install
```

Run the local preview server:

```bash
caffeinate -i npm run dev
```

Build templates:

```bash
caffeinate -i npm run build
```

## Environment variables

See:

- `emails/.env.example`
- root `ENVIRONMENT.md`

Local preview does not require secrets.

## Maintainer-only dependencies and quirks

- Resend publishing and template syncing are maintainer-only workflows.
- `npm run email` requires a valid `RESEND_API_KEY`
- syncing is an operational workflow and should not be presented as default contributor setup

## Related docs

- root [README.md](../README.md)
- root [ENVIRONMENT.md](../ENVIRONMENT.md)
