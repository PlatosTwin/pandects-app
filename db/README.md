# DB

## Purpose

`db/` contains Fly deployment configuration for the main MariaDB service and related operational env expectations.

## What outside contributors can do here

- review deployment config
- improve docs
- make carefully scoped config changes with maintainer coordination
- propose narrowly scoped reviewed patches

Outside contributors should not expect to run or administer this service locally.

## Runtime status

This directory is maintainer-only for actual execution and deployment.

## Environment variables

See:

- `db/.env.example`
- root `ENVIRONMENT.md`

## Maintainer-only dependencies and quirks

- private database credentials are required
- deployment assumes maintainer Fly access
- this directory is not part of the public-safe onboarding flow

## Related docs

- root [README.md](../README.md)
- root [ENVIRONMENT.md](../ENVIRONMENT.md)
