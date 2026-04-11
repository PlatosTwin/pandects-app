# Security Policy

## Reporting a vulnerability

Please do not file public GitHub issues for security-sensitive reports.

Send a report to:

- `nmbogdan@alumni.stanford.edu`

Include:

- a clear description of the issue
- affected paths or components
- reproduction steps or proof of concept
- impact assessment if known
- any suggested remediation

## Response expectations

This is a maintainer-run project, not a staffed security program. Reports are handled on a best-effort basis, but legitimate security issues will be reviewed and addressed as quickly as practical.

## Scope notes

Useful reports include vulnerabilities in:

- the Flask API and auth flows
- deployment or configuration mistakes in tracked files
- credential handling or accidental secret exposure in the repository
- docs or examples that encourage unsafe practices

Out of scope:

- private infrastructure the reporter cannot actually verify
- generic hardening advice without a concrete issue
- reports that rely on guessed production configuration rather than tracked repo behavior
