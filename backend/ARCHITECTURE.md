# Backend API Architecture

## Overview
- `backend/app.py` is the application composition layer.
- `backend/routes/*` own HTTP resource wiring and response shaping.
- `backend/services/*` own query orchestration and side-effect orchestration.
- `backend/models/*` own ORM/reflection surfaces.
- `backend/schemas/*` own request/response validation and serialization.
- `backend/core/*` owns shared runtime plumbing:
  - `config.py` for app/bootstrap config composition
  - `errors.py` for API error response wiring
  - `hooks.py` for request hook wiring and header policy
- `backend/auth/runtime.py` is a compatibility facade only.
- Auth runtime logic is split by domain:
  - `backend/auth/session_runtime.py` for session transport, cookies, auth DB/mock store, session tokens, request identity helpers
  - `backend/auth/email_runtime.py` for email normalization and Resend delivery helpers
  - `backend/auth/oauth_runtime.py` for provider token exchange / JSON fetch helpers
  - `backend/auth/captcha_runtime.py` for Turnstile verification helpers
  - `backend/auth/legal_runtime.py` for legal acceptance validation/persistence and sign-on event logging

## Runtime Flow
```mermaid
flowchart LR
    A["backend/app.py (composition)"] --> B["Route deps"]
    B --> C["backend/routes/*"]
    C --> D["backend/services/*"]
    D --> E["backend/models/*"]
```

## Auth Flow
```mermaid
flowchart LR
    R["backend/routes/auth.py"] --> F["backend/auth/runtime.py (facade)"]
    F --> S["session_runtime"]
    F --> E["email_runtime"]
    F --> O["oauth_runtime"]
    F --> C["captcha_runtime"]
    F --> L["legal_runtime"]
```

## Route Dependency Injection
- Route registration is explicit and typed through `backend/routes/deps.py`.
- `backend.app._build_route_deps()` constructs:
  - `SectionsDeps`
  - `AgreementsDeps`
  - `ReferenceDataDeps`
  - `AuthDeps`
- Routes consume these dependency objects rather than dynamic module lookups.
- `AccessContextProtocol` is the shared typed access-context contract for runtime route/service calls.
- Keep dependency typing strict for helper call signatures; keep reflected SQLAlchemy model handles pragmatic where strict typing would add fragile casts.

## Runtime Contracts
- Public API behavior is unchanged:
  - no endpoint additions/removals
  - no response/request schema changes
  - no auth/session/CORS behavior changes
- Internal contracts are explicit:
  - `register_sections_routes(*, deps: SectionsDeps) -> Blueprint`
  - `register_agreements_routes(app: Flask, *, deps: AgreementsDeps) -> tuple[Blueprint, Blueprint]`
  - `register_reference_data_routes(*, deps: ReferenceDataDeps) -> tuple[Blueprint, Blueprint, Blueprint]`
  - `register_auth_routes(app: Flask, *, deps: AuthDeps) -> Blueprint`

## Agreements Domain
- `backend/filtering.py` owns the agreement metadata filter chain and the public-eligibility rule:
  - `agreement_filter_predicates(cols, args, ...)` / `apply_agreement_filters(query, cols, args, ...)` build every agreement metadata filter (year and filing-date ranges, parties, price buckets, types, counsel, industries, deal fields, PE flags, `agreement_uuid`). `cols` is either `agreements` or `latest_sections_search`, which carry the same column names. Every REST route and MCP tool that filters agreements goes through it.
  - `agreement_is_public_eligible_expr(agreements)` is the gated/unverified rule; `agreement_uuid_is_public_eligible_expr(uuid_col, agreements)` applies it to tables keyed by `agreement_uuid` (section search) without joining `agreements` per row.
- `backend/services/agreements_service.py` owns agreement query logic for REST and MCP: `AGREEMENT_METADATA_COLUMNS` / `agreement_metadata_columns()` / `row_to_agreement_metadata()` (the one metadata projection), list/search/index, single reads, tax-clause reads, summaries, trends, and filter options. `backend/routes/agreements/*` only parses arguments, applies caches, and shapes responses.
- `backend/services/snippets.py` owns XML text extraction and match-focused snippets shared by agreement search and the MCP snippet tools.
- `backend/services/sections_service.py` exposes the text-search plan race (`fetch_with_text_plan_race`) and the budgeted exact count (`budgeted_exact_count`) that agreement text search reuses.
- Services never import route modules.

## Testing Guardrails
- `backend/tests/test_route_contracts.py` checks route and operationId stability.
- `backend/tests/test_auth_dependencies.py` checks auth dependency wiring.
- `backend/tests/test_runtime_typing_guards.py` blocks broad file-level pyright suppressions in runtime files.
- `backend/tests/test_runtime_typing_guards.py` also guards:
  - narrow `Callable[..., Any]` usage in `backend/routes/deps.py` (allowed only where SQLAlchemy expression typing is still dynamic)
  - `backend/auth/runtime.py` staying facade-sized with no runtime logic definitions

## Where New Code Should Live
- New endpoint logic: `backend/routes/*` + `backend/services/*`.
- New request/response schema: `backend/schemas/*`.
- New DB model/reflection logic: `backend/models/*`.
- App bootstrap/config-only changes: `backend/app.py`.
