"""Dependency container for the MCP blueprint.

Split from ``__init__`` so helper submodules can import the type without
circularity.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.routes.deps import AgreementsDeps, ReferenceDataDeps, SectionsServiceDeps


@dataclass(frozen=True)
class McpDeps:
    sections_service_deps: SectionsServiceDeps
    agreements_deps: AgreementsDeps
    reference_data_deps: ReferenceDataDeps
