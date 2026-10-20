from __future__ import annotations

DEFAULT_PAGE_SIZE = 25
AUTHENTICATED_MAX_PAGE_SIZE = 100
ANONYMOUS_MAX_PAGE_SIZE = 10


def search_max_page_size(*, is_authenticated: bool) -> int:
    return AUTHENTICATED_MAX_PAGE_SIZE if is_authenticated else ANONYMOUS_MAX_PAGE_SIZE


def clamp_page_size(page_size: int, *, max_page_size: int) -> int:
    """Clamp a requested page size into [1, max_page_size].

    A size below 1 falls back to the default. A size above the ceiling becomes
    the ceiling, so a caller asking for "as many as allowed" gets a full page
    rather than silently dropping back to the default.
    """
    if page_size < 1:
        return min(DEFAULT_PAGE_SIZE, max_page_size)
    return min(page_size, max_page_size)
