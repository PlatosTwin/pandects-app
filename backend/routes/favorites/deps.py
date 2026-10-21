"""Compatibility import path for ``FavoritesDeps``.

The dataclass lives in ``backend/services/favorites_service.py`` next to the
code that consumes it.
"""

from __future__ import annotations

from backend.services.favorites_service import FavoritesDeps

__all__ = ["FavoritesDeps"]
