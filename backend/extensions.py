"""Flask extensions: SQLAlchemy (db) and flask-smorest (api)."""

from flask_sqlalchemy import SQLAlchemy
from flask_smorest import Api

from backend.schemas.errors import ErrorSchema


class PandectsApi(Api):
    # Documents the `{"error", "message"}` envelope that `backend/core/errors.py`
    # returns, in place of flask-smorest's `{code, status, message, errors}`.
    # Spec-only: our error handlers, not flask-smorest's, build error bodies.
    ERROR_SCHEMA = ErrorSchema


db = SQLAlchemy()
api = PandectsApi()
