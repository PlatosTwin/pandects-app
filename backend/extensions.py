"""Flask extensions: SQLAlchemy (db) and flask-smorest (api)."""

from flask_sqlalchemy import SQLAlchemy
from flask_smorest import Api

db = SQLAlchemy()
api = Api()
