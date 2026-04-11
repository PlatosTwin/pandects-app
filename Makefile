SHELL := /bin/bash

.PHONY: help backend-test dev-backend-safe frontend-test frontend-typecheck dev-frontend docs-build etl-typecheck

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make backend-test         Run backend unit tests in public-safe mode' \
		'  make dev-backend-safe     Run the Flask backend with main DB reflection disabled' \
		'  make frontend-test        Run frontend tests' \
		'  make frontend-typecheck   Run frontend TypeScript checks' \
		'  make dev-frontend         Start the frontend dev server' \
		'  make docs-build           Build the docs site' \
		'  make etl-typecheck        Run ETL basedpyright checks'

backend-test:
	caffeinate -i env SKIP_MAIN_DB_REFLECTION=1 backend/venv/bin/python -m unittest discover backend/tests -v

dev-backend-safe:
	caffeinate -i env SKIP_MAIN_DB_REFLECTION=1 FLASK_APP=backend.app backend/venv/bin/flask run --port 5113

frontend-test:
	cd frontend && caffeinate -i npm test

frontend-typecheck:
	cd frontend && caffeinate -i npm run typecheck

dev-frontend:
	cd frontend && caffeinate -i npm run dev

docs-build:
	cd docs && caffeinate -i npm run build

etl-typecheck:
	caffeinate -i etl/.venv/bin/basedpyright etl/src
