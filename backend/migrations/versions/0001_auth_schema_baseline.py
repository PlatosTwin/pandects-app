"""Auth schema baseline.

Single migration that brings any auth database to the current schema:

- A fresh database (no ``auth_users`` table) gets the full schema created.
- An existing database gets the legacy catch-up previously performed at app
  startup by ``ensure_auth_schema_upgrades``: any missing tables are created,
  and columns added after the original rollout (``api_keys.deleted_at``,
  ``auth_oauth_clients.last_used_at``, ``auth_oauth_refresh_tokens.family_id``
  with backfill, ``favorite_projects.sort_order`` / ``color``) are added when
  absent.

Both branches are idempotent with respect to the current production and dev
databases; after this revision, schema changes are normal Alembic deltas.

Revision ID: 0001
Revises:
Create Date: 2026-06-11

"""
from __future__ import annotations

from collections.abc import Callable

from alembic import op
import sqlalchemy as sa


revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def _create_auth_oauth_clients() -> None:
    op.create_table('auth_oauth_clients',
    sa.Column('client_id', sa.String(length=128), nullable=False),
    sa.Column('client_name', sa.String(length=255), nullable=True),
    sa.Column('redirect_uris', sa.JSON(), nullable=False),
    sa.Column('token_endpoint_auth_method', sa.String(length=32), nullable=False),
    sa.Column('grant_types', sa.JSON(), nullable=False),
    sa.Column('response_types', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('created_by_ip', sa.String(length=64), nullable=True),
    sa.Column('last_used_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('client_id')
    )


def _create_auth_oauth_signing_keys() -> None:
    op.create_table('auth_oauth_signing_keys',
    sa.Column('kid', sa.String(length=128), nullable=False),
    sa.Column('algorithm', sa.String(length=16), nullable=False),
    sa.Column('private_pem', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('activated_at', sa.DateTime(), nullable=False),
    sa.Column('active', sa.Boolean(), nullable=False),
    sa.PrimaryKeyConstraint('kid')
    )


def _create_auth_users() -> None:
    op.create_table('auth_users',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('email', sa.String(length=320), nullable=False),
    sa.Column('password_hash', sa.Text(), nullable=True),
    sa.Column('email_verified_at', sa.DateTime(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('auth_users', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_auth_users_email'), ['email'], unique=True)


def _create_api_keys() -> None:
    op.create_table('api_keys',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('name', sa.String(length=120), nullable=True),
    sa.Column('prefix', sa.String(length=18), nullable=False),
    sa.Column('key_hash', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('last_used_at', sa.DateTime(), nullable=True),
    sa.Column('revoked_at', sa.DateTime(), nullable=True),
    sa.Column('deleted_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['auth_users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_api_keys_prefix'), ['prefix'], unique=False)
        batch_op.create_index(batch_op.f('ix_api_keys_user_id'), ['user_id'], unique=False)


def _create_auth_external_subjects() -> None:
    op.create_table('auth_external_subjects',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('issuer', sa.String(length=255), nullable=False),
    sa.Column('subject', sa.String(length=255), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['auth_users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('issuer', 'subject')
    )
    with op.batch_alter_table('auth_external_subjects', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_auth_external_subjects_user_id'), ['user_id'], unique=False)
        batch_op.create_index('ix_auth_external_subjects_user_issuer', ['user_id', 'issuer'], unique=False)


def _create_auth_oauth_authorization_codes() -> None:
    op.create_table('auth_oauth_authorization_codes',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('code_hash', sa.String(length=64), nullable=False),
    sa.Column('client_id', sa.String(length=128), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('redirect_uri', sa.Text(), nullable=False),
    sa.Column('scope', sa.Text(), nullable=False),
    sa.Column('code_challenge', sa.String(length=255), nullable=False),
    sa.Column('code_challenge_method', sa.String(length=16), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=False),
    sa.Column('used_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['client_id'], ['auth_oauth_clients.client_id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['auth_users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('auth_oauth_authorization_codes', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_auth_oauth_authorization_codes_client_id'), ['client_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_auth_oauth_authorization_codes_code_hash'), ['code_hash'], unique=True)
        batch_op.create_index(batch_op.f('ix_auth_oauth_authorization_codes_user_id'), ['user_id'], unique=False)


def _create_auth_oauth_refresh_tokens() -> None:
    op.create_table('auth_oauth_refresh_tokens',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('client_id', sa.String(length=128), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('scope', sa.Text(), nullable=False),
    sa.Column('family_id', sa.String(length=36), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=False),
    sa.Column('used_at', sa.DateTime(), nullable=True),
    sa.Column('revoked_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['client_id'], ['auth_oauth_clients.client_id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['auth_users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('auth_oauth_refresh_tokens', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_auth_oauth_refresh_tokens_client_id'), ['client_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_auth_oauth_refresh_tokens_family_id'), ['family_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_auth_oauth_refresh_tokens_token_hash'), ['token_hash'], unique=True)
        batch_op.create_index('ix_auth_oauth_refresh_tokens_user_client', ['user_id', 'client_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_auth_oauth_refresh_tokens_user_id'), ['user_id'], unique=False)


def _create_auth_oauth_user_grants() -> None:
    op.create_table('auth_oauth_user_grants',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('client_id', sa.String(length=128), nullable=False),
    sa.Column('scope', sa.Text(), nullable=False),
    sa.Column('granted_at', sa.DateTime(), nullable=False),
    sa.Column('revoked_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['client_id'], ['auth_oauth_clients.client_id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['auth_users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'client_id', name='uq_oauth_user_grant')
    )
    with op.batch_alter_table('auth_oauth_user_grants', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_auth_oauth_user_grants_client_id'), ['client_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_auth_oauth_user_grants_user_id'), ['user_id'], unique=False)


def _create_auth_password_reset_tokens() -> None:
    op.create_table('auth_password_reset_tokens',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=False),
    sa.Column('used_at', sa.DateTime(), nullable=True),
    sa.Column('ip_address', sa.String(length=64), nullable=True),
    sa.Column('user_agent', sa.String(length=512), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['auth_users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('auth_password_reset_tokens', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_auth_password_reset_tokens_user_id'), ['user_id'], unique=False)


def _create_auth_sessions() -> None:
    op.create_table('auth_sessions',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=False),
    sa.Column('revoked_at', sa.DateTime(), nullable=True),
    sa.Column('last_used_at', sa.DateTime(), nullable=True),
    sa.Column('ip_address', sa.String(length=64), nullable=True),
    sa.Column('user_agent', sa.String(length=512), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['auth_users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('auth_sessions', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_auth_sessions_token_hash'), ['token_hash'], unique=True)
        batch_op.create_index(batch_op.f('ix_auth_sessions_user_id'), ['user_id'], unique=False)


def _create_auth_signon_events() -> None:
    op.create_table('auth_signon_events',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('provider', sa.String(length=32), nullable=False),
    sa.Column('action', sa.String(length=32), nullable=False),
    sa.Column('occurred_at', sa.DateTime(), nullable=False),
    sa.Column('ip_address', sa.String(length=64), nullable=True),
    sa.Column('user_agent', sa.String(length=512), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['auth_users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('auth_signon_events', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_auth_signon_events_user_id'), ['user_id'], unique=False)


def _create_favorite_projects() -> None:
    op.create_table('favorite_projects',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('name', sa.String(length=120), nullable=False),
    sa.Column('color', sa.String(length=16), nullable=False),
    sa.Column('is_default', sa.Boolean(), nullable=False),
    sa.Column('sort_order', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['auth_users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'name', name='uq_favorite_projects_user_name')
    )
    with op.batch_alter_table('favorite_projects', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_favorite_projects_user_id'), ['user_id'], unique=False)


def _create_favorite_tags() -> None:
    op.create_table('favorite_tags',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('name', sa.String(length=64), nullable=False),
    sa.Column('color', sa.String(length=16), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['auth_users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'name', name='uq_favorite_tags_user_name')
    )
    with op.batch_alter_table('favorite_tags', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_favorite_tags_user_id'), ['user_id'], unique=False)


def _create_legal_acceptances() -> None:
    op.create_table('legal_acceptances',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('document', sa.String(length=24), nullable=False),
    sa.Column('version', sa.String(length=64), nullable=False),
    sa.Column('document_hash', sa.String(length=64), nullable=True),
    sa.Column('checked_at', sa.DateTime(), nullable=False),
    sa.Column('submitted_at', sa.DateTime(), nullable=False),
    sa.Column('ip_address', sa.String(length=64), nullable=True),
    sa.Column('user_agent', sa.String(length=512), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['auth_users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('legal_acceptances', schema=None) as batch_op:
        batch_op.create_index('ix_legal_acceptances_user_doc_ver', ['user_id', 'document', 'version'], unique=False)
        batch_op.create_index(batch_op.f('ix_legal_acceptances_user_id'), ['user_id'], unique=False)


def _create_api_request_events() -> None:
    op.create_table('api_request_events',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('api_key_id', sa.String(length=36), nullable=False),
    sa.Column('occurred_at', sa.DateTime(), nullable=False),
    sa.Column('route', sa.String(length=256), nullable=False),
    sa.Column('method', sa.String(length=8), nullable=False),
    sa.Column('status_code', sa.Integer(), nullable=False),
    sa.Column('status_class', sa.Integer(), nullable=False),
    sa.Column('latency_ms', sa.Integer(), nullable=False),
    sa.Column('request_bytes', sa.Integer(), nullable=True),
    sa.Column('response_bytes', sa.Integer(), nullable=True),
    sa.Column('ip_hash', sa.String(length=64), nullable=True),
    sa.Column('user_agent', sa.String(length=512), nullable=True),
    sa.ForeignKeyConstraint(['api_key_id'], ['api_keys.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('api_request_events', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_api_request_events_api_key_id'), ['api_key_id'], unique=False)
        batch_op.create_index('ix_api_request_events_ip_time', ['ip_hash', 'occurred_at'], unique=False)
        batch_op.create_index('ix_api_request_events_key_time', ['api_key_id', 'occurred_at'], unique=False)
        batch_op.create_index(batch_op.f('ix_api_request_events_occurred_at'), ['occurred_at'], unique=False)


def _create_api_usage_daily() -> None:
    op.create_table('api_usage_daily',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('api_key_id', sa.String(length=36), nullable=False),
    sa.Column('day', sa.Date(), nullable=False),
    sa.Column('count', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['api_key_id'], ['api_keys.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('api_key_id', 'day')
    )
    with op.batch_alter_table('api_usage_daily', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_api_usage_daily_api_key_id'), ['api_key_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_api_usage_daily_day'), ['day'], unique=False)


def _create_api_usage_daily_ips() -> None:
    op.create_table('api_usage_daily_ips',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('api_key_id', sa.String(length=36), nullable=False),
    sa.Column('day', sa.Date(), nullable=False),
    sa.Column('ip_hash', sa.String(length=64), nullable=False),
    sa.Column('first_seen_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['api_key_id'], ['api_keys.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('api_key_id', 'day', 'ip_hash')
    )
    with op.batch_alter_table('api_usage_daily_ips', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_api_usage_daily_ips_api_key_id'), ['api_key_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_api_usage_daily_ips_day'), ['day'], unique=False)
        batch_op.create_index('ix_api_usage_daily_ips_key_day', ['api_key_id', 'day'], unique=False)


def _create_api_usage_hourly() -> None:
    op.create_table('api_usage_hourly',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('api_key_id', sa.String(length=36), nullable=False),
    sa.Column('hour', sa.DateTime(), nullable=False),
    sa.Column('route', sa.String(length=256), nullable=False),
    sa.Column('method', sa.String(length=8), nullable=False),
    sa.Column('status_class', sa.Integer(), nullable=False),
    sa.Column('count', sa.Integer(), nullable=False),
    sa.Column('total_ms', sa.Integer(), nullable=False),
    sa.Column('max_ms', sa.Integer(), nullable=False),
    sa.Column('latency_buckets', sa.JSON(), nullable=True),
    sa.Column('request_bytes', sa.Integer(), nullable=False),
    sa.Column('response_bytes', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['api_key_id'], ['api_keys.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('api_key_id', 'hour', 'route', 'method', 'status_class')
    )
    with op.batch_alter_table('api_usage_hourly', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_api_usage_hourly_api_key_id'), ['api_key_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_api_usage_hourly_hour'), ['hour'], unique=False)
        batch_op.create_index('ix_api_usage_hourly_route_method', ['route', 'method'], unique=False)


def _create_favorites() -> None:
    op.create_table('favorites',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('project_id', sa.String(length=36), nullable=False),
    sa.Column('item_type', sa.String(length=16), nullable=False),
    sa.Column('item_uuid', sa.String(length=36), nullable=False),
    sa.Column('note', sa.Text(), nullable=True),
    sa.Column('context', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['favorite_projects.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['auth_users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'item_type', 'item_uuid', name='uq_favorites_user_item')
    )
    with op.batch_alter_table('favorites', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_favorites_project_id'), ['project_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_favorites_user_id'), ['user_id'], unique=False)
        batch_op.create_index('ix_favorites_user_type', ['user_id', 'item_type'], unique=False)


def _create_favorite_project_assignments() -> None:
    op.create_table('favorite_project_assignments',
    sa.Column('favorite_id', sa.String(length=36), nullable=False),
    sa.Column('project_id', sa.String(length=36), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['favorite_id'], ['favorites.id'], ),
    sa.ForeignKeyConstraint(['project_id'], ['favorite_projects.id'], ),
    sa.PrimaryKeyConstraint('favorite_id', 'project_id')
    )
    with op.batch_alter_table('favorite_project_assignments', schema=None) as batch_op:
        batch_op.create_index('ix_favorite_project_assignments_project', ['project_id'], unique=False)


def _create_favorite_tag_assignments() -> None:
    op.create_table('favorite_tag_assignments',
    sa.Column('favorite_id', sa.String(length=36), nullable=False),
    sa.Column('tag_id', sa.String(length=36), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['favorite_id'], ['favorites.id'], ),
    sa.ForeignKeyConstraint(['tag_id'], ['favorite_tags.id'], ),
    sa.PrimaryKeyConstraint('favorite_id', 'tag_id')
    )
    with op.batch_alter_table('favorite_tag_assignments', schema=None) as batch_op:
        batch_op.create_index('ix_favorite_tag_assignments_tag', ['tag_id'], unique=False)


# Creation order respects foreign-key dependencies (as emitted by
# autogenerate against the models).
_TABLE_CREATORS: list[tuple[str, Callable[[], None]]] = [
    ("auth_oauth_clients", _create_auth_oauth_clients),
    ("auth_oauth_signing_keys", _create_auth_oauth_signing_keys),
    ("auth_users", _create_auth_users),
    ("api_keys", _create_api_keys),
    ("auth_external_subjects", _create_auth_external_subjects),
    ("auth_oauth_authorization_codes", _create_auth_oauth_authorization_codes),
    ("auth_oauth_refresh_tokens", _create_auth_oauth_refresh_tokens),
    ("auth_oauth_user_grants", _create_auth_oauth_user_grants),
    ("auth_password_reset_tokens", _create_auth_password_reset_tokens),
    ("auth_sessions", _create_auth_sessions),
    ("auth_signon_events", _create_auth_signon_events),
    ("favorite_projects", _create_favorite_projects),
    ("favorite_tags", _create_favorite_tags),
    ("legal_acceptances", _create_legal_acceptances),
    ("api_request_events", _create_api_request_events),
    ("api_usage_daily", _create_api_usage_daily),
    ("api_usage_daily_ips", _create_api_usage_daily_ips),
    ("api_usage_hourly", _create_api_usage_hourly),
    ("favorites", _create_favorites),
    ("favorite_project_assignments", _create_favorite_project_assignments),
    ("favorite_tag_assignments", _create_favorite_tag_assignments),
]


def _column_names(inspector: sa.Inspector, table: str) -> set[str]:
    return {column["name"] for column in inspector.get_columns(table)}


def _apply_legacy_catchup(existing_tables: set[str]) -> None:
    """Port of the startup-time ensure_auth_schema_upgrades for pre-Alembic DBs."""
    bind = op.get_bind()

    for table_name, creator in _TABLE_CREATORS:
        if table_name not in existing_tables:
            creator()

    inspector = sa.inspect(bind)

    if "deleted_at" not in _column_names(inspector, "api_keys"):
        op.add_column("api_keys", sa.Column("deleted_at", sa.DateTime(), nullable=True))

    if "last_used_at" not in _column_names(inspector, "auth_oauth_clients"):
        op.add_column(
            "auth_oauth_clients", sa.Column("last_used_at", sa.DateTime(), nullable=True)
        )

    if "family_id" not in _column_names(inspector, "auth_oauth_refresh_tokens"):
        # Added nullable (matching the legacy upgrade) and backfilled from the
        # token id so every existing token forms its own family.
        op.add_column(
            "auth_oauth_refresh_tokens",
            sa.Column("family_id", sa.String(length=36), nullable=True),
        )
        op.execute(
            sa.text(
                "UPDATE auth_oauth_refresh_tokens SET family_id = id "
                "WHERE family_id IS NULL"
            )
        )

    project_columns = _column_names(inspector, "favorite_projects")
    if "sort_order" not in project_columns:
        op.add_column(
            "favorite_projects",
            sa.Column("sort_order", sa.Integer(), nullable=False, server_default="0"),
        )
    if "color" not in project_columns:
        op.add_column(
            "favorite_projects",
            sa.Column("color", sa.String(length=16), nullable=False, server_default="slate"),
        )


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "auth_users" in existing_tables:
        _apply_legacy_catchup(existing_tables)
        return

    for _table_name, creator in _TABLE_CREATORS:
        creator()


def downgrade() -> None:
    raise NotImplementedError(
        "The auth schema baseline cannot be downgraded; it would drop every "
        "auth table."
    )
