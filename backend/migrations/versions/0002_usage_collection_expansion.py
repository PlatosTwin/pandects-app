"""Usage-data collection expansion.

Adds the tables and columns that make non-API-key usage visible to the
usage-analytics dashboard:

- ``mcp_usage_hourly``: hourly rollup of MCP tool calls (user + client + tool).
- ``web_usage_hourly``: hourly rollup of session-authenticated web traffic.
- ``page_views``: first-party SPA page views.
- ``auth_signup_attributions``: referrer/UTM capture at account creation.
- ``country`` columns on ``api_usage_daily_ips`` / ``api_request_events`` for
  coarse geo alongside the existing ip_hash.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-19

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('mcp_usage_hourly',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('client_id', sa.String(length=128), nullable=False),
    sa.Column('hour', sa.DateTime(), nullable=False),
    sa.Column('tool_name', sa.String(length=128), nullable=False),
    sa.Column('status', sa.String(length=32), nullable=False),
    sa.Column('count', sa.Integer(), nullable=False),
    sa.Column('total_ms', sa.Integer(), nullable=False),
    sa.Column('max_ms', sa.Integer(), nullable=False),
    sa.Column('latency_buckets', sa.JSON(), nullable=True),
    sa.Column('request_bytes', sa.Integer(), nullable=False),
    sa.Column('response_bytes', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['auth_users.id']),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'client_id', 'hour', 'tool_name', 'status'),
    )
    op.create_index(op.f('ix_mcp_usage_hourly_user_id'), 'mcp_usage_hourly', ['user_id'])
    op.create_index(op.f('ix_mcp_usage_hourly_hour'), 'mcp_usage_hourly', ['hour'])
    op.create_index('ix_mcp_usage_hourly_tool_hour', 'mcp_usage_hourly', ['tool_name', 'hour'])

    op.create_table('web_usage_hourly',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
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
    sa.ForeignKeyConstraint(['user_id'], ['auth_users.id']),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'hour', 'route', 'method', 'status_class'),
    )
    op.create_index(op.f('ix_web_usage_hourly_user_id'), 'web_usage_hourly', ['user_id'])
    op.create_index(op.f('ix_web_usage_hourly_hour'), 'web_usage_hourly', ['hour'])
    op.create_index('ix_web_usage_hourly_route_method', 'web_usage_hourly', ['route', 'method'])

    op.create_table('page_views',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('occurred_at', sa.DateTime(), nullable=False),
    sa.Column('path', sa.String(length=512), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=True),
    sa.Column('referrer', sa.String(length=512), nullable=True),
    sa.Column('country', sa.String(length=8), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['auth_users.id']),
    sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_page_views_occurred_at'), 'page_views', ['occurred_at'])
    op.create_index(op.f('ix_page_views_user_id'), 'page_views', ['user_id'])
    op.create_index('ix_page_views_path_time', 'page_views', ['path', 'occurred_at'])

    op.create_table('auth_signup_attributions',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('referrer', sa.String(length=512), nullable=True),
    sa.Column('landing_path', sa.String(length=512), nullable=True),
    sa.Column('utm_source', sa.String(length=255), nullable=True),
    sa.Column('utm_medium', sa.String(length=255), nullable=True),
    sa.Column('utm_campaign', sa.String(length=255), nullable=True),
    sa.Column('utm_term', sa.String(length=255), nullable=True),
    sa.Column('utm_content', sa.String(length=255), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['auth_users.id']),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id'),
    )

    op.add_column('api_usage_daily_ips', sa.Column('country', sa.String(length=8), nullable=True))
    op.add_column('api_request_events', sa.Column('country', sa.String(length=8), nullable=True))


def downgrade() -> None:
    op.drop_column('api_request_events', 'country')
    op.drop_column('api_usage_daily_ips', 'country')
    op.drop_table('auth_signup_attributions')
    op.drop_index('ix_page_views_path_time', table_name='page_views')
    op.drop_index(op.f('ix_page_views_user_id'), table_name='page_views')
    op.drop_index(op.f('ix_page_views_occurred_at'), table_name='page_views')
    op.drop_table('page_views')
    op.drop_index('ix_web_usage_hourly_route_method', table_name='web_usage_hourly')
    op.drop_index(op.f('ix_web_usage_hourly_hour'), table_name='web_usage_hourly')
    op.drop_index(op.f('ix_web_usage_hourly_user_id'), table_name='web_usage_hourly')
    op.drop_table('web_usage_hourly')
    op.drop_index('ix_mcp_usage_hourly_tool_hour', table_name='mcp_usage_hourly')
    op.drop_index(op.f('ix_mcp_usage_hourly_hour'), table_name='mcp_usage_hourly')
    op.drop_index(op.f('ix_mcp_usage_hourly_user_id'), table_name='mcp_usage_hourly')
    op.drop_table('mcp_usage_hourly')
