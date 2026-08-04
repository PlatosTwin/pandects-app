"""MCP agent feedback channel.

Adds ``mcp_feedback``: experience/usage feedback submitted by LLM agents via
the ``submit_feedback`` MCP tool (pain points, missing capabilities,
data-quality issues, praise), keyed to the submitting user and OAuth client.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-04

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('mcp_feedback',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('client_id', sa.String(length=128), nullable=False),
    sa.Column('scopes', sa.Text(), nullable=False),
    sa.Column('category', sa.String(length=32), nullable=False),
    sa.Column('severity', sa.String(length=16), nullable=True),
    sa.Column('tool_name', sa.String(length=128), nullable=True),
    sa.Column('summary', sa.String(length=200), nullable=False),
    sa.Column('detail', sa.Text(), nullable=False),
    sa.Column('suggestions', sa.Text(), nullable=True),
    sa.Column('context', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['auth_users.id']),
    sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_mcp_feedback_user_id'), 'mcp_feedback', ['user_id'])
    op.create_index(op.f('ix_mcp_feedback_created_at'), 'mcp_feedback', ['created_at'])
    op.create_index('ix_mcp_feedback_user_time', 'mcp_feedback', ['user_id', 'created_at'])


def downgrade() -> None:
    op.drop_index('ix_mcp_feedback_user_time', table_name='mcp_feedback')
    op.drop_index(op.f('ix_mcp_feedback_created_at'), table_name='mcp_feedback')
    op.drop_index(op.f('ix_mcp_feedback_user_id'), table_name='mcp_feedback')
    op.drop_table('mcp_feedback')
