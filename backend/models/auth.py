from __future__ import annotations

import uuid
from datetime import datetime, timezone

from backend.extensions import db


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class AuthUser(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "auth_users"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(320), unique=True, index=True, nullable=False)
    password_hash = db.Column(db.Text, nullable=True)
    email_verified_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=_utc_now_naive)


class AuthExternalSubject(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "auth_external_subjects"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(
        db.String(36), db.ForeignKey("auth_users.id"), index=True, nullable=False
    )
    issuer = db.Column(db.String(255), nullable=False)
    subject = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=_utc_now_naive)

    __table_args__ = (
        db.UniqueConstraint("issuer", "subject"),
        db.Index("ix_auth_external_subjects_user_issuer", "user_id", "issuer"),
    )


class AuthSession(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "auth_sessions"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(
        db.String(36), db.ForeignKey("auth_users.id"), index=True, nullable=False
    )
    token_hash = db.Column(db.String(64), unique=True, index=True, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=_utc_now_naive)
    expires_at = db.Column(db.DateTime, nullable=False)
    revoked_at = db.Column(db.DateTime, nullable=True)
    last_used_at = db.Column(db.DateTime, nullable=True)
    ip_address = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.String(512), nullable=True)


class AuthPasswordResetToken(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "auth_password_reset_tokens"

    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(
        db.String(36), db.ForeignKey("auth_users.id"), index=True, nullable=False
    )
    created_at = db.Column(db.DateTime, nullable=False, default=_utc_now_naive)
    expires_at = db.Column(db.DateTime, nullable=False)
    used_at = db.Column(db.DateTime, nullable=True)
    ip_address = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.String(512), nullable=True)


class ApiKey(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "api_keys"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(
        db.String(36), db.ForeignKey("auth_users.id"), index=True, nullable=False
    )
    name = db.Column(db.String(120), nullable=True)
    prefix = db.Column(db.String(18), index=True, nullable=False)
    key_hash = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=_utc_now_naive)
    last_used_at = db.Column(db.DateTime, nullable=True)
    revoked_at = db.Column(db.DateTime, nullable=True)
    deleted_at = db.Column(db.DateTime, nullable=True)


class ApiUsageDaily(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "api_usage_daily"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    api_key_id = db.Column(
        db.String(36), db.ForeignKey("api_keys.id"), index=True, nullable=False
    )
    day = db.Column(db.Date, index=True, nullable=False)
    count = db.Column(db.Integer, nullable=False, default=0)

    __table_args__ = (db.UniqueConstraint("api_key_id", "day"),)


class ApiUsageHourly(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "api_usage_hourly"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    api_key_id = db.Column(
        db.String(36), db.ForeignKey("api_keys.id"), index=True, nullable=False
    )
    hour = db.Column(db.DateTime, index=True, nullable=False)
    route = db.Column(db.String(256), nullable=False)
    method = db.Column(db.String(8), nullable=False)
    status_class = db.Column(db.Integer, nullable=False)
    count = db.Column(db.Integer, nullable=False, default=0)
    total_ms = db.Column(db.Integer, nullable=False, default=0)
    max_ms = db.Column(db.Integer, nullable=False, default=0)
    latency_buckets = db.Column(db.JSON, nullable=True)
    request_bytes = db.Column(db.Integer, nullable=False, default=0)
    response_bytes = db.Column(db.Integer, nullable=False, default=0)

    __table_args__ = (
        db.UniqueConstraint(
            "api_key_id", "hour", "route", "method", "status_class"
        ),
        db.Index("ix_api_usage_hourly_route_method", "route", "method"),
    )


class ApiRequestEvent(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "api_request_events"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    api_key_id = db.Column(
        db.String(36), db.ForeignKey("api_keys.id"), index=True, nullable=False
    )
    occurred_at = db.Column(db.DateTime, index=True, nullable=False)
    route = db.Column(db.String(256), nullable=False)
    method = db.Column(db.String(8), nullable=False)
    status_code = db.Column(db.Integer, nullable=False)
    status_class = db.Column(db.Integer, nullable=False)
    latency_ms = db.Column(db.Integer, nullable=False)
    request_bytes = db.Column(db.Integer, nullable=True)
    response_bytes = db.Column(db.Integer, nullable=True)
    ip_hash = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.String(512), nullable=True)

    __table_args__ = (
        db.Index("ix_api_request_events_key_time", "api_key_id", "occurred_at"),
        db.Index("ix_api_request_events_ip_time", "ip_hash", "occurred_at"),
    )


class ApiUsageDailyIp(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "api_usage_daily_ips"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    api_key_id = db.Column(
        db.String(36), db.ForeignKey("api_keys.id"), index=True, nullable=False
    )
    day = db.Column(db.Date, index=True, nullable=False)
    ip_hash = db.Column(db.String(64), nullable=False)
    first_seen_at = db.Column(db.DateTime, nullable=False)

    __table_args__ = (
        db.UniqueConstraint("api_key_id", "day", "ip_hash"),
        db.Index("ix_api_usage_daily_ips_key_day", "api_key_id", "day"),
    )


class LegalAcceptance(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "legal_acceptances"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(
        db.String(36), db.ForeignKey("auth_users.id"), index=True, nullable=False
    )
    document = db.Column(db.String(24), nullable=False)
    version = db.Column(db.String(64), nullable=False)
    document_hash = db.Column(db.String(64), nullable=True)
    checked_at = db.Column(db.DateTime, nullable=False)
    submitted_at = db.Column(db.DateTime, nullable=False)
    ip_address = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.String(512), nullable=True)

    __table_args__ = (
        db.Index("ix_legal_acceptances_user_doc_ver", "user_id", "document", "version"),
    )


class AuthSignonEvent(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "auth_signon_events"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(
        db.String(36), db.ForeignKey("auth_users.id"), index=True, nullable=False
    )
    provider = db.Column(db.String(32), nullable=False)
    action = db.Column(db.String(32), nullable=False)
    occurred_at = db.Column(db.DateTime, nullable=False, default=_utc_now_naive)
    ip_address = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.String(512), nullable=True)


class AuthOAuthClient(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "auth_oauth_clients"

    client_id = db.Column(db.String(128), primary_key=True)
    client_name = db.Column(db.String(255), nullable=True)
    redirect_uris = db.Column(db.JSON, nullable=False)
    token_endpoint_auth_method = db.Column(db.String(32), nullable=False, default="none")
    grant_types = db.Column(db.JSON, nullable=False)
    response_types = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=_utc_now_naive)
    created_by_ip = db.Column(db.String(64), nullable=True)


class AuthOAuthAuthorizationCode(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "auth_oauth_authorization_codes"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    code_hash = db.Column(db.String(64), unique=True, index=True, nullable=False)
    client_id = db.Column(
        db.String(128), db.ForeignKey("auth_oauth_clients.client_id"), index=True, nullable=False
    )
    user_id = db.Column(
        db.String(36), db.ForeignKey("auth_users.id"), index=True, nullable=False
    )
    redirect_uri = db.Column(db.Text, nullable=False)
    scope = db.Column(db.Text, nullable=False)
    code_challenge = db.Column(db.String(255), nullable=False)
    code_challenge_method = db.Column(db.String(16), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=_utc_now_naive)
    expires_at = db.Column(db.DateTime, nullable=False)
    used_at = db.Column(db.DateTime, nullable=True)


class FavoriteProject(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "favorite_projects"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(
        db.String(36), db.ForeignKey("auth_users.id"), index=True, nullable=False
    )
    name = db.Column(db.String(120), nullable=False)
    color = db.Column(db.String(16), nullable=False, default="slate")
    is_default = db.Column(db.Boolean, nullable=False, default=False)
    sort_order = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, nullable=False, default=_utc_now_naive)

    __table_args__ = (
        db.UniqueConstraint("user_id", "name", name="uq_favorite_projects_user_name"),
    )


class Favorite(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "favorites"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(
        db.String(36), db.ForeignKey("auth_users.id"), index=True, nullable=False
    )
    project_id = db.Column(
        db.String(36), db.ForeignKey("favorite_projects.id"), index=True, nullable=False
    )
    item_type = db.Column(db.String(16), nullable=False)
    item_uuid = db.Column(db.String(36), nullable=False)
    note = db.Column(db.Text, nullable=True)
    context = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=_utc_now_naive)
    updated_at = db.Column(
        db.DateTime, nullable=False, default=_utc_now_naive, onupdate=_utc_now_naive
    )

    __table_args__ = (
        db.UniqueConstraint(
            "user_id", "item_type", "item_uuid", name="uq_favorites_user_item"
        ),
        db.Index("ix_favorites_user_type", "user_id", "item_type"),
    )


class FavoriteTag(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "favorite_tags"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(
        db.String(36), db.ForeignKey("auth_users.id"), index=True, nullable=False
    )
    name = db.Column(db.String(64), nullable=False)
    color = db.Column(db.String(16), nullable=False, default="slate")
    created_at = db.Column(db.DateTime, nullable=False, default=_utc_now_naive)

    __table_args__ = (
        db.UniqueConstraint("user_id", "name", name="uq_favorite_tags_user_name"),
    )


class FavoriteTagAssignment(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "favorite_tag_assignments"

    favorite_id = db.Column(
        db.String(36), db.ForeignKey("favorites.id"), primary_key=True
    )
    tag_id = db.Column(
        db.String(36), db.ForeignKey("favorite_tags.id"), primary_key=True
    )
    created_at = db.Column(db.DateTime, nullable=False, default=_utc_now_naive)

    __table_args__ = (
        db.Index("ix_favorite_tag_assignments_tag", "tag_id"),
    )


class FavoriteProjectAssignment(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "favorite_project_assignments"

    favorite_id = db.Column(
        db.String(36), db.ForeignKey("favorites.id"), primary_key=True
    )
    project_id = db.Column(
        db.String(36), db.ForeignKey("favorite_projects.id"), primary_key=True
    )
    created_at = db.Column(db.DateTime, nullable=False, default=_utc_now_naive)

    __table_args__ = (
        db.Index("ix_favorite_project_assignments_project", "project_id"),
    )


class AuthOAuthSigningKey(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "auth_oauth_signing_keys"

    kid = db.Column(db.String(128), primary_key=True)
    algorithm = db.Column(db.String(16), nullable=False, default="RS256")
    private_pem = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=_utc_now_naive)
    activated_at = db.Column(db.DateTime, nullable=False, default=_utc_now_naive)
    active = db.Column(db.Boolean, nullable=False, default=True)
