-- Duplicate-agreement guard: persisted content signatures, pending-retry state,
-- and the review queue for content hits that lack cover-identity confirmation.
-- Idempotent; apply to the pdx schema. The ETL asset asserts these tables exist
-- (etl/src/etl/utils/schema_guards.py) and never issues DDL itself.
-- `etl/src/etl/utils/backfill_agreement_signatures.py --create-tables` applies
-- this same DDL programmatically.

CREATE TABLE IF NOT EXISTS agreement_signatures (
    agreement_uuid CHAR(36) NOT NULL PRIMARY KEY,
    url VARCHAR(512) NOT NULL,
    content_fingerprint CHAR(64) NOT NULL,
    minhash_json MEDIUMTEXT NOT NULL,
    shingle_count INT NOT NULL,
    char_count BIGINT NOT NULL,
    page_count INT NULL,
    dated_as_of DATE NULL,
    party_tokens_json TEXT NULL,
    amends_and_restates TINYINT(1) NOT NULL DEFAULT 0,
    computed_at DATETIME NOT NULL DEFAULT UTC_TIMESTAMP(),
    KEY idx_agreement_signatures_fingerprint (content_fingerprint),
    KEY idx_agreement_signatures_dated_as_of (dated_as_of)
);

CREATE TABLE IF NOT EXISTS agreement_signature_pending (
    agreement_uuid CHAR(36) NOT NULL PRIMARY KEY,
    url VARCHAR(512) NOT NULL,
    reason VARCHAR(255) NOT NULL,
    attempts INT NOT NULL DEFAULT 0,
    first_recorded_at DATETIME NOT NULL DEFAULT UTC_TIMESTAMP(),
    last_attempt_at DATETIME NULL,
    last_error TEXT NULL
);

CREATE TABLE IF NOT EXISTS agreement_dedupe_review (
    review_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    created_at DATETIME NOT NULL DEFAULT UTC_TIMESTAMP(),
    new_agreement_uuid CHAR(36) NOT NULL,
    new_url VARCHAR(512) NOT NULL,
    existing_agreement_uuid CHAR(36) NOT NULL,
    existing_url VARCHAR(512) NOT NULL,
    jaccard DOUBLE NULL,
    containment DOUBLE NULL,
    fingerprint_match TINYINT(1) NOT NULL DEFAULT 0,
    reason VARCHAR(255) NOT NULL,
    resolved TINYINT(1) NOT NULL DEFAULT 0,
    UNIQUE KEY uq_agreement_dedupe_review_pair (new_agreement_uuid, existing_agreement_uuid)
);
