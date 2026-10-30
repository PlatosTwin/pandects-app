-- Smart reader index (docs/plans/smart-reader.md, Section 6.2): one row per
-- agreement XML version, computed by the 12_reader_index asset from latest
-- verified XML. The asset asserts this table exists and never issues DDL.
-- Idempotent; apply to the pdx schema. Before it reaches prod it also needs
-- bulk/schema_docs/table_docs.yml coverage and the serving schema (Phase 2).

CREATE TABLE IF NOT EXISTS reader_index (
    agreement_uuid CHAR(36) NOT NULL,
    xml_version INT NOT NULL,
    enricher_version VARCHAR(16) NOT NULL,
    text_hash CHAR(64) NOT NULL,
    payload_gz LONGBLOB NOT NULL,
    n_terms INT NOT NULL,
    n_term_uses INT NOT NULL,
    n_xrefs INT NOT NULL,
    n_xrefs_resolved INT NOT NULL,
    n_xrefs_unresolved INT NOT NULL,
    body_section_level_rate DECIMAL(5,4) NULL,
    created_date DATETIME(3) NOT NULL DEFAULT UTC_TIMESTAMP(3),
    PRIMARY KEY (agreement_uuid, xml_version),
    KEY idx_reader_index_enricher (enricher_version),
    KEY idx_reader_index_rate (body_section_level_rate)
);
