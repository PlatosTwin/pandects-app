CREATE TABLE section_text_search (
    section_uuid CHAR(36) NOT NULL,
    agreement_uuid CHAR(36) NOT NULL,
    xml_version INT NULL,
    source_xml_sha256 BINARY(32) NOT NULL,
    normalized_text LONGTEXT COLLATE utf8mb4_unicode_ci NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (section_uuid),
    KEY idx_section_text_search_agreement_uuid (agreement_uuid),
    FULLTEXT KEY ft_section_text_search_normalized_text (normalized_text)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  ROW_FORMAT=DYNAMIC;
