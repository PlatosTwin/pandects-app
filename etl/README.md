## Stage—Stage agreements

**Description**: Checks EDGAR for new filings, and stages for filings for ingestion by writing to a temp staging file.

**Output**:
* EDGAR link
* Filing metadata
    * Acquirer
    * Target
    * Date signed
    * Transaction price
    * Transaction type
    * Consideration type
    * Target type

**Tables**:
* pdx.staging
    * agreement_uuid
* pdf.agreements
    * agreement_uuid
    * url
    * target
    * acquirer
    * transaction_date
    * transaction_price
    * transaction_type
    * transaction_consideration
    * consideration_type
    * target_type
    * processed

## Stage 2—Pre-process staged agreements

**Description**: Pulls staged agreements, splits agreements into pages, classifies page type, and processes HTML into formatted text, in preparation for LLM tagging in next stage.

**Output**:
* Main body pages only.
    * Agreement UUID
    * Page UUID
    * Formatted text

**Tables**:
* pdx.pages
    * agreement_uuid
    * page_uuid
    * page_order
    * raw_page_content
    * processed_page_content
    * source_is_txt
    * source_is_html
    * source_page_type
    * page_type_prob_front_matter
    * page_type_prob_toc
    * page_type_prob_body 

## Stage 3—Tag pre-processed agreements via LLM

**Description**: Feeds pre-processed page batches to LLM for tagging.

**Output**:
* Agreement UUID
* Page UUID
* LLM output

**Tables**:
* pdx.llm_output
    * page_uuid
    * prompt_id
    * llm_output
    * cost
    * model
    * full_metadata
* pdx.prompts
    * prompt_id
    * prompt_description
    * prompt_text

## Stage 4—Assemble XML from LLM output

**Description**: Compiles LLM-tagged outputs into agreement XML

**Output**:
* Agreement UUID
* XML

**Tables**:
* pdx.xml
    * agreement_uuid
    * xml
    * version
    * updated_at

## Stage 5—Assign sections to taxonomy

**Description**: Assigns sections to taxonomy, and republishes updated XML

**Output**:
* Agreement UUID
* XML

**Tables**:
* pdx.sections
    * agreement_uuid
    * section_uuid
    * article_title
    * article_title_normed
    * section_title
    * section_title_normed
    * xml_content
    * article_standard_id
    * section_standard_id
    * article_title_embedding
    * section_title_embedding
    * xml_content_embedding
* pdx.taxonomy
    * type
    * standard_id
    * description