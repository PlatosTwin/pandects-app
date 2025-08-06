## Stage 1—Stage agreements

**Description**: Checks EDGAR for new filings, and stages for filings for ingestion by writing to a temp staging file. Set `processed = 0` until XML is generated in Step 4.

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

**ETL processes**:
* `insert into pdx.agreements` + `on duplicate key update`

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

**Description**: Pulls staged agreements, splits agreements into pages, classifies page type, and processes HTML into formatted text, in preparation for LLM tagging in next stage. Set `processed = 0` until XML is generated in Step 4.

**Output**:
* Main body pages only.
    * Agreement UUID
    * Page UUID
    * Formatted text

**ETL processes**:
* Select all _unprocessed_ agreements (`processed = 0`)
* Split into pages, and format and classify pages
* `insert into pdx.pages` + `on duplicate key update`
* Note: MariaDB generates page UUIDs automatically

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
    * page_type_prob_sig 
    * processed

## Stage 3—Tag pre-processed agreements via NER and LLM

**Description**: Feeds pre-processed page batches to the NER model and then uncertain spans to an LLM.

**Output**:
* Agreement UUID
* Page UUID
* LLM output

**ETL processes**:
* Select all _unprocessed_ pages (`processed = 0`)
* Run through tagging model
* `insert into pdx.tagged_outputs` + `on duplicate key update`
* Set `processed = 1` for all pages successfully tagged

**Tables**:
* pdx.tagged_outputs
    * page_uuid
    * tagged_output
    * uncertain_spans

**Tables (TBD)**:
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

## Stage 4—Assemble XML from LLM output, including taxonimizing

**Description**: Compiles LLM-tagged outputs into agreement XML, including adding taxonomy labels

**Output**:
* Agreement UUID
* XML (with taxonomy labels)

**ETL processes**:
* Select all tagged output for _unprocessed_ agreements (`processed = 0`)
* Run through XML generation functions
* `insert into pdx.xml`
* Set `processed = 1` for all agreements successfully XML'd

**Tables**:
* pdx.xml
    * agreement_uuid
    * xml
    * version
    * updated_at
* pdx.taxonomy
    * type
    * standard_id
    * description

## Stage 5—Splits XML into sections in the database

**Description**: Splits XML into sections in the database

**Output**:
* Agreement UUID
* XML

**ETL processes**:
* Select all _unprocessed_ XML (`processed = 0`)
* Run through taxonomy functions
* Update pdx.xml
* Set `processed = 1` for all XML successfully taxonomized

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