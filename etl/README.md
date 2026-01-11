## Overview
1. Stage 1: stage agreements ([defs/staging](src/etl/defs/a_staging_asset.py))
    * Identify and pull new agreements from EDGAR
2. Stage 2: pre-process staged agreements ([defs/pre_process](src/etl/defs/b_pre_processing_asset.py))
    * Split to pages
    * Classify pages
    * Format text
3. Stage 3: tag entities (via NER model) ([defs/tagging](src/etl/defs/c_tagging_asset.py))
    * Feed `body` pages to NER model
4. Stage 4: validate uncertain spans ([defs/ai_repair](src/etl/defs/d_ai_repair_asset.py))
    * Feed uncertain spans to LLM to validate
5. Stage 5: reconcile tags ([defs/reconcile_tags](src/etl/defs/e_reconcile_tages_asset.py))
6. Stage 6: create XML ([defs/xml](src/etl/defs/xml_asset.py))
    * Assemble XML from tagged pages
7. Stage 7: split XML to sections ([defs/sections](src/etl/defs/g_sections_asset.py))
    * Splits XML into sections and upsert to DB
8. Stage 8: taxonomize sections ([defs/taxonomy](src/etl/defs/h_taxonomy_asset.py))
    * Assign each section to a taxonomy and update table + XML accordingly
9. Stage 9: Enrich ([defs/taxonomy](src/etl/defs/h_taxonomy_asset.py))
    * Enrich each agreement with metadata, e.g., transaction value, party industries, etc.

## Running the ETL with config

Run config is centralized in `etl/configs/etl_pipeline.yaml`. Edit `mode`, `scope`,
and batch sizes there, then execute:

```bash
dagster job execute -f etl/src/etl/defs/jobs.py -j etl_pipeline -c etl/configs/etl_pipeline.yaml
```

If you're using `dg dev`, open the asset (or job) in the UI, click the arrow near Materialize, open the Launchpad, and edit config values in the JSON block, then click Materialize.

## Stage 1—Stage agreements

**Description**: Checks <em>n</em> days' worth of EDGAR daily index files since the last daily index file we checked. E.g., if the last asset ran up to and including 1/13/25, the next run would begin by looking for the daily index for 1/14/25. Sometimes daily index files contain duplicate filings—i.e., two companies each filed the same agreement. We identify duplicate filings using minhash and take as the primary filing either the filing that has pages, or else if both have or do not have pages, then the filing that appears earlier in the index file.

**Models**: This stage uses the [Exhibit Model](etl/src/etl/models/exhibit_classifier/) to identify Exhibit 2 and Exhibit 10 filings as M&A agreements vs. not. For each agreement identified as a positive sample, we store the model's output probability.

**Validation**: We manually validate all agreements with probability <= 0.75. Agreements flagged for validation do not move on in the pipeline until validated.

**Output**:
* Agreement UUID (generated from the url)
* EDGAR link
* Associated SEC form
* Exhibit number
* Filing company's name + CIK

**ETL processes**:
* `insert into pdx.agreements` + `on duplicate key update`

**Tables**:
* pdf.agreements
    * agreement_uuid
    * url
    * filing_date
    * prob_filing
    * filing_company_name
    * filing_company_cik
    * form_type
    * exhibit_type

## Stage 2—Pre-process staged agreements

**Description**: Pulls the .txt or .html source of all agreements with URLs in `pdx.agreements` but no pages in `pdx.pages`, then splits paginated agreements into pages, classifies page type, and processes HTML into formatted text.

**Models**: This stage uses the [Page Classifier Model](etl/src/etl/models/page_classifier/) to classify pages into one of five classes: `front_matter`, `toc`, `body`, `sig`, `back_matter`.

**Validation**: We manually validate all agreements where either:
1. Page labels are applied out of order—e.g., `back_matter` comes before `sig`; or
2. There is at least one low-confidence page prior to a high-confidence `sig` block (we trust that high-confidence `sig` blocks are accurate, and thus that all pages after the `sig` block are safely assumed to be `back_matter`). Agreements with pages flagged for validation do not get ingested until validated.

**Output**:
* Main body pages only.
    * Agreement UUID
    * Page UUID (auto-increments)
    * Raw page content
    * Formatted page content
    * Predicted class
    * Probabilities for all classes

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
    * page_type_prob_back_matter

## Stage 3—Tag pre-processed agreements via NER

**Description**: Feeds pre-processed page batches to the NER model.

**Models**: This stage uses the [NER Model](etl/src/etl/models/ner/) to identify Article, Section, and Page entities in `body` pages.

**Validation**: All validation happens as part of Stage 4.

**Output**:
* Agreement UUID
* Page UUID
* NER model output: tagged text + uncertain spans

**Tables**:
* pdx.tagged_outputs
    * page_uuid
    * tagged_output
    * low_count
    * spans
    * tokens

## Stage 4—Validate uncertain NER spans via LLM
**Description**: Sends uncertain NER spans to an LLM for validation, and then reconciles NER tags with LLM-validated tags. Depending on the density of uncertain spans on a given page, we send to the LLM either a single span with context on either side or the full text of the page.

**Models**: OpenAI's `gpt-5-mini` for individual spans and `gpt-5` for full pages.

**Validation**: Page with span conflicts are labeled as such, and reviewed manually. Agreements containing such pages do not move on to the next stage until validated.

**Output**:
* Page UUID
* Either span ruling or full-page tagged text

**Tables**:
* pdx.ai_repair_requests
* pdx.ai_repair_batches
* pdx.ai_repair_rulings
    * page_uid
    * start_char
    * end_char
    * label
* pdx.ai_repair_full_pages
    * page_uuid
    * tagged_text

## Stage 5—Assemble XML from tagged text

**Description**: Compiles tagged outputs into agreement XML

**Models**: None.

**Validation**: Methodology pending...

**Output**:
* Agreement UUID
* XML

**Tables**:
* pdx.xml
    * agreement_uuid
    * xml

**Tables**:
* pdx.xml
    * agreement_uuid
    * xml
    * version
    * updated_at (pending)

## Stage 6—Split XML into sections

**Description**: Splits XML into sections.

**Models**: None.

**Validation**: Methodology pending...

**Output**:
* Agreement UUID
* XML

**Tables**:
* pdx.sections
    * agreement_uuid
    * section_uuid
    * article_title
    * article_title_normed
    * article_order
    * section_title
    * section_title_normed
    * section_order
    * xml_content

## Stage 7—Assign sections into taxonomy

**Description**: Assigns individual section to classes from a taxonomy.

**Models**: This stage uses the [Taxonomy Model](etl/src/etl/models/taxonomy/) to associate sections with one or more classes in the Pandects taxonomy.

**Validation**: Methodology pending...

**Output**:
* Section UUID
* Taxonomy class or classes

**Tables**:
* pdx.sections
    * section_standard_id

## Stage 8—Enrich agreement metadata

**Description**: Uses an LLM with web-search functionality to enrich agreements with transaction metadata, such as total transaction price, party industries, consideration types, etc.

**Models**: OpenAI's `gpt-5.1` with web-search tooling.

**Validation**: Methodology pending...

**Output**:
* Considertion fields
    * Value: total, stock, cash, assets
    * Type: stock, cash, assets, mixed, unknown
* Party fields
    * Target name, industry, public/private, and whether it's owned by a PE shop
    * Acquirer name, industry, public/private, and whether it's a PE shop
* Deal fields
    * Announcement date
    * Close date
    * Deal status
    * Attitude (friendly, hostile, unsolicited)
    * Deal type
    * Purpose


**Tables**:
* pdx.agreements
    * transaction_price_total
    * transaction_price_stock
    * transaction_price_stock
    * transaction_price_cash
    * transaction_price_assets
    * transaction_consideration
    * target_type
    * acquirer_type
    * target_industry
    * acquirer_industry
    * announce_date
    * close_date
    * deal_status
    * attitude
    * deal_type
    * purpose
    * target_pe
    * acquirer_pe