# M1, M2 and M2 holdout pilots (span taxonomy)

Results and discussion: [section 7.4](../../span-taxonomy.md#74-m1-pilot-results-2026-10-21) (M1, 3 concepts, 50 agreements) and [section 7.5](../../span-taxonomy.md#75-m2-pilot-results-2026-10-21) (M2, 10 concepts, 150 agreements) and [section 7.6](../../span-taxonomy.md#76-m2-holdout-results-2026-10-21) (M2 holdout, 50 fresh agreements) and [section 7.7](../../span-taxonomy.md#77-definition-resolution-re-check-2026-10-21) (definition-resolution re-check on the same holdout) of the plan.

## Rebuild the inputs

From the repo root of the main checkout (read-only; `./bin/mariadb-read` refuses writes):

```sql
SELECT t.agreement_uuid, t.target_type, t.era, t.deal_type, t.target, t.acquirer, t.dt,
       t.transaction_price_total, TO_BASE64(x.xml)
FROM (SELECT a.agreement_uuid, a.target_type, a.deal_type, a.target, a.acquirer,
             COALESCE(a.filing_date, a.announce_date) dt, a.transaction_price_total,
             CASE WHEN YEAR(COALESCE(a.filing_date,a.announce_date)) < 2008 THEN 'a'
                  WHEN YEAR(COALESCE(a.filing_date,a.announce_date)) < 2016 THEN 'b'
                  WHEN YEAR(COALESCE(a.filing_date,a.announce_date)) < 2021 THEN 'c' ELSE 'd' END era,
             ROW_NUMBER() OVER (PARTITION BY a.target_type, <same era expression>
                                ORDER BY MD5(CONCAT('m1pilot', a.agreement_uuid))) rn
      FROM agreements a
      JOIN xml x0 ON x0.agreement_uuid = a.agreement_uuid AND x0.latest = 1 AND x0.status = 'verified'
      WHERE COALESCE(a.status, 'verified') = 'verified' AND a.target_type IS NOT NULL) t
JOIN xml x ON x.agreement_uuid = t.agreement_uuid AND x.latest = 1 AND x.status = 'verified'
WHERE (t.target_type = 'public' AND t.rn <= 9)
   OR (t.target_type = 'private' AND t.rn <= IF(t.era IN ('a', 'd'), 4, 3));
```

Then `python3 prepare.py m1.tsv OUT` writes the text layers, reading views and stage 1 candidates. The sample depends on the corpus at query time, so it drifts as agreements are added.

M2 uses the same query with three changes: the seed is `'m2'` instead of `'m1pilot'`, the M1 agreements are excluded (`AND a.agreement_uuid NOT IN (...)`), and the final filter is `WHERE t.rn <= IF(t.era = 'd', 18, 19)` for both target types, giving 150 agreements. Then `python3 prepare_v2.py m2.tsv OUT` writes the same three views plus `manifest.json`, which assigns every third agreement (sorted by target type, era and uuid) to dev and the rest to test. `python3 prepare_v2.py --rebuild OUT [split]` re-runs stage 1 from the saved layers.

The M2 holdout uses the seed `'m2holdout'`, excludes the M1 and M2 agreements, and filters on `WHERE t.rn <= IF(t.era = 'a', 7, 6)` for both target types, giving 50 agreements. Then `python3 prepare_v3.py holdout.tsv OUT` writes the views with defined-terms exhibits admitted and the stage 1 fixes from section 7.5; every agreement is in the `holdout` split.

The definition-resolution re-check reuses that dump: `python3 prepare_v4.py holdout.tsv OUT` adds page-blob exhibits and definition resolution, and `python3 score_v2.py D ARM GOLD --split holdout --concepts ma.termination.fee.target,ma.termination.fee.reverse,ma.indemn.cap,ma.mae.definition` scores the four concepts.

## Files

- `text_layer.py`: the canonical text layer (body `<text>` elements, unstripped, one per line).
- `prepare.py`: stage 1 retrieval and the reading views given to extractors and gold labelers.
- `schema_m1.md`: the extraction schema as run. Schema v2 changes are in section 7.4 of the plan.
- `score.py`: quote validation, span anchoring, detection, span and field metrics.
- `report.py`: final numbers against adjudicated gold; output in `m1_results.json`.
- `prepare_v2.py`: M2 stage 1 for four concept groups, with long definitions sections trimmed to relevant terms.
- `schema_v2.md`: the ten-concept schema run on the dev split. `schema_v2_1.md`: the same with 14 clarifications from dev, run on the test split and used for all adjudication.
- `score_v2.py`: M2 detection, span and key-field metrics (`python3 score_v2.py D ARM GOLD [--split dev|test] [--disagreements OUT]`).
- `report_v2.py`: final M2 numbers, stage 1 recall, verdicts and fee percentages (`python3 report_v2.py D [SPLIT ...]`, default dev and test); output in `m2_results.json`.
- `text_layer.project(exhibits=True)`: also admits defined-terms exhibits from back matter (continuation lines keep their term) and splits section-less body text into synthetic sections.
- `prepare_v3.py`: holdout stage 1 (funds reps, indemnity-limit titles, bare-number titles, trimming only for definitions sections).
- `schema_v2_2.md`: v2.1 plus 13 clarifications, run on the holdout and used for its adjudication.
- `m2_holdout_results.json`: holdout numbers, as run and with the exhibit continuation fix.
- `prepare_v4.py`: definition resolution (adds the defining paragraphs of relevant terms that candidate sections use) on top of `prepare_v3.py`; uses `text_layer.project(page_blobs=True)` for exhibits stored one page per `<text>`.
- `schema_v2_3.md`: v2.2 plus 14 clarifications, run on the re-check and used for its adjudication.
- `m2_defres_results.json`: re-check numbers for both arms (v3 candidates and v4 definition resolution).

## How it ran

Extraction (Sonnet, plus a Haiku arm in M1), gold (Opus), adjudication (Opus) and the M2 equity-value lookups (Sonnet with web search) ran as subagents in a Claude Code session, with no API calls. The per-agreement JSON was scratch and was deleted after scoring.
