# M1 and M2 pilots (span taxonomy)

Results and discussion: [section 7.4](../../span-taxonomy.md#74-m1-pilot-results-2026-10-21) (M1, 3 concepts, 50 agreements) and [section 7.5](../../span-taxonomy.md#75-m2-pilot-results-2026-10-21) (M2, 10 concepts, 150 agreements) of the plan.

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

## Files

- `text_layer.py`: the canonical text layer (body `<text>` elements, unstripped, one per line).
- `prepare.py`: stage 1 retrieval and the reading views given to extractors and gold labelers.
- `schema_m1.md`: the extraction schema as run. Schema v2 changes are in section 7.4 of the plan.
- `score.py`: quote validation, span anchoring, detection, span and field metrics.
- `report.py`: final numbers against adjudicated gold; output in `m1_results.json`.
- `prepare_v2.py`: M2 stage 1 for four concept groups, with long definitions sections trimmed to relevant terms.
- `schema_v2.md`: the ten-concept schema run on the dev split. `schema_v2_1.md`: the same with 14 clarifications from dev, run on the test split and used for all adjudication.
- `score_v2.py`: M2 detection, span and key-field metrics (`python3 score_v2.py D ARM GOLD [--split dev|test] [--disagreements OUT]`).
- `report_v2.py`: final M2 numbers for both splits, stage 1 recall, verdicts and fee percentages; output in `m2_results.json`.

## How it ran

Extraction (Sonnet, plus a Haiku arm in M1), gold (Opus), adjudication (Opus) and the M2 equity-value lookups (Sonnet with web search) ran as subagents in a Claude Code session, with no API calls. The per-agreement JSON was scratch and was deleted after scoring.
