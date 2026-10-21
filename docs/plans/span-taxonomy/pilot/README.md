# M1 pilot (span taxonomy)

Results and discussion: [section 7.4 of the plan](../../span-taxonomy.md#74-m1-pilot-results-2026-10-21).

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

## Files

- `text_layer.py`: the canonical text layer (body `<text>` elements, unstripped, one per line).
- `prepare.py`: stage 1 retrieval and the reading views given to extractors and gold labelers.
- `schema_m1.md`: the extraction schema as run. Schema v2 changes are in section 7.4 of the plan.
- `score.py`: quote validation, span anchoring, detection, span and field metrics.
- `report.py`: final numbers against adjudicated gold; output in `m1_results.json`.

## How it ran

Extraction (Sonnet and Haiku arms), gold (Opus) and adjudication (Opus) ran as subagents in a Claude Code session, with no API calls. The per-agreement JSON was scratch and was deleted after scoring.
