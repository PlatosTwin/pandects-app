# Span-taxonomy pilots (M1, M2, M2 holdout, M3 holdout, financing-retrieval round, v2.6 re-score, M5 holdout, structural retrieval, model-chosen stage 1, M6 holdout and re-score, M7 holdout, re-score, run-to-run variance and v2.10, M8 holdout, v2.11, M9 holdout)

Results and discussion: [section 7.4](../../span-taxonomy.md#74-m1-pilot-results-2026-10-21) (M1, 3 concepts, 50 agreements) and [section 7.5](../../span-taxonomy.md#75-m2-pilot-results-2026-10-21) (M2, 10 concepts, 150 agreements) and [section 7.6](../../span-taxonomy.md#76-m2-holdout-results-2026-10-21) (M2 holdout, 50 fresh agreements) and [section 7.7](../../span-taxonomy.md#77-definition-resolution-re-check-2026-10-21) (definition-resolution re-check on the same holdout) and [section 7.8](../../span-taxonomy.md#78-m3-holdout-2026-10-21) (M3 holdout, 50 fresh agreements) and [section 7.9](../../span-taxonomy.md#79-financing-retrieval-round-2026-10-21) (financing-retrieval round, 50 fresh agreements, four concepts) and [section 7.10](../../span-taxonomy.md#710-schema-v26-and-the-holdout-re-score-2026-10-22) (schema v2.6, the same holdout re-scored) and [section 7.11](../../span-taxonomy.md#711-m5-holdout-2026-10-22) (M5 holdout, 50 fresh agreements, three concepts) and [section 7.12](../../span-taxonomy.md#712-structural-financing-retrieval-and-schema-v28-2026-10-22) (structural financing retrieval and schema v2.8) and [section 7.13](../../span-taxonomy.md#713-model-chosen-stage-1-2026-10-23) (a Haiku section picker) and [section 7.14](../../span-taxonomy.md#714-m6-holdout-rules-plus-picker-schema-v28-2026-10-23) (M6 holdout, the rules plus the picker) and [section 7.15](../../span-taxonomy.md#715-two-fixes-from-m6-re-scored-2026-10-23) (two fixes, M6 re-scored) and [section 7.16](../../span-taxonomy.md#716-m7-holdout-all-ten-concepts-rules-plus-picker-schema-v28-2026-10-23) (M7 holdout, all ten concepts) and [section 7.17](../../span-taxonomy.md#717-fixes-from-m7-re-scored-2026-10-23) (fixes from M7, re-scored) and [section 7.18](../../span-taxonomy.md#718-run-to-run-variance-on-m7-2026-10-23) (a second extraction of M7, compared run to run) and [section 7.19](../../span-taxonomy.md#719-schema-v210-and-post-checks-for-the-shared-errors-2026-10-23) (schema v2.10 and post-checks for the errors both runs share) and [section 7.20](../../span-taxonomy.md#720-m8-holdout-blind-schema-v210-pooled-with-m7-2026-10-23) (M8 holdout, blind, pooled with M7) and [section 7.21](../../span-taxonomy.md#721-schema-v211-two-post-checks-and-m7-and-m8-re-scored-2026-10-24) (schema v2.11 and two post-checks, M7 and M8 re-scored) and [section 7.22](../../span-taxonomy.md#722-m9-holdout-blind-schema-v211-pooled-with-m7-and-m8-2026-10-24) (M9 holdout, blind, pooled with M7 and M8) of the plan.

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

The M3 holdout uses the seed `'m3holdout'`, excludes the M1, M2 and M2 holdout agreements (regenerate each list from its own seeded query, in that order), and filters on `WHERE t.rn <= IF(t.era = 'a', 7, 6)`, giving 50 agreements. `python3 prepare_v4.py m3.tsv OUT` writes the views; after extraction, `python3 postchecks.py OUT sonnet sonnet_pc` applies the post-checks and `python3 report_m3.py OUT` reports both arms against adjudicated gold. Run the post-check tests with `python3 -m unittest test_postchecks` from this directory.

The financing-retrieval round uses the seed `'m4holdout'`, excludes the M1, M2, M2 holdout and M3 holdout agreements (use the M3 manifest itself: the corpus has drifted, and its query now reproduces 49 of 50), and filters on `WHERE t.rn <= IF(t.era = 'a', 7, 6)`, giving 50 agreements. `python3 prepare_v5.py m4.tsv OUT` writes the views. After extraction and post-checks, `python3 report_m3.py OUT ma.financing.structure,ma.termination.fee.reverse,ma.indemn.cap,ma.indemn.basket` reports the four concepts, with a percentage-only comparison of amounts under `pct_first_sonnet`.

The v2.6 re-score reuses that run: `python3 postchecks.py OUT sonnet sonnet_pc` with the v2.6 check, the same check applied to adjudicated gold against the full text (5 amounts in 2 agreements), then the same `report_m3.py` command. `python3 prepare_v6.py --rebuild OUT CAND` re-runs stage 1 from the saved layers for the not-blind retrieval check.

The next holdout uses the seed `'m5holdout'` and filters on `WHERE t.rn <= IF(t.era = 'a', 7, 6)`, giving 50 agreements. It excludes the M3 and M4 manifests by uuid, and M1, M2 and the M2 holdout by re-running their seeded queries as nested subqueries (`./bin/mariadb-read` refuses `WITH`). The query is saved as `.claude/span-taxonomy-runs/m5holdout/sample.sql`, and `python3 prepare_v6.py m5.tsv OUT` built the views there.
After extraction and post-checks, `python3 report_m3.py OUT ma.financing.structure,ma.indemn.cap,ma.indemn.basket` reports the three concepts. The gold, extraction and adjudication briefs given to the subagents are in `OUT/briefs/`.

The M7 holdout uses the seed `'m7holdout'` and adds the M6 manifest to the uuid exclusions (`.claude/span-taxonomy-runs/m7holdout/sample.sql`). `python3 prepare_v7.py m7.tsv OUT` builds the layers and v7 candidates; `python3 router.py outline OUT OUT/router/outline`, the picker brief in `OUT/router/brief.md`, then `python3 router.py union OUT OUT/router/picks OUT/candidates` build the stage 1 that was run. `python3 report_m3.py OUT` reports all ten concepts.

The M8 holdout uses the seed `'m8holdout'` and adds the M7 manifest to the uuid exclusions (`.claude/span-taxonomy-runs/m8holdout/sample.sql`), built the same way with schema v2.10. Three agreements under non-US law have split `holdout_non_us` in its manifest, so `score_v2.py D ARM GOLD --split holdout` scores the US agreements alone. `python3 pool_holdouts.py D1:ARM:GOLD[:SPLIT] D2:ARM:GOLD[:SPLIT]` pools holdouts against the bars. The v2.11 re-score reruns `postchecks.py` on M7 (`sonnet`, `sonnet_r2`) and M8 (`sonnet`), keeping the v2.10 outputs as `*_v210` and the M8 gold before v2.11 as `gold_pre_v211`, then pools `m7holdout:sonnet_r2_pc:gold m8holdout:sonnet_pc:gold:holdout`. Run all the tests with `python3 -m unittest test_postchecks test_prepare_v7 test_score_v2`.

The M9 holdout uses the seed `'m9holdout'` and adds the M8 manifest to the uuid exclusions (`.claude/span-taxonomy-runs/m9holdout/sample.sql`), built the same way with schema v2.11. Five agreements under non-US law have split `holdout_non_us`. It is pooled with `m7holdout:sonnet_r2_pc:gold m8holdout:sonnet_pc:gold:holdout m9holdout:sonnet_pc:gold:holdout`.

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
- `schema_v2_4.md`: v2.3 plus four rulings (prior bidder's fee refunds, indemnification outside the text, R&W-insurance-only deals, prices stated as components), run on the M3 holdout and used for its adjudication.
- `postchecks.py`: deterministic checks applied to extractor output (recommendation failure, cap equal to the price, expense-labeled fees); `test_postchecks.py` holds its tests. `postchecks.py OUT ARM_IN ARM_OUT` reads `OUT/read/` when present, so a cited section number resolves to its title even when stage 1 did not keep the section.
- `report_m3.py`, `m3_holdout_results.json`: M3 holdout numbers, as extracted and after the post-checks.
- `prepare_v5.py`: stage 1 with financing, deposit, trust-account and guarantee patterns and whole indemnification articles, on top of `prepare_v4.py`.
- `schema_v2_5.md`: v2.4 plus 12 rulings (committed financing, funds representations, reverse-fee triggers, escrow recourse, deductibles, outside-date extensions, MAE carve-outs), run on the financing-retrieval round and used for its adjudication.
- `m4_holdout_results.json`: financing-retrieval round numbers for the four concepts.
- `prepare_v6.py`: stage 1 with the financing wordings the financing-retrieval round missed, on top of `prepare_v5.py`.
- `schema_v2_6.md`: v2.5 plus 2 rulings (amounts hold only stated figures; three more funds-representation wordings).
- `schema_v2_7.md`: v2.6 plus 5 rulings (words against numerals, commitments named in a funds representation, section-scoped cumulative remedies, de minimis by indemnitor, reverse-fee kind), for the next holdout.
- `prepare_v7.py`: stage 1 that adds financing sections by structure (recitals, funds sections of any representations article, any section using "financing") and resolves cap-term definitions, on top of `prepare_v6.py`. `python3 prepare_v7.py --rebuild D OUT` re-runs it from saved layers.
- `schema_v2_8.md`: v2.7 plus 8 rulings (parent guarantees, rollovers, set-off recourse, five cap-coding rules), for the next holdout.
- `stage1_v7_check.json`: v6 against v7 stage 1 recall and candidate share on M3, M4 and M5.
- `router.py`: model-chosen stage 1. `outline D OUT` writes the outlines the picker reads, `candidates D PICKS OUT` builds candidate views from its picks, `score D PICKS` measures gold-span recall and candidate share against `prepare_v7.py`, and `union D PICKS OUT` builds the v7 sections plus the picks (the stage 1 of 7.14). The picker brief and picks are in the gitignored `.claude/span-taxonomy-runs/router/`.
- `router_check.json`: picker, v7 and union recall and candidate share on M3, M4 and M5.
- `m6_holdout_results.json`: the numbers in 7.14, with the stage 1 comparison and the scores before the v2.3 rule 2 post-check fix.
- `schema_v2_9.md`: v2.8 plus 11 rulings from the M7 adjudication.
- `m7_rescore_results.json`: the numbers in 7.17: M7 against v2.9 gold, with the old extraction scored too, the re-extracted agreements, and how many candidate files the definition fixes changed.
- `compare_runs.py`: two extraction runs compared with each other and with gold, field by field (`python3 compare_runs.py D ARM_A ARM_B GOLD [--only UUIDS.json]`).
- `m7_variance_results.json`: the numbers in 7.18: detection and field differences between the two M7 runs, each field's score in both runs against the 0.97 bar, and run 2 scored against gold.
- `schema_v2_10.md`: v2.9 plus 4 rulings from the M7 run-to-run comparison.
- `schema_v2_11.md`: v2.10 plus 10 rulings from the M8 adjudication.
- `m7_v210_results.json`: the numbers in 7.19: both M7 runs re-scored under v2.10 gold and the new post-checks, every fire of those checks on the five holdouts, and the two gold changes.
- `pool_holdouts.py`: detection, field and set-field counts summed across holdouts and checked against the bars, with an exact one-sided 95% lower bound per scalar field.
- `m8_holdout_results.json`: the numbers in 7.20: M8 alone (all 50 and the 47 US agreements) and pooled with M7, stage 1 recall, verdicts, the schema-ambiguous items and subagent token counts.
- `m9_holdout_results.json`: the numbers in 7.22: M9 alone (all 50 and the 45 US agreements, against raw and adjudicated gold), pooled with M7 and M8 (97 and 142 US agreements), stage 1 recall, verdicts, the schema-ambiguous items and subagent token counts.
- `m8_v211_results.json`: the numbers in 7.21: the M8 gold changes and the rulings behind them, every v2.11 post-check fire on M3 to M8, and M8 and the 97-agreement pool before and after.
- `m7_holdout_results.json`: the numbers in 7.16, all ten concepts, with the stage 1 comparison and subagent token counts.
- `m6_rescore_results.json`: the numbers in 7.15, with the `cash_on_hand` post-check's effect on M3 to M5.
- `test_prepare_v7.py`: tests for the definition forms resolution reads (`python3 -m unittest test_prepare_v7`).
- `m5_holdout_results.json`: M5 holdout numbers for financing, cap and basket.
- `m4_holdout_v26_results.json`: the financing-retrieval round re-scored under v2.6, with the gold corrections and the not-blind stage 1 check.

## How it ran

Extraction (Sonnet, plus a Haiku arm in M1), gold (Opus), adjudication (Opus) and the M2 equity-value lookups (Sonnet with web search) ran as subagents in a Claude Code session, with no API calls. The per-agreement JSON for M1 to the re-check was scratch and was deleted after scoring. The M3 holdout's inputs, gold, extractions and verdicts were kept in a gitignored folder of the worktree (`.claude/span-taxonomy-runs/m3holdout/` and `m4holdout/`), which lasts as long as the worktree does.
