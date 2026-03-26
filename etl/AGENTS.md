# ETL Subtree Instructions

## Sections Schema Notes
- In ETL, `sections.section_standard_id` and `sections.section_standard_id_gold_label` are often stored as serialized JSON lists of ids, not guaranteed scalar strings.
- Do not assume a predicate like `COALESCE(section_standard_id_gold_label, section_standard_id) = :id` is sufficient for membership checks.
- When filtering for a specific section id in SQL, handle both shapes:
  - direct scalar equality
  - membership inside the serialized JSON array representation
- When parsing these values in Python, reuse the existing helpers in `etl/src/etl/utils/latest_sections_search.py` instead of open-coding a new parser.
