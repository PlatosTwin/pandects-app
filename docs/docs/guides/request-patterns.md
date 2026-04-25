---
id: request-patterns
title: Request Patterns
description: Practical patterns for search and retrieval flows.
sidebar_position: 2
---

# Request Patterns

## Search by year and clause type

```bash
curl "https://api.pandects.org/v1/sections?year=2023&standard_id=1.1&page=1&page_size=25"
```

## Drill into an agreement

Use an agreement UUID from search results:

```bash
curl "https://api.pandects.org/v1/agreements/{agreement_uuid}"
```

## Fetch a specific section

```bash
curl "https://api.pandects.org/v1/sections/{section_uuid}"
```

## Track data provenance across requests

Every response carries the dump hash in the `X-Pandects-Dump-Hash` header and in the `dump_version` field of the response body. Log it alongside your query results to pin each data point to a specific snapshot.

```bash
# Capture the hash alongside results
curl -si "https://api.pandects.org/v1/sections?standard_id=1.1&page=1" \
  | grep -E "X-Pandects-Dump-Hash|dump_version"
```

To omit `dump_version` from the JSON body (e.g., for strict schema parsers):

```bash
curl "https://api.pandects.org/v1/sections?standard_id=1.1&include_dump=false"
```

## Taxonomy bootstrap

```bash
curl "https://api.pandects.org/v1/taxonomy"
```

Use taxonomy IDs to align your downstream classification and analytics pipeline.
