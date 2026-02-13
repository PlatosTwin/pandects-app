---
id: request-patterns
title: Request Patterns
description: Practical patterns for search and retrieval flows.
sidebar_position: 2
---

# Request Patterns

## Search by year and clause type

```bash
curl "https://api.pandects.org/v1/search?year=2023&standard_id=1.1&page=1&page_size=25"
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

## Taxonomy bootstrap

```bash
curl "https://api.pandects.org/v1/taxonomy"
```

Use taxonomy IDs to align your downstream classification and analytics pipeline.
