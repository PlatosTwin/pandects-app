---
id: request-patterns
title: Request Patterns
description: Practical patterns for search and retrieval flows.
sidebar_position: 2
---

# Request Patterns

## Search by year and transaction type

```bash
curl "https://api.pandects.org/v1/search?year=2023&transactionType=Merger&page=1&pageSize=25"
```

## Drill into an agreement

Use an agreement UUID from search results:

```bash
curl "https://api.pandects.org/v1/agreements/{agreement_uuid}?neighborSections=1"
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
