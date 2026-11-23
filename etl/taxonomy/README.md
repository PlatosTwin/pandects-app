# Taxonomy packs

This folder is the source of truth for the Pandects provision taxonomy. The design and its reasons are in `docs/plans/full-taxonomy.md`. The packs are a draft (release `2026.1`). Nothing reads them in production yet.

## Layout

| Path | Contents |
|---|---|
| `ma/*.yaml` | The M&A pack. One file per navigation branch. |
| `tax/*.yaml` | The tax pack. One file per navigation branch. It mirrors the tax clause tree that a tax professor helped build. |
| `legacy/*.yaml` | Read-only snapshots of the ids in the three older trees. The validator uses them to check crosswalks. Do not edit them. |
| `crosswalk/dealos.yaml` | Links from concepts to DealOS knowledge graph ids. It is a separate file so that a public dump can leave it out. |

## Two kinds of node

A **navigation node** is a place in the browse tree. Its kind is `branch` (top level) or `group` (second level). Its id starts with `ma.nav.` or `tax.nav.`.

A **concept** is a thing that a label can point to. Its kind is one of:

| Kind | Meaning |
|---|---|
| `provision` | A provision that answers one research question, such as `ma.governing_law`. |
| `component` | A part of a provision that researchers study by itself, such as one negative covenant. It needs a `parent` concept. |
| `definition` | A defined term whose content is a research object, such as `ma.def.superior_proposal`. |
| `modifier` | A clause that changes how another concept applies, such as `ma.mae.disproportionate_effect`. It needs a `parent` concept. |

A concept's id never states where the concept sits in the tree. The tree can change and the id stays.

## Fields

| Field | Applies to | Meaning |
|---|---|---|
| `id` | all | Lowercase dotted id with the pack prefix. It never changes after release. |
| `label` | all | Display name. |
| `kind` | all | See above. |
| `nav_parent` | groups, concepts | The navigation node that holds this node. It must be in the same pack. |
| `parent` | concepts | The concept that this concept is a part of. Required for `component` and `modifier`. |
| `also_listed_under` | all | Navigation nodes, in any pack, where the node also appears. This is how a concept on the seam between the packs shows up in both trees with one owner. |
| `assignable` | navigation nodes | `true` when a label may point to the navigation node itself. Every tax navigation node is assignable, as the tax labeler assigns a level 1 or level 2 node today. An M&A group is assignable when it holds a wave 2 provision or definition. See "Waves and filter levels". No M&A branch is assignable. |
| `tier` | concepts | `value` when the concept has a piloted value schema. Otherwise `label`. |
| `wave` | M&A concepts | `1` or `2`: the labeling wave. See "Waves and filter levels". The tax pack has no waves. |
| `attributes.party` | concepts | The role of the party that a label instance must record: `maker` (of a representation), `obligor`, `tested_party`, `holder` (of a right), `indemnitor` or `subject`. The party is stored on the instance. There is no separate subtree per party. |
| `description` | all | One or two sentences. Written for this project. |
| `canonical_terms` | concepts | Words and defined terms that signal the concept. |
| `example_phrases` | concepts | Short illustrative wording. These are examples, not verified quotations. |
| `exclusion_cues` | concepts | What the concept is not, with the id of the right concept. |
| `value_schema` | concepts | `status: piloted` for the fields that the value pilot tested. `status: stub` for a first sketch of fields that nobody has tested. |
| `crosswalk` | all | Links to older ids. See below. |
| `new_in_redesign` | tax nodes | `true` for a node that the legacy tax tree does not have. The 7 such nodes were accepted on 2026-11-22. |
| `introduced_in` | all | The release that added the node. |
| `deprecated_in`, `replaced_by` | all | For a node that a later release retires. A retired node stays in the file. |

## Waves and filter levels

Labeling of the M&A pack runs in two waves (decided 2026-11-22). Each M&A concept states its wave.

| Wave | Contents | Rule |
|---|---|---|
| `wave: 1` | Section-level topics and the catch-alls | A provision that is the heading of a section of its own in at least 1% of agreements, measured with `docs/plans/full-taxonomy/prevalence.py`. Also the catch-all concepts. |
| `wave: 2` | Clause-level provisions, parts and definitions | A provision that sits inside a larger section, such as a termination trigger or the indemnity cap. Every `component`, `modifier` and `definition`. The catch-all for termination rights, because its siblings are in wave 2. |

A new M&A concept gets its wave by the same rule. The validator rejects an M&A concept with no wave, and a part or a definition in wave 1.

**Group labels in wave 1** (decided 2026-11-23). In wave 1, a section whose content is only wave 2 concepts gets its group as the label. A "Termination" section gets `ma.nav.termination.rights`. A group is therefore marked `assignable: true` when it holds a wave 2 provision or definition. That is 16 of the 54 groups. A part does not count, because its text sits inside the text of its parent concept, and the parent carries the label. The validator rejects a group that breaks this rule in either direction, and it rejects an assignable M&A branch. A group label does not replace a catch-all: content that fits no concept of any wave goes to the catch-all concept.

The tax pack has no waves. One tax pass labels it after wave 1 has found the tax sections.

The groups are the default level for browsing and filtering. A label on a concept counts for its group and its branch, so a filter on a group returns every section that carries one of the group's concepts. A filter on one concept is the finer choice.

## Crosswalks

`crosswalk` maps a scheme name to a list of `{id, relation, party?}`.

| Scheme | The older tree |
|---|---|
| `section_v1` | The live section tree (12 / 61 / 99). Ids are the hashes in `taxonomy_l1`, `taxonomy_l2` and `taxonomy_l3`. |
| `section_v2` | The 140-leaf redesign in `etl/src/etl/models/taxonomy/deepseek/taxonomy_enrichment.yaml`. |
| `tax_v1` | The tax clause tree (9 / 45 / 96), ids such as `tax.4.1.3`. |

`relation` describes the new node compared with the older node:

- `exact`: the same scope.
- `narrower`: the new node covers a part of the older node. The older node was split.
- `broader`: the new node covers more than the older node. Older nodes were merged.
- `related`: the scopes overlap.

`party` (`target` or `buyer`) records which party-specific older leaf the link came from. A new representation concept that replaces a target leaf and a buyer leaf links to both, each as `exact` with its party.

## Rules for changes

1. Never reuse or rename a released id. To retire a node, set `deprecated_in` and `replaced_by`.
2. A new node takes the next release number in `introduced_in`.
3. A concept on the seam between the packs has one owner. The other pack lists it with `also_listed_under`.
4. Keep the tax pack in step with the legacy tax tree. Mark an addition `new_in_redesign: true`, give the reason in the design doc, and get the maintainer's approval.
5. Run the validator and the test before a commit.

## Validate

From the repo root:

```
etl/.venv/bin/python -m etl.domain.taxonomy_pack
```

From `etl/`:

```
.venv/bin/python -m unittest tests.test_taxonomy_pack -v
```

Both read YAML files only. They need no network and no database.

The validator checks that:

- ids are unique, well formed and carry their pack prefix;
- each parent exists and has the right kind, in the right pack;
- there is no cycle in `nav_parent` or in `parent`;
- each crosswalk id exists in its legacy snapshot;
- each leaf of the three older trees maps to at least one node;
- each node of the legacy tax tree has exactly one `exact` match;
- a concept is `tier: value` only when its value schema is piloted;
- each M&A concept has `wave: 1` or `wave: 2`, and a part or a definition is in wave 2;
- an M&A group is `assignable` exactly when it holds a wave 2 provision or definition, and no M&A branch is assignable;
- each id that an exclusion cue names exists.
