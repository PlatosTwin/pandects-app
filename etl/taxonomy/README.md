# Taxonomy packs

This folder is the source of truth for the Pandects provision taxonomy. The design and its reasons are in `docs/plans/full-taxonomy.md`. The packs are a draft (release `2026.1`). Nothing reads them in production yet.

## Layout

| Path | Contents |
|---|---|
| `ma/*.yaml` | The M&A pack. One file per navigation branch. |
| `tax/*.yaml` | The tax pack. One file per navigation branch. It mirrors the tax clause tree that a tax professor helped build. |
| `legacy/*.yaml` | Read-only snapshots of the ids in the three older trees. The validator uses them to check crosswalks. Do not edit them. |

## Two kinds of node

A **navigation node** is a place in the browse tree. Its kind is `branch` (top level) or `group` (second level). Its id starts with `ma.nav.` or `tax.nav.`. A concept id never starts with `ma.nav.` or `tax.nav.`.

Each pack file has the keys `pack`, `release`, `branch`, `groups` and `concepts`. The `branch` holds one branch, `groups` holds groups only, and `concepts` holds concepts only.

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
| `order` | branches | The position of the branch in the browse tree. |
| `nav_parent` | groups, concepts | The navigation node that holds this node. It must be in the same pack. Every M&A concept sits under a group. Some tax concepts sit directly under a branch. |
| `parent` | concepts | The concept that this concept is a part of, in the same pack. Required for `component` and `modifier`. |
| `also_listed_under` | all | A list of navigation nodes, in any pack, where the node also appears. This is how a concept on the seam between the packs shows up in both trees with one owner. It never repeats the node's own `nav_parent`. |
| `assignable` | navigation nodes | `true` when a label may point to the navigation node itself. Every tax navigation node is assignable, as the tax labeler assigns a level 1 or level 2 node today. An M&A group is assignable when it holds a wave 2 provision or definition that is not derived. See "Waves and filter levels". No M&A branch is assignable. |
| `tier` | concepts | `value` when the concept has a piloted value schema. Otherwise `label`. |
| `wave` | M&A concepts | `1` or `2`: the labeling wave. See "Waves and filter levels". The tax pack has no waves, so a tax concept has no `wave`. |
| `derived` | concepts | `true` for a concept that labels no spans of its own. Its value is built from the spans of other concepts. See "Waves and filter levels". Leave the field out for every other concept. |
| `attributes.party` | concepts | The role of the party that a label instance must record: `maker` (of a representation), `obligor`, `tested_party`, `holder` (of a right), `indemnitor` or `subject`. The party is stored on the instance. There is no separate subtree per party. `party` is the only key of `attributes`. |
| `description` | all | One or two sentences, as one string. Written for this project. |
| `canonical_terms` | concepts | Words and defined terms that signal the concept. |
| `example_phrases` | concepts | Short illustrative wording. These are examples, not verified quotations. |
| `exclusion_cues` | concepts | What the concept is not, with the id of the right concept. A cue of a wave 1 concept also carries the wave notes. See "Wave notes in cues". |
| `value_schema` | concepts | `status: piloted` for the fields that the value pilot tested. `status: stub` for a first sketch of fields that nobody has tested. The keys are `status`, `fields`, and optionally `version` and `source` (quoted strings). Each entry of `fields` is a name, or a mapping with `name` and an optional `type`. |
| `crosswalk` | all | Links to older ids. See below. |
| `new_in_redesign` | tax nodes | `true` for a node that the legacy tax tree does not have. There are 10 such nodes: 7 were accepted on 2026-11-22 and 3 on 2026-11-23. |
| `introduced_in` | all | The release that added the node, as a quoted string such as `'2026.1'`. Without quotes, YAML reads `2026.10` as the number `2026.1`. |
| `deprecated_in`, `replaced_by` | all | For a node that a later release retires. `deprecated_in` is a quoted release string. `replaced_by` is a list of ids. A retired node stays in the file. |

The file-level `release` is a quoted string too. The validator rejects a field that is not in this table, or a field on a kind of node that the table does not name. For example, it rejects `exclusion_cue` (a misspelling of `exclusion_cues`) and `assignable` on a concept. It also rejects a key that a file states twice.

## Waves and filter levels

Labeling of the M&A pack runs in two waves (decided 2026-11-22). Each M&A concept states its wave.

| Wave | Contents | Rule |
|---|---|---|
| `wave: 1` | Section-level topics and the catch-alls | A provision that is the heading of a section of its own in at least 1% of agreements, measured with `docs/plans/full-taxonomy/prevalence.py`. Also the catch-all concepts, and `ma.other.lead_in` for the opening words of an article, which have no heading. |
| `wave: 2` | Clause-level provisions, parts and definitions | A provision that sits inside a larger section, such as a termination trigger or the indemnity cap. Every `component`, `modifier` and `definition`. The catch-all for termination rights, because its siblings are in wave 2. |

A new M&A concept gets its wave by the same rule. The validator rejects an M&A concept with no wave, a part or a definition in wave 1, and a tax concept with a wave.

**Group labels in wave 1** (decided 2026-11-23). In wave 1, a section whose content is only wave 2 concepts gets its group as the label. A "Termination" section gets `ma.nav.termination.rights`. A group is therefore marked `assignable: true` when it holds a wave 2 provision or definition. That is 14 of the 54 groups. A part does not count, because its text sits inside the text of its parent concept, and the parent carries the label. The validator rejects a group that breaks this rule in either direction, and it rejects an assignable M&A branch. A group label does not replace a catch-all: content that fits no concept of any wave goes to the catch-all concept.

**Wave notes in cues** (2026-11-23, after the 50-agreement labeling trial). A wave 1 reader can give only the wave 1 concepts and the assignable groups. In the trial, a cue that named a wave 2 concept sent readers to a label they did not have, and the text ended in a catch-all. Two rules fix this, and the validator checks both for every wave 1 concept:

- A cue that names a wave 2 concept also names the wave 1 label of that concept and says "wave 1". The wave 1 label is the parent for a part, and the group for any other wave 2 concept. Example: "The ceiling on recovery is ma.indemn.cap (in wave 1, label its group ma.nav.indemnification.limits)."
- A wave 1 concept that has wave 2 parts names each part in a cue that says the part keeps the concept's label in wave 1. Example, on `ma.approval.stockholder_meeting`: "In wave 1 this label also covers its part ma.approval.force_the_vote, which wave 2 marks inside its text".

`taxonomy_pack.wave_one_label(taxonomy, concept_id)` returns the wave 1 label of any M&A concept, and `taxonomy_pack.wave_two_parts(taxonomy)` lists the parts of each wave 1 concept. Code that builds reader cards should use them.

**Derived concepts.** A concept with `derived: true` labels no spans of its own. Its value is built from the spans of other concepts. `ma.financing.structure` is one: it combines the financing representation, covenant and conditions. A derived concept still states its wave, but it does not make its group assignable, because no section gets it as a label. The validator summary counts the derived concepts of each pack.

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

`party` (`target` or `buyer`) records which party-specific older leaf the link came from. A new representation concept that replaces a target leaf and a buyer leaf links to both, each as `exact` with its party. An entry has no other keys.

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

- each file parses as YAML, holds a mapping and states no key twice;
- each node, file, value schema and crosswalk entry has only the documented fields;
- ids are unique across both packs, well formed and carry their pack prefix, and only navigation ids start with `ma.nav.` or `tax.nav.`;
- the ids in the legacy snapshots are present and unique;
- `release`, `introduced_in` and `deprecated_in` are quoted strings such as `'2026.1'`, and `description` is a non-empty string;
- each parent exists and has the right kind, in the right pack;
- each M&A concept sits under a group, not directly under a branch;
- `also_listed_under` names navigation nodes and does not repeat the node's own `nav_parent`;
- there is no cycle in `nav_parent` or in `parent`;
- each crosswalk id exists in its legacy snapshot;
- each leaf of the three older trees maps to at least one node;
- each node of the legacy tax tree has exactly one `exact` match;
- a concept is `tier: value` only when its value schema is piloted;
- each M&A concept has `wave: 1` or `wave: 2`, a part or a definition is in wave 2, and no tax concept has a wave;
- `derived` is `true` or `false`, and only a concept has it;
- an M&A group is `assignable` exactly when it holds a wave 2 provision or definition that is not derived, and no M&A branch is assignable;
- a cue of a wave 1 M&A concept that names a wave 2 concept also names its wave 1 label, and a wave 1 concept names each of its wave 2 parts in a cue that says "wave 1";
- every tax navigation node is `assignable`;
- `attributes.party` is one of the six party roles;
- each id that an exclusion cue names exists.
