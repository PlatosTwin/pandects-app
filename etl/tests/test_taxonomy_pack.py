# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnusedCallResult=false, reportMissingModuleSource=false
"""Tests for the taxonomy packs under etl/taxonomy/ and their validator.

They read YAML files only. They need no network and no database.

Run from the etl folder:
  .venv/bin/python -m unittest tests.test_taxonomy_pack -v
"""

from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from etl.domain import taxonomy_pack

# Concept ids that the value pilot already uses (docs/plans/span-taxonomy.md). They must not change.
PILOT_VALUE_IDS = (
    "ma.termination.fee.target",
    "ma.termination.fee.reverse",
    "ma.termination.expense_reimbursement",
    "ma.termination.outside_date",
    "ma.mae.definition",
    "ma.mae.carveout",
    "ma.financing.structure",
    "ma.indemn.cap",
    "ma.indemn.basket",
    "ma.indemn.survival",
    "tax.election.338h10",
)
PILOT_SIBLING_IDS = (
    "ma.financing.cooperation_costs",
    "ma.termination.collection_costs",
    "ma.regulatory.filing_fee_allocation",
)

# Leaves folded into a neighbor on 2026-11-22 (docs/plans/full-taxonomy.md, section 4.4).
# An independent review on 2026-11-23 split five of the first 20 back out. See RESTORED_2026_11_23.
MERGED_2026_11_22 = (
    "ma.structure.effects_of_merger",
    "ma.structure.governing_documents",
    "ma.structure.directors_officers",
    "ma.exchange.fractional_shares",
    "ma.counterparts",
    "ma.remedies_cumulative",
    "ma.waiver",
    "ma.tender.rule_14d10",
    "ma.tender.board_designees",
    "ma.spac.trust_waiver",
    "ma.spac.business_activities",
    "ma.employee.service_credit",
    "ma.employee.plan_termination",
    "ma.termination.notice",
    "ma.indemn.third_party_claims",
)

# Restored id: (kind, wave). The wave follows the 1% heading rule. A component is always in wave 2.
RESTORED_2026_11_23 = {
    "ma.spac.extension": ("provision", 1),
    "ma.spac.domestication": ("provision", 1),
    "ma.condition.frustration": ("provision", 1),
    "ma.condition.covenant_performance": ("provision", 1),
    "ma.employee.no_third_party_rights": ("component", 2),
}

Change = Callable[[dict[str, Any]], None]


def _concept(doc: dict[str, Any], concept_id: str) -> dict[str, Any]:
    return next(row for row in doc["concepts"] if row["id"] == concept_id)


class ShippedPacksTest(unittest.TestCase):
    """Checks on the packs as committed."""

    taxonomy: taxonomy_pack.Taxonomy = taxonomy_pack.Taxonomy(nodes={}, legacy={}, dealos_links=[], load_errors=[])

    @classmethod
    def setUpClass(cls) -> None:
        cls.taxonomy = taxonomy_pack.load()

    def test_packs_are_valid(self) -> None:
        self.assertEqual(taxonomy_pack.validate(self.taxonomy), [])

    def test_cli_reports_success(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = taxonomy_pack.main([])
        self.assertEqual(code, 0)
        self.assertIn("OK", out.getvalue())

    def test_both_packs_have_branches_groups_and_concepts(self) -> None:
        counts = taxonomy_pack.summary(self.taxonomy)
        for pack in taxonomy_pack.PACKS:
            for key in ("branch", "group", "concepts"):
                self.assertGreater(counts[pack][key], 0, f"{pack} {key}")

    def test_pilot_ids_are_kept(self) -> None:
        for concept_id in PILOT_VALUE_IDS:
            node = self.taxonomy.nodes[concept_id]
            self.assertEqual(node.data["tier"], "value", concept_id)
            self.assertEqual(node.data["value_schema"]["status"], "piloted", concept_id)
        for concept_id in PILOT_SIBLING_IDS:
            self.assertTrue(self.taxonomy.nodes[concept_id].is_concept, concept_id)

    def test_only_pilot_concepts_are_value_tier(self) -> None:
        value_ids = {node.id for node in self.taxonomy.nodes.values() if node.data.get("tier") == "value"}
        self.assertEqual(value_ids, set(PILOT_VALUE_IDS))

    def test_ids_do_not_reuse_legacy_tax_numbers(self) -> None:
        """New ids are readable names. None of them is a legacy number such as tax.1.2.3."""
        self.assertEqual(set(self.taxonomy.nodes) & set(self.taxonomy.legacy["tax_v1"]), set())

    def test_tax_tree_mirrors_the_legacy_tree(self) -> None:
        """Each legacy tax node maps to one node with the same label. Nodes new in the redesign map to none."""
        legacy = self.taxonomy.legacy["tax_v1"]
        matched: set[str] = set()
        for node in self.taxonomy.nodes.values():
            if node.pack != "tax":
                continue
            entries = (node.data.get("crosswalk") or {}).get("tax_v1") or []
            exact = [entry["id"] for entry in entries if entry["relation"] == "exact"]
            if node.data.get("new_in_redesign"):
                self.assertEqual(exact, [], node.id)
                continue
            self.assertEqual(len(exact), 1, node.id)
            self.assertEqual(legacy[exact[0]]["label"], node.label, node.id)
            matched.add(exact[0])
        self.assertEqual(matched, set(legacy))

    def test_tax_navigation_is_assignable(self) -> None:
        for node in self.taxonomy.nodes.values():
            if node.is_nav and node.pack == "tax":
                self.assertIs(node.data.get("assignable"), True, node.id)

    def test_ma_groups_with_wave_two_content_are_assignable(self) -> None:
        """
        In wave 1, a section that holds only wave 2 concepts gets its group as the label
        (maintainer, 2026-11-23). No other M&A navigation node is assignable.
        """
        nodes = self.taxonomy.nodes
        assignable = {node.id for node in nodes.values() if node.pack == "ma" and node.data.get("assignable")}
        self.assertEqual(assignable, taxonomy_pack.groups_with_wave_two_content(self.taxonomy))
        self.assertEqual(len(assignable), 16)
        self.assertEqual({nodes[node_id].kind for node_id in assignable}, {"group"})
        self.assertIn("ma.nav.termination.rights", assignable)
        # Only wave 1 concepts, or wave 2 parts of wave 1 concepts: the section gets those concepts.
        self.assertNotIn("ma.nav.remedies.law_forum", assignable)
        self.assertNotIn("ma.nav.financing.covenants", assignable)

    def test_seam_concepts_have_one_owner(self) -> None:
        """A seam concept lives in one pack and is listed under a navigation node of the other pack."""
        nodes = self.taxonomy.nodes
        self.assertIn("ma.nav.reps.financial", nodes["tax.nav.reps"].data["also_listed_under"])
        self.assertIn("tax.nav.other", nodes["ma.employee.section_280g"].data["also_listed_under"])
        self.assertNotIn("ma.election.338h10", nodes)

    def test_every_ma_concept_has_a_wave(self) -> None:
        """Wave 1 holds provisions only. Parts and definitions wait for wave 2. The tax pack has no waves."""
        waves: dict[int, int] = {1: 0, 2: 0}
        for node in self.taxonomy.nodes.values():
            if not node.is_concept:
                continue
            if node.pack == "tax":
                self.assertNotIn("wave", node.data, node.id)
                continue
            wave = node.data["wave"]
            self.assertIn(wave, taxonomy_pack.WAVES, node.id)
            waves[wave] += 1
            if node.kind != "provision" or node.data.get("parent"):
                self.assertEqual(wave, 2, node.id)
        self.assertGreater(waves[1], 0)
        self.assertGreater(waves[2], 0)

    def test_merged_leaves_are_gone(self) -> None:
        """The 15 leaves that stay merged were never released, so they are removed, not deprecated."""
        self.assertEqual(len(MERGED_2026_11_22), 15)
        for concept_id in MERGED_2026_11_22:
            self.assertNotIn(concept_id, self.taxonomy.nodes)
        for node in self.taxonomy.nodes.values():
            self.assertNotIn("deprecated_in", node.data, node.id)

    def test_restored_leaves_are_back(self) -> None:
        """The review of 2026-11-23 split five merged leaves back out, each with its old id."""
        nodes = self.taxonomy.nodes
        for concept_id, (kind, wave) in RESTORED_2026_11_23.items():
            self.assertEqual((nodes[concept_id].kind, nodes[concept_id].data["wave"]), (kind, wave), concept_id)
        self.assertEqual(
            nodes["ma.employee.no_third_party_rights"].data["parent"], "ma.employee.comp_benefits_continuation"
        )
        self.assertEqual(nodes["ma.condition.covenant_performance"].data["value_schema"]["status"], "stub")
        self.assertIn("tax.nav.deal.restructuring", nodes["ma.spac.domestication"].data["also_listed_under"])


class ValidatorCatchesProblemsTest(unittest.TestCase):
    """Each test breaks a copy of the packs in one way and checks that the validator reports it."""

    root: Path = taxonomy_pack.DEFAULT_ROOT

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "taxonomy"
        shutil.copytree(taxonomy_pack.DEFAULT_ROOT, self.root)

    def _edit(self, change: Change, relative: str = "ma/remedies.yaml") -> None:
        path = self.root / relative
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        change(doc)
        path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def _edit_text(self, old: str, new: str, relative: str = "ma/remedies.yaml") -> None:
        """Change the raw text of a file, for problems that a YAML dump cannot write."""
        path = self.root / relative
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1), encoding="utf-8")

    def _errors(self) -> list[str]:
        return taxonomy_pack.validate(taxonomy_pack.load(self.root))

    def _assert_reported(self, *fragments: str) -> None:
        """Assert that one error holds every fragment."""
        errors = self._errors()
        self.assertTrue(
            any(all(fragment in error for fragment in fragments) for error in errors),
            f"{fragments!r} not in {errors[:5]}",
        )

    def test_untouched_copy_is_valid(self) -> None:
        self._edit(lambda doc: None)
        self.assertEqual(self._errors(), [])

    def test_duplicate_id(self) -> None:
        self._edit(lambda doc: doc["concepts"].append(dict(_concept(doc, "ma.governing_law"))))
        self._assert_reported("duplicate id ma.governing_law")

    def test_duplicate_id_across_files_of_one_pack(self) -> None:
        def change(doc: dict[str, Any]) -> None:
            clone = dict(doc["concepts"][0])
            clone["id"] = "ma.governing_law"
            doc["concepts"].append(clone)

        self._edit(change, "ma/general.yaml")
        # Files load in name order, so the copy in general.yaml comes first.
        self._assert_reported("ma/remedies.yaml: duplicate id ma.governing_law (also in ma/general.yaml)")

    def test_duplicate_id_across_packs(self) -> None:
        def change(doc: dict[str, Any]) -> None:
            clone = dict(doc["concepts"][0])
            clone["id"] = "ma.governing_law"
            doc["concepts"].append(clone)

        self._edit(change, "tax/deal.yaml")
        self._assert_reported("tax/deal.yaml: duplicate id ma.governing_law (also in ma/remedies.yaml)")

    def test_missing_nav_parent(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").update(nav_parent="ma.nav.no_such_group"))
        self._assert_reported("ma.governing_law: nav_parent 'ma.nav.no_such_group' does not exist")

    def test_nav_parent_in_another_pack(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").update(nav_parent="tax.nav.other"))
        self._assert_reported("ma.governing_law: nav_parent tax.nav.other is in another pack")

    def test_missing_concept_parent(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").update(parent="ma.no_such_concept"))
        self._assert_reported("ma.governing_law: parent 'ma.no_such_concept' does not exist")

    def test_navigation_cycle(self) -> None:
        def change(doc: dict[str, Any]) -> None:
            first, second = doc["groups"][0], doc["groups"][1]
            first["nav_parent"] = second["id"]
            second["nav_parent"] = first["id"]

        self._edit(change)
        self._assert_reported("cycle in nav_parent")

    def test_concept_parent_cycle(self) -> None:
        def change(doc: dict[str, Any]) -> None:
            _concept(doc, "ma.governing_law")["parent"] = "ma.forum_selection"
            _concept(doc, "ma.forum_selection")["parent"] = "ma.governing_law"

        self._edit(change)
        self._assert_reported("cycle in parent")

    def test_unknown_crosswalk_id(self) -> None:
        def change(doc: dict[str, Any]) -> None:
            entries = _concept(doc, "ma.governing_law")["crosswalk"]["section_v1"]
            entries.append({"id": "no_such_id", "relation": "exact"})

        self._edit(change)
        self._assert_reported("ma.governing_law: crosswalk id 'no_such_id' is not in section_v1")

    def test_unknown_crosswalk_relation(self) -> None:
        def change(doc: dict[str, Any]) -> None:
            _concept(doc, "ma.governing_law")["crosswalk"]["section_v1"][0]["relation"] = "same"

        self._edit(change)
        self._assert_reported("must be one of ('exact', 'narrower', 'broader', 'related')")

    def test_unmapped_legacy_leaf(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").pop("crosswalk"))
        self._assert_reported("has no crosswalk")

    def test_value_tier_without_piloted_schema(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").update(tier="value"))
        self._assert_reported("ma.governing_law: tier value needs a piloted value_schema")

    def test_missing_wave(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").pop("wave"))
        self._assert_reported("ma.governing_law: wave must be one of (1, 2)")

    def test_unknown_wave(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").update(wave=3))
        self._assert_reported("ma.governing_law: wave must be one of (1, 2)")

    def test_part_in_wave_one(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.mae.carveout").update(wave=1), "ma/definitions.yaml")
        self._assert_reported("ma.mae.carveout: a component belongs to wave 2")

    def test_wave_on_navigation_node(self) -> None:
        self._edit(lambda doc: doc["groups"][0].update(wave=1))
        self._assert_reported("a navigation node cannot have wave")

    def test_group_with_wave_two_content_not_assignable(self) -> None:
        self._edit(lambda doc: doc["groups"][0].pop("assignable"), "ma/termination.yaml")
        self._assert_reported("ma.nav.termination.rights: a group that holds a wave 2 provision or definition needs assignable")

    def test_assignable_group_without_wave_two_content(self) -> None:
        self._edit(lambda doc: doc["groups"][0].update(assignable=True))
        self._assert_reported("ma.nav.remedies.law_forum: assignable is for a group that holds a wave 2 provision")

    def test_assignable_ma_branch(self) -> None:
        self._edit(lambda doc: doc["branch"].update(assignable=True))
        self._assert_reported("ma.nav.remedies: an M&A branch is not assignable")

    def test_exclusion_cue_with_unknown_id(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law")["exclusion_cues"].append("See ma.no_such_concept."))
        self._assert_reported("exclusion cue names unknown id ma.no_such_concept")

    def test_unknown_dealos_concept(self) -> None:
        def change(doc: dict[str, Any]) -> None:
            doc["links"].append({"concept": "ma.no_such_concept", "dealos_id": "cra:1", "relation": "exact"})

        self._edit(change, "crosswalk/dealos.yaml")
        self._assert_reported("crosswalk/dealos.yaml: unknown concept 'ma.no_such_concept'")

    # Finding 1: a concept parent in the other pack.
    def test_concept_parent_in_another_pack(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").update(parent="tax.election.338h10", wave=2))
        self._assert_reported("ma.governing_law: parent tax.election.338h10 is in another pack")

    def test_concept_parent_that_is_not_an_id(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").update(parent=["ma.forum_selection"], wave=2))
        self._assert_reported("ma.governing_law: parent must be a concept id")

    # Finding 2: an M&A concept directly under a branch.
    def test_ma_concept_under_a_branch(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").update(nav_parent="ma.nav.remedies", wave=2))
        self._assert_reported("ma.governing_law: an M&A concept must sit under a group, not under branch ma.nav.remedies")
        wave_two = taxonomy_pack.groups_with_wave_two_content(taxonomy_pack.load(self.root))
        self.assertNotIn("ma.nav.remedies", wave_two)

    # Finding 3: a tax concept with a wave.
    def test_wave_on_tax_concept(self) -> None:
        self._edit(lambda doc: _concept(doc, "tax.election.338h10").update(wave=1), "tax/deal.yaml")
        self._assert_reported("tax.election.338h10: the tax pack has no waves")

    # Finding 4: a tax navigation node that is not assignable.
    def test_tax_navigation_node_not_assignable(self) -> None:
        def change(doc: dict[str, Any]) -> None:
            doc["branch"]["assignable"] = False
            doc["groups"][0].pop("assignable")

        self._edit(change, "tax/deal.yaml")
        self._assert_reported("tax.nav.deal: every tax navigation node needs assignable: true")
        self._assert_reported("tax.nav.deal.treatment: every tax navigation node needs assignable: true")

    # Finding 5: the DealOS crosswalk.
    def test_missing_dealos_file(self) -> None:
        (self.root / "crosswalk" / "dealos.yaml").unlink()
        self._assert_reported("crosswalk/dealos.yaml: missing")

    def test_duplicate_dealos_link(self) -> None:
        self._edit(lambda doc: doc["links"].append(dict(doc["links"][0])), "crosswalk/dealos.yaml")
        self._assert_reported("crosswalk/dealos.yaml: duplicate link")

    def test_dealos_link_with_bad_relation_and_bad_id(self) -> None:
        def change(doc: dict[str, Any]) -> None:
            doc["links"].append({"concept": "ma.governing_law", "dealos_id": "cra-39", "relation": "same"})

        self._edit(change, "crosswalk/dealos.yaml")
        self._assert_reported("crosswalk/dealos.yaml: relation for ma.governing_law must be one of")
        self._assert_reported("crosswalk/dealos.yaml: dealos_id 'cra-39' for ma.governing_law must look like")

    def test_dealos_id_form(self) -> None:
        for bad in ("cra-39", "cra:stub-termination", "dt:12", "cra:R-Foo", "cra:r-", "cra:39 ", 39):
            with self.subTest(dealos_id=bad):
                self.setUp()

                def change(doc: dict[str, Any], bad: Any = bad) -> None:
                    doc["links"].append({"concept": "ma.governing_law", "dealos_id": bad, "relation": "related"})

                self._edit(change, "crosswalk/dealos.yaml")
                self._assert_reported(f"crosswalk/dealos.yaml: dealos_id {bad!r} for ma.governing_law must look like")

    # Finding 6: the loader reports instead of crashing.
    def test_legacy_row_without_id(self) -> None:
        self._edit(lambda doc: doc["nodes"].append({"label": "No id", "leaf": False}), "legacy/section_v1.yaml")
        self._assert_reported("legacy/section_v1.yaml: a row has no string id")

    def test_duplicate_legacy_id(self) -> None:
        def change(doc: dict[str, Any]) -> None:
            doc["nodes"].append(dict(doc["nodes"][0], label="Copy"))

        self._edit(change, "legacy/section_v1.yaml")
        self._assert_reported("legacy/section_v1.yaml: duplicate id")

    def test_malformed_yaml(self) -> None:
        self._edit_text("concepts:\n", "concepts: [unclosed\n")
        self._assert_reported("ma/remedies.yaml", "cannot parse YAML")

    # Finding 7: duplicate YAML keys.
    def test_duplicate_yaml_key(self) -> None:
        self._edit_text("    label: Governing Law\n", "    label: Governing Law\n    label: Choice of Law\n")
        self._assert_reported("ma/remedies.yaml", "duplicate key 'label'")

    # Finding 8: unknown fields.
    def test_unknown_node_field(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").update(exclusion_cue=["Not forum."]))
        self._assert_reported("ma.governing_law: unknown field exclusion_cue")

    def test_field_of_another_kind(self) -> None:
        def change(doc: dict[str, Any]) -> None:
            doc["groups"][0]["order"] = 1
            _concept(doc, "ma.governing_law")["assignable"] = True

        self._edit(change)
        self._assert_reported("ma.nav.remedies.law_forum: a group cannot have order")
        self._assert_reported("ma.governing_law: a provision cannot have assignable")

    def test_unknown_file_field(self) -> None:
        self._edit(lambda doc: doc.update(notes="draft"))
        self._assert_reported("ma/remedies.yaml: unknown field notes")

    def test_unknown_value_schema_field(self) -> None:
        def change(doc: dict[str, Any]) -> None:
            schema = _concept(doc, "ma.termination.outside_date")["value_schema"]
            schema["statuss"] = "piloted"
            schema["fields"].append({"name": "notice_days", "kind": "int"})

        self._edit(change, "ma/termination.yaml")
        self._assert_reported("ma.termination.outside_date: value_schema has unknown field statuss")
        self._assert_reported("ma.termination.outside_date: value_schema.fields entry", "must be a name or a mapping")

    def test_unknown_crosswalk_entry_field(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law")["crosswalk"]["section_v1"][0].update(relaton="exact"))
        self._assert_reported("ma.governing_law: crosswalk entry", "has unknown field relaton")

    def test_crosswalk_scheme_not_a_list(self) -> None:
        def change(doc: dict[str, Any]) -> None:
            crosswalk = _concept(doc, "ma.governing_law")["crosswalk"]
            crosswalk["section_v1"] = crosswalk["section_v1"][0]

        self._edit(change)
        self._assert_reported("ma.governing_law: crosswalk section_v1 must be a list")

    def test_unknown_dealos_link_field(self) -> None:
        def change(doc: dict[str, Any]) -> None:
            doc["links"].append({"concept": "ma.governing_law", "dealos_id": "rw:9999", "relation": "related", "note": "x"})

        self._edit(change, "crosswalk/dealos.yaml")
        self._assert_reported("crosswalk/dealos.yaml: link for ma.governing_law has unknown field note")

    def test_row_kind_does_not_match_its_list(self) -> None:
        self._edit(lambda doc: doc["groups"].append(dict(_concept(doc, "ma.governing_law"), id="ma.governing_law_two")))
        self._assert_reported("ma.governing_law_two: a row under groups must have kind group")

    # Finding 9: releases must be strings.
    def test_unquoted_introduced_in(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").update(introduced_in=2026.10))
        self._assert_reported("ma.governing_law: introduced_in must be a quoted string")

    def test_unquoted_deprecated_in(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").update(deprecated_in=2026.2))
        self._assert_reported("ma.governing_law: deprecated_in must be a quoted string")

    def test_unquoted_release(self) -> None:
        self._edit(lambda doc: doc.update(release=2026.1))
        self._assert_reported("ma/remedies.yaml: release must be a quoted string")

    # Finding 10: smaller checks.
    def test_description_not_a_string(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").update(description=["One.", "Two."]))
        self._assert_reported("ma.governing_law: description must be a non-empty string")

    def test_also_listed_under_its_own_nav_parent(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").update(also_listed_under=["ma.nav.remedies.law_forum"]))
        self._assert_reported("ma.governing_law: also_listed_under repeats its nav_parent ma.nav.remedies.law_forum")

    def test_also_listed_under_not_a_list(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").update(also_listed_under="tax.nav.other"))
        self._assert_reported("ma.governing_law: also_listed_under must be a list of ids")

    def test_navigation_id_without_nav(self) -> None:
        def change(doc: dict[str, Any]) -> None:
            doc["groups"].append(
                {
                    "id": "ma.remedies_extra",
                    "label": "Extra",
                    "kind": "group",
                    "nav_parent": "ma.nav.remedies",
                    "description": "A group with a concept-style id.",
                    "introduced_in": "2026.1",
                }
            )

        self._edit(change)
        self._assert_reported("ma.remedies_extra: a navigation node id must start with ma.nav.")

    def test_concept_id_with_nav(self) -> None:
        self._edit(
            lambda doc: doc["concepts"].append(dict(_concept(doc, "ma.governing_law"), id="ma.nav.remedies.law_copy"))
        )
        self._assert_reported("ma.nav.remedies.law_copy: a concept id must not start with ma.nav.")

    # Finding 11: the party attribute.
    def test_unknown_party(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").update(attributes={"party": "seller"}))
        self._assert_reported("ma.governing_law: attributes.party must be one of")

    # Finding 12: derived concepts.
    def test_derived_on_navigation_node(self) -> None:
        self._edit(lambda doc: doc["groups"][0].update(derived=True))
        self._assert_reported("ma.nav.remedies.law_forum: a navigation node cannot have derived")

    def test_derived_not_boolean(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").update(derived="yes"))
        self._assert_reported("ma.governing_law: derived must be true or false")

    def test_derived_concept_does_not_make_a_group_assignable(self) -> None:
        def change(doc: dict[str, Any]) -> None:
            for row in doc["concepts"]:
                if row["nav_parent"] == "ma.nav.termination.rights" and row["wave"] == 2 and "parent" not in row:
                    row["derived"] = True

        self._edit(change, "ma/termination.yaml")
        wave_two = taxonomy_pack.groups_with_wave_two_content(taxonomy_pack.load(self.root))
        self.assertNotIn("ma.nav.termination.rights", wave_two)
        self._assert_reported("ma.nav.termination.rights: assignable is for a group that holds a wave 2 provision")

    def test_summary_counts_derived_concepts(self) -> None:
        before = taxonomy_pack.summary(taxonomy_pack.load(self.root))["ma"]["derived"]
        self._edit(lambda doc: _concept(doc, "ma.governing_law").update(derived=True))
        after = taxonomy_pack.summary(taxonomy_pack.load(self.root))["ma"]["derived"]
        self.assertEqual(after, before + 1)

    def test_cli_reports_failure(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").update(label=""))
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = taxonomy_pack.main([str(self.root)])
        self.assertEqual(code, 1)
        self.assertIn("ma.governing_law: label is empty", err.getvalue())


if __name__ == "__main__":
    unittest.main()
