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

# Leaves folded into a sibling on 2026-11-22 (docs/plans/full-taxonomy.md, section 4.4).
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
    "ma.spac.extension",
    "ma.spac.domestication",
    "ma.employee.service_credit",
    "ma.employee.no_third_party_rights",
    "ma.employee.plan_termination",
    "ma.condition.covenant_performance",
    "ma.condition.frustration",
    "ma.termination.notice",
    "ma.indemn.third_party_claims",
)

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

    def test_tax_navigation_is_assignable_and_ma_navigation_is_not(self) -> None:
        for node in self.taxonomy.nodes.values():
            if node.is_nav:
                self.assertEqual(bool(node.data.get("assignable")), node.pack == "tax", node.id)

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
        """The 20 leaves merged on 2026-11-22 were never released, so they are removed, not deprecated."""
        for concept_id in MERGED_2026_11_22:
            self.assertNotIn(concept_id, self.taxonomy.nodes)
        for node in self.taxonomy.nodes.values():
            self.assertNotIn("deprecated_in", node.data, node.id)


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

    def _errors(self) -> list[str]:
        return taxonomy_pack.validate(taxonomy_pack.load(self.root))

    def _assert_reported(self, fragment: str) -> None:
        errors = self._errors()
        self.assertTrue(any(fragment in error for error in errors), f"{fragment!r} not in {errors[:5]}")

    def test_untouched_copy_is_valid(self) -> None:
        self._edit(lambda doc: None)
        self.assertEqual(self._errors(), [])

    def test_duplicate_id(self) -> None:
        self._edit(lambda doc: doc["concepts"].append(dict(_concept(doc, "ma.governing_law"))))
        self._assert_reported("duplicate id ma.governing_law")

    def test_duplicate_id_across_packs_files(self) -> None:
        def change(doc: dict[str, Any]) -> None:
            clone = dict(doc["concepts"][0])
            clone["id"] = "ma.governing_law"
            doc["concepts"].append(clone)

        self._edit(change, "ma/general.yaml")
        self._assert_reported("duplicate id ma.governing_law")

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

    def test_exclusion_cue_with_unknown_id(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law")["exclusion_cues"].append("See ma.no_such_concept."))
        self._assert_reported("exclusion cue names unknown id ma.no_such_concept")

    def test_unknown_dealos_concept(self) -> None:
        def change(doc: dict[str, Any]) -> None:
            doc["links"].append({"concept": "ma.no_such_concept", "dealos_id": "cra:1", "relation": "exact"})

        self._edit(change, "crosswalk/dealos.yaml")
        self._assert_reported("crosswalk/dealos.yaml: unknown concept 'ma.no_such_concept'")

    def test_cli_reports_failure(self) -> None:
        self._edit(lambda doc: _concept(doc, "ma.governing_law").update(label=""))
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = taxonomy_pack.main([str(self.root)])
        self.assertEqual(code, 1)
        self.assertIn("ma.governing_law: label is empty", err.getvalue())


if __name__ == "__main__":
    unittest.main()
