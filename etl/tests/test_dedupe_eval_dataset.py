"""Tests for the dedupe-eval labeled dataset artifact and its loaders."""

import json
import tempfile
import unittest
from pathlib import Path

from etl.utils.dedupe_eval_dataset import (
    CORPUS_MANIFEST_PATH,
    LABELS_PATH,
    load_corpus_manifest,
    load_labels,
    manifest_drift,
    missing_from_manifest,
    pair_identity,
    sha256_text,
)

_KUSHCO_PAIR_ID = pair_identity(
    "cc3f3711-811e-5bbe-ba57-f81ec0faff8b", "8e3d96e1-9084-5ad6-b544-beca55a48c45"
)
_KCS_PAIR_ID = pair_identity(
    "3652de8d-a05b-541d-86fc-d4f477004555", "acd2c7b6-f85d-5b28-9569-fca073568f33"
)


def _side(uuid: str, url: str = "https://www.sec.gov/x") -> dict[str, object]:
    return {
        "agreement_uuid": uuid,
        "url": url,
        "filing_date": "2021-03-01",
        "target": "Target Co",
        "acquirer": "Acquirer Co",
    }


def _entry(loser: str, survivor: str, label: str = "duplicate") -> dict[str, object]:
    return {
        "pair_id": pair_identity(loser, survivor),
        "label": label,
        "batch": "A",
        "adjudication": {"method": "document audit", "date": "2026-08-05", "evidence": None},
        "loser": _side(loser),
        "survivor": _side(survivor),
    }


def _write_labels(pairs: list[dict[str, object]], counts: dict[str, int]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    )
    with handle as fh:
        json.dump({"dataset": "test", "counts": counts, "pairs": pairs}, fh)
    return Path(handle.name)


class TestPairIdentity(unittest.TestCase):
    def test_orientation_insensitive(self) -> None:
        self.assertEqual(pair_identity("b", "a"), pair_identity("a", "b"))
        self.assertEqual(pair_identity("a", "b"), "a__b")


class TestLoadLabelsValidation(unittest.TestCase):
    def test_flipped_duplicate_pair_rejected(self) -> None:
        pairs = [_entry("uuid-a", "uuid-b"), _entry("uuid-b", "uuid-a")]
        path = _write_labels(pairs, {"positive": 2, "negative": 0})
        with self.assertRaisesRegex(ValueError, "Duplicate pair"):
            _ = load_labels(path)

    def test_pair_id_must_match_sides(self) -> None:
        entry = _entry("uuid-a", "uuid-b")
        entry["pair_id"] = "uuid-a__uuid-c"
        path = _write_labels([entry], {"positive": 1, "negative": 0})
        with self.assertRaisesRegex(ValueError, "does not match its sides"):
            _ = load_labels(path)

    def test_count_header_drift_rejected(self) -> None:
        path = _write_labels(
            [_entry("uuid-a", "uuid-b")], {"positive": 2, "negative": 0}
        )
        with self.assertRaisesRegex(ValueError, "counts drifted"):
            _ = load_labels(path)

    def test_unknown_label_rejected(self) -> None:
        path = _write_labels(
            [_entry("uuid-a", "uuid-b", label="maybe")],
            {"positive": 0, "negative": 1},
        )
        with self.assertRaisesRegex(ValueError, "Unknown label"):
            _ = load_labels(path)

    def test_identical_sides_rejected(self) -> None:
        entry = _entry("uuid-a", "uuid-a")
        path = _write_labels([entry], {"positive": 1, "negative": 0})
        with self.assertRaisesRegex(ValueError, "identical sides"):
            _ = load_labels(path)

    def test_valid_file_parses(self) -> None:
        path = _write_labels(
            [
                _entry("uuid-a", "uuid-b"),
                _entry("uuid-c", "uuid-d", label="not_duplicate"),
            ],
            {"positive": 1, "negative": 1},
        )
        pairs = load_labels(path)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(pairs[0].is_dupe)
        self.assertFalse(pairs[1].is_dupe)
        self.assertEqual(pairs[0].loser.agreement_uuid, "uuid-a")
        self.assertEqual(pairs[0].survivor.agreement_uuid, "uuid-b")


class TestManifestDrift(unittest.TestCase):
    def test_clean_corpus_has_no_drift(self) -> None:
        manifest = {"u1": "aa", "u2": "bb"}
        self.assertEqual(manifest_drift(manifest, {"u1": "aa", "u2": "bb"}), [])

    def test_missing_document_reported(self) -> None:
        drift = manifest_drift({"u1": "aa", "u2": "bb"}, {"u1": "aa"})
        self.assertEqual(drift, ["missing document u2"])

    def test_sha_mismatch_reported(self) -> None:
        drift = manifest_drift({"u1": "aa"}, {"u1": "cc"})
        self.assertEqual(len(drift), 1)
        self.assertIn("sha256 mismatch for u1", drift[0])

    def test_extra_documents_are_not_drift(self) -> None:
        self.assertEqual(manifest_drift({"u1": "aa"}, {"u1": "aa", "u9": "zz"}), [])

    def test_missing_from_manifest(self) -> None:
        path = _write_labels(
            [_entry("uuid-a", "uuid-b")], {"positive": 1, "negative": 0}
        )
        pairs = load_labels(path)
        self.assertEqual(missing_from_manifest(pairs, {"uuid-a": "aa"}), ["uuid-b"])
        self.assertEqual(
            missing_from_manifest(pairs, {"uuid-a": "aa", "uuid-b": "bb"}), []
        )


class TestSha256Text(unittest.TestCase):
    def test_known_digest(self) -> None:
        self.assertEqual(
            sha256_text("merger agreement"),
            "d87ed54a2cd0725bf8fa80b085dcacaa9eed9d7415e8e645565259b1f4feffdb",
        )


class TestCommittedDataset(unittest.TestCase):
    """Invariants of the committed labels.json / corpus_manifest.json."""

    def test_counts(self) -> None:
        pairs = load_labels()
        positives = [pair for pair in pairs if pair.is_dupe]
        negatives = [pair for pair in pairs if not pair.is_dupe]
        self.assertEqual(len(positives), 187)
        self.assertEqual(len(negatives), 28)

    def test_batch_breakdown(self) -> None:
        pairs = load_labels()
        counts: dict[tuple[str, bool], int] = {}
        for pair in pairs:
            key = (pair.batch, pair.is_dupe)
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(counts[("A", True)], 171)
        self.assertEqual(counts[("A", False)], 4)
        self.assertEqual(counts[("B", True)], 15)
        self.assertEqual(counts[("B", False)], 13)
        self.assertEqual(counts[("C", True)], 1)
        self.assertEqual(counts[("D", False)], 11)

    def test_kushco_override_is_a_single_positive(self) -> None:
        pairs = [pair for pair in load_labels() if pair.pair_id == _KUSHCO_PAIR_ID]
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0].is_dupe)
        self.assertEqual(pairs[0].batch, "C")

    def test_kcs_pair_is_a_single_negative(self) -> None:
        pairs = [pair for pair in load_labels() if pair.pair_id == _KCS_PAIR_ID]
        self.assertEqual(len(pairs), 1)
        self.assertFalse(pairs[0].is_dupe)
        self.assertEqual(pairs[0].batch, "A")

    def test_every_labeled_document_is_in_the_corpus_manifest(self) -> None:
        pairs = load_labels()
        manifest = load_corpus_manifest()
        self.assertEqual(missing_from_manifest(pairs, manifest), [])

    def test_manifest_has_no_unreferenced_documents(self) -> None:
        pairs = load_labels()
        referenced = {
            uuid
            for pair in pairs
            for uuid in (pair.loser.agreement_uuid, pair.survivor.agreement_uuid)
        }
        manifest = load_corpus_manifest()
        self.assertEqual(sorted(set(manifest) - referenced), [])

    def test_artifact_files_are_committed(self) -> None:
        self.assertTrue(LABELS_PATH.exists())
        self.assertTrue(CORPUS_MANIFEST_PATH.exists())


if __name__ == "__main__":
    _ = unittest.main()
