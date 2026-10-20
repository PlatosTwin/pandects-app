import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import requests

from etl.models.page_classifier_revamp import typesafe_judgments as tj
from etl.models.page_classifier_revamp.page_types import AgreementDocument, FeatureDict
from etl.models.page_classifier_revamp.inference import _coerce_typesafe_features


def _judgment(main_sig: float = 0.1, attachment_start: float = 0.1) -> tj.PageJudgment:
    return {
        "part_probs": {"front_matter": 0.1, "toc": 0.1, "body": 0.6, "sig": 0.1, "back_matter": 0.1},
        "main_sig": main_sig,
        "attachment_start": attachment_start,
    }


def _response(status_code: int, payload: dict[str, object] | None = None) -> mock.Mock:
    response = mock.Mock()
    response.status_code = status_code
    response.json.return_value = payload or {}
    response.text = "error body"
    return response


_OK_PAYLOAD: dict[str, object] = {
    "answers": {
        "part": {"type": "choice", "probabilities": {"body": 0.9, "sig": 0.1}},
        "main_sig": {"type": "noul", "noul": 0.2},
        "attachment_start": {"type": "noul", "noul": 0.05},
    }
}


class BuildPageStateTests(unittest.TestCase):
    def test_first_middle_and_last_pages(self) -> None:
        texts = ["cover page", "x" * 20000, "end text"]

        first = tj.build_page_state(texts, 0)
        middle = tj.build_page_state(texts, 1)
        last = tj.build_page_state(texts, 2)

        self.assertEqual(first["position"], "page 1 of 3 (33% of the way through the document)")
        self.assertEqual(first["previous_page_end"], "(none: this is the first page)")
        self.assertEqual(len(middle["page"]), tj.TYPESAFE_MAX_PAGE_CHARS)
        self.assertEqual(middle["previous_page_end"], "cover page")
        self.assertEqual(last["next_page_start"], "(none: this is the last page)")
        self.assertEqual(len(last["previous_page_end"]), tj.TYPESAFE_NEIGHBOR_EXCERPT_CHARS)


class AppendTypesafeFeaturesTests(unittest.TestCase):
    def test_adds_page_and_neighbor_features(self) -> None:
        sequences: list[list[FeatureDict]] = [[{"relative_page": 0.3}, {"relative_page": 0.6}, {"relative_page": 1.0}]]
        judgments = [[_judgment(), _judgment(main_sig=0.9), _judgment(attachment_start=0.8)]]

        _ = tj.append_typesafe_features(sequences, judgments)

        pages = sequences[0]
        self.assertEqual(pages[0]["ts_p_body"], 0.6)
        self.assertEqual(pages[0]["ts_prev_main_sig"], 0.0)
        self.assertEqual(pages[0]["ts_next_att_start"], 0.1)
        self.assertEqual(pages[2]["ts_prev_main_sig"], 0.9)
        self.assertEqual(pages[2]["ts_next_att_start"], 0.0)
        self.assertEqual([page["ts_seen_main_sig"] for page in pages], [0.0, 1.0, 1.0])

    def test_rejects_page_count_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            _ = tj.append_typesafe_features([[{"relative_page": 1.0}]], [[_judgment(), _judgment()]])


DOCUMENT = AgreementDocument("a1", ["page one", "page two"], [])


class FetchPageJudgmentsTests(unittest.TestCase):
    def test_fully_cached_pages_need_no_api_key(self) -> None:
        keys = [tj._cache_key(tj.build_page_state(DOCUMENT.page_texts, index)) for index in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "judgments.jsonl"
            tj.JudgmentCache(cache_path).put_many([(keys[0], _judgment(main_sig=0.3)), (keys[1], _judgment())])

            with mock.patch.dict(os.environ, {}, clear=True):
                judgments = tj.fetch_page_judgments([DOCUMENT], cache_path=cache_path)

        self.assertEqual([judgment["main_sig"] for judgment in judgments[0]], [0.3, 0.1])

    def test_missing_api_key_raises(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                _ = tj.fetch_page_judgments([DOCUMENT])

    def test_retries_overload_then_caches_answers(self) -> None:
        responses = [_response(529), _response(200, _OK_PAYLOAD), _response(200, _OK_PAYLOAD)]
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "judgments.jsonl"
            with mock.patch.object(requests.Session, "post", side_effect=responses) as post, mock.patch.object(
                time, "sleep"
            ):
                judgments = tj.fetch_page_judgments([DOCUMENT], cache_path=cache_path, api_key="k")
            cached_lines = cache_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(post.call_count, 3)
        self.assertEqual(judgments[0][0]["part_probs"]["body"], 0.9)
        self.assertEqual(judgments[0][0]["part_probs"]["toc"], 0.0)
        self.assertEqual(judgments[0][1]["main_sig"], 0.2)
        self.assertEqual(len(cached_lines), 2)

    def test_non_retryable_status_raises(self) -> None:
        with mock.patch.object(requests.Session, "post", return_value=_response(402)), mock.patch.object(
            time, "sleep"
        ):
            with self.assertRaises(tj.TypeSafeError):
                _ = tj.fetch_page_judgments([DOCUMENT], api_key="k")


class CoerceTypesafeFeaturesTests(unittest.TestCase):
    def test_absent_metadata_means_no_typesafe_features(self) -> None:
        self.assertIsNone(_coerce_typesafe_features(None))

    def test_matching_metadata_is_accepted(self) -> None:
        self.assertEqual(
            _coerce_typesafe_features(tj.typesafe_artifact_metadata()),
            tj.typesafe_artifact_metadata(),
        )

    def test_mismatched_prompt_version_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _ = _coerce_typesafe_features({"model": tj.TYPESAFE_MODEL, "prompt_version": "v1"})


if __name__ == "__main__":
    _ = unittest.main()
