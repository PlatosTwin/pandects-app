# pyright: reportPrivateUsage=false
import unittest

from etl.models.ner.ner import NERInference


def _offsets_for_tokens(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = text.find(token, cursor)
        if start < 0:
            raise ValueError(f"Token {token!r} not found in expected order.")
        end = start + len(token)
        offsets.append((start, end))
        cursor = end
    return offsets


class NerInferencePrettyPrintTests(unittest.TestCase):
    def _make_inference(self) -> NERInference:
        inference = NERInference.__new__(NERInference)
        inference.id2label = {
            0: "O",
            1: "B-SECTION",
            2: "I-SECTION",
            3: "E-SECTION",
            4: "B-ARTICLE",
            5: "I-ARTICLE",
            6: "E-ARTICLE",
            7: "S-PAGE",
        }
        return inference

    def test_section_trailing_newline_is_outside_closing_tag(self) -> None:
        inference = self._make_inference()
        text = "Section 3.2 Capitalization.\n\n(a) If Final Working Capital exceeds Closing..."
        tokens = ["Section", "3.2", "Capitalization.", "(a)", "If", "Final"]
        offsets = _offsets_for_tokens(text, tokens)
        preds = [1, 2, 3, 0, 0, 0]

        tagged = inference._pretty_print_from_tokens(text, preds, offsets)

        self.assertIn(
            "<section>Section 3.2 Capitalization.</section>\n\n(a)",
            tagged,
        )

    def test_section_trailing_space_is_outside_closing_tag(self) -> None:
        inference = self._make_inference()
        text = "Section 2.04. Adjustment of Purchase Price. (a) If Final Working Capital exceeds Closing..."
        tokens = [
            "Section",
            "2.04.",
            "Adjustment",
            "of",
            "Purchase",
            "Price.",
            "(a)",
            "If",
        ]
        offsets = _offsets_for_tokens(text, tokens)
        preds = [1, 2, 2, 2, 2, 3, 0, 0]

        tagged = inference._pretty_print_from_tokens(text, preds, offsets)

        self.assertIn(
            "<section>Section 2.04. Adjustment of Purchase Price.</section> (a)",
            tagged,
        )

    def test_adjacent_same_type_entities_get_separate_tags(self) -> None:
        inference = self._make_inference()
        text = "3.06 [INTENTIONALLY OMITTED].\n3.07 Absence of Certain Developments. Since"
        tokens = [
            "3.06",
            "[INTENTIONALLY",
            "OMITTED].",
            "3.07",
            "Absence",
            "of",
            "Certain",
            "Developments.",
            "Since",
        ]
        offsets = _offsets_for_tokens(text, tokens)
        preds = [1, 2, 3, 1, 2, 2, 2, 3, 0]

        tagged = inference._pretty_print_from_tokens(text, preds, offsets)

        self.assertEqual(
            "<section>3.06 [INTENTIONALLY OMITTED].</section>\n"
            + "<section>3.07 Absence of Certain Developments.</section> Since",
            tagged,
        )

    def test_adjacent_single_token_entities_get_separate_tags(self) -> None:
        inference = self._make_inference()
        text = "12 13"
        offsets = _offsets_for_tokens(text, ["12", "13"])

        tagged = inference._pretty_print_from_tokens(text, [7, 7], offsets)

        self.assertEqual("<page>12</page> <page>13</page>", tagged)

    def test_single_entity_with_o_gaps_and_trailing_text(self) -> None:
        inference = self._make_inference()
        text = "Intro text  ARTICLE I DEFINITIONS\nbody here.  "
        tokens = ["Intro", "text", "ARTICLE", "I", "DEFINITIONS", "body", "here."]
        offsets = _offsets_for_tokens(text, tokens)
        preds = [0, 0, 4, 5, 6, 0, 0]

        tagged = inference._pretty_print_from_tokens(text, preds, offsets)

        self.assertEqual(
            "Intro text  <article>ARTICLE I DEFINITIONS</article>\nbody here.  ",
            tagged,
        )

    def test_different_type_entities_back_to_back(self) -> None:
        inference = self._make_inference()
        text = "ARTICLE II\nSection 2.1 Sale."
        tokens = ["ARTICLE", "II", "Section", "2.1", "Sale."]
        offsets = _offsets_for_tokens(text, tokens)
        preds = [4, 6, 1, 2, 3]

        tagged = inference._pretty_print_from_tokens(text, preds, offsets)

        self.assertEqual(
            "<article>ARTICLE II</article>\n<section>Section 2.1 Sale.</section>",
            tagged,
        )

    def test_entity_at_end_of_text_is_closed(self) -> None:
        inference = self._make_inference()
        text = "body text 47"
        offsets = _offsets_for_tokens(text, ["body", "text", "47"])

        tagged = inference._pretty_print_from_tokens(text, [0, 0, 7], offsets)

        self.assertEqual("body text <page>47</page>", tagged)

    def test_zero_width_offsets_are_skipped(self) -> None:
        inference = self._make_inference()
        text = "Section 1.1 Terms. Body"
        offsets = _offsets_for_tokens(text, ["Section", "1.1", "Terms.", "Body"])
        offsets.insert(0, (0, 0))
        preds = [0, 1, 2, 3, 0]

        tagged = inference._pretty_print_from_tokens(text, preds, offsets)

        self.assertEqual("<section>Section 1.1 Terms.</section> Body", tagged)


if __name__ == "__main__":
    _ = unittest.main()
