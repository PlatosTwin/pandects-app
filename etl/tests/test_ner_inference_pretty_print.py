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


if __name__ == "__main__":
    _ = unittest.main()
