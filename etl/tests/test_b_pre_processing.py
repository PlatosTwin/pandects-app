import unittest

from etl.domain.b_pre_processing import format_content


class PreProcessingTests(unittest.TestCase):
    FORMATTING_TAGS = [
        "font",
        "span",
        "b",
        "strong",
        "i",
        "em",
        "u",
        "small",
        "big",
        "sup",
        "sub",
        "tt",
        "s",
        "strike",
        "ins",
        "del",
        "a",
    ]

    def _wrap(self, tag: str, text: str) -> str:
        if tag == "a":
            return f'<a href="#">{text}</a>'
        return f"<{tag}>{text}</{tag}>"

    def test_format_content_keeps_split_definitions_word_joined(self) -> None:
        for tag in self.FORMATTING_TAGS:
            with self.subTest(tag=tag):
                html = (
                    '<P STYLE="font-size:10pt; font-family:Times New Roman" ALIGN="center">'
                    f"D{self._wrap(tag, 'EFINITIONS')} "
                    "</P>"
                )

                text = format_content(html, is_txt=False, is_html=True)

                self.assertIn("DEFINITIONS", text)
                self.assertNotIn("D EFINITIONS", text)

    def test_format_content_keeps_space_before_tagged_phrase_after_number(self) -> None:
        for tag in self.FORMATTING_TAGS:
            with self.subTest(tag=tag):
                html = f"<p>Section 1.1{self._wrap(tag, 'Defined Terms')}</p>"

                text = format_content(html, is_txt=False, is_html=True)

                self.assertEqual(text, "Section 1.1 Defined Terms")

    def test_format_content_keeps_space_between_word_and_tagged_number(self) -> None:
        for tag in self.FORMATTING_TAGS:
            with self.subTest(tag=tag):
                html = f"<p>Section{self._wrap(tag, '1.1')}</p>"

                text = format_content(html, is_txt=False, is_html=True)

                self.assertEqual(text, "Section 1.1")

    def test_format_content_preserves_space_before_font_after_period(self) -> None:
        for tag in self.FORMATTING_TAGS:
            with self.subTest(tag=tag):
                html = f"<p>Section 1.{self._wrap(tag, 'The Merger')}</p>"

                text = format_content(html, is_txt=False, is_html=True)

                self.assertEqual(text, "Section 1. The Merger")

    def test_format_content_keeps_space_after_number_with_empty_anchor_between(self) -> None:
        html = (
            "<h2><font>3.22</font><font>           </font>"
            "<a name='x'></a><a name='y'><u>Information in the Proxy Statement</u></a>.</h2>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn("3.22 Information in the Proxy Statement.", text)

    def test_format_content_does_not_add_space_before_punctuation(self) -> None:
        html = "<h2><font>3.21</font><font>      </font><a><u>Financial Advisor</u></a>.</h2>"

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn("3.21 Financial Advisor.", text)
        self.assertNotIn("Financial Advisor .", text)

    def test_format_content_does_not_split_section_ref_before_closing_paren(self) -> None:
        html = "<p>Section 6.2<b><font>)</font></b></p>"

        text = format_content(html, is_txt=False, is_html=True)

        self.assertEqual(text, "Section 6.2)")


if __name__ == "__main__":
    _ = unittest.main()
