import unittest

from etl.domain.b_pre_processing import (
    format_content,
    normalize_padded_quoted_terms,
    normalize_text,
)


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

    def test_format_content_does_not_add_spaces_inside_quoted_tagged_term(self) -> None:
        for tag in self.FORMATTING_TAGS:
            with self.subTest(tag=tag):
                html = (
                    '<div align="left" style="font-size: 10pt; margin-top: 6pt; margin-left: 4%">'
                    f'&#147;{self._wrap(tag, "NZ Share Consideration")}&#148;'
                    " has the meaning given in clause 3.3(a)(ii);"
                    "</div>"
                )

                text = format_content(html, is_txt=False, is_html=True)

                self.assertIn("“NZ Share Consideration” has the meaning given in clause 3.3(a)(ii);", text)
                self.assertNotIn("“ NZ Share Consideration ”", text)

    def test_format_content_does_not_add_spaces_inside_multi_tag_quoted_term(self) -> None:
        html = (
            "<p>"
            'Parent’s calculation of the Final Merger Consideration (the '
            '“<b>Initial</b><b> Calculation</b>”), including such schedules.'
            "</p>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn('the “Initial Calculation”), including such schedules.', text)
        self.assertNotIn('“ Initial Calculation ”', text)

    def test_format_content_does_not_add_spaces_when_quotes_are_formatting_tags(self) -> None:
        html = (
            "<div>"
            '<b>“</b><u><b>License Agreement</b></u><b>” </b>'
            "has the meaning specified in Section 6.13."
            "</div>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn('“License Agreement” has the meaning specified in Section 6.13.', text)
        self.assertNotIn('“ License Agreement ”', text)

    def test_format_content_does_not_add_space_before_closing_quote_outside_tag(self) -> None:
        html = (
            "<p>"
            '<i>“Sub-Advisory Agreement</i>” means that certain Sub-Advisory Agreement.'
            "</p>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn('“Sub-Advisory Agreement” means that certain Sub-Advisory Agreement.', text)
        self.assertNotIn('“ Sub-Advisory Agreement ”', text)

    def test_format_content_does_not_add_spaces_when_quote_glyphs_are_separate_tags(self) -> None:
        html = (
            "<p>"
            '<i>“</i><b><i>Material IP Agreements</i></b><i>” </i>'
            "means all agreements to which any Company Entity is a party."
            "</p>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn(
            '“Material IP Agreements” means all agreements to which any Company Entity is a party.',
            text,
        )
        self.assertNotIn('“ Material IP Agreements ”', text)

    def test_format_content_does_not_reintroduce_quote_padding_in_tables(self) -> None:
        html = (
            "<table><tr><td>"
            '(the “<u>Cariflex Business</u>”).'
            "</td></tr></table>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertEqual(text, '(the “Cariflex Business”).')

    def test_format_content_does_not_add_quote_padding_after_punctuation_tag(self) -> None:
        html = (
            "<div>"
            "<b>Legal Requirement</b><b><i>. </i></b>"
            '<i>“</i><b><i>Legal Requirement</i></b><i>” </i>'
            "shall mean any law."
            "</div>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn('Legal Requirement. “Legal Requirement” shall mean any law.', text)
        self.assertNotIn('“ Legal Requirement ”', text)

    def test_format_content_does_not_add_quote_padding_with_empty_tags_inside_quotes(self) -> None:
        html = (
            "<p>"
            'Governmental Authorization. “<b></b><b><i>Governmental Authorization</i></b><b></b>” '
            "shall mean any permit."
            "</p>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn('Governmental Authorization. “Governmental Authorization” shall mean any permit.', text)
        self.assertNotIn('“ Governmental Authorization ”', text)

    def test_format_content_trims_nbsp_inside_quoted_term_tag(self) -> None:
        html = (
            "<p>"
            '(the “\u00a0<u>Closing Date\u00a0</u>”).'
            "</p>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertEqual(text, '(the “Closing Date”).')

    def test_format_content_handles_punctuation_plus_quote_wrapper_tags(self) -> None:
        html = (
            "<p>"
            'Parent <b>(“</b><b><i>Merger LLC</i></b><b>”) </b>and Company'
            "</p>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn('Parent (“Merger LLC”) and Company', text)
        self.assertNotIn('“ Merger LLC ”', text)

    def test_format_content_handles_empty_inline_tags_around_all_caps_definition(self) -> None:
        html = (
            "<p>"
            '<i></i>“<i></i><b>COFECE</b><i></i>” means the Mexican Federal Anti-Trust Commission.'
            "</p>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn('“COFECE” means the Mexican Federal Anti-Trust Commission.', text)
        self.assertNotIn('“ COFECE ”', text)

    def test_format_content_handles_phase_ii_with_empty_inline_tags(self) -> None:
        html = (
            "<p>"
            '(“<i></i><b><i>Phase II</i></b><i></i>”), Purchaser may submit a written request.'
            "</p>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn('(“Phase II”), Purchaser may submit a written request.', text)
        self.assertNotIn('“ Phase II ”', text)

    def test_format_content_keeps_css_hidden_text(self) -> None:
        body_text = " ".join(["agreement clause"] * 2000)
        html = f"<div style='display:none'>{body_text}</div><p>Visible heading</p>"

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn("Visible heading", text)
        self.assertIn("agreement clause", text)
        self.assertGreater(len(text), 20000)

    def test_format_content_drops_semantically_hidden_text(self) -> None:
        body_text = " ".join(["agreement clause"] * 2000)
        html = f"<div hidden>{body_text}</div><p>Visible heading</p>"

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn("Visible heading", text)
        self.assertNotIn("agreement clause", text)
        self.assertLess(len(text), 20000)

    def test_format_content_uses_relaxed_fallback_on_suspicious_shrink(self) -> None:
        body_text = " ".join(["agreement clause"] * 2000)
        html = (
            "<html><head><title>"
            f"{body_text}"
            "</title></head><body><p>Visible heading</p></body></html>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn("Visible heading", text)
        self.assertIn("agreement clause", text)
        self.assertGreater(len(text), 20000)

    def test_normalize_text_strips_disallowed_control_chars(self) -> None:
        dirty = "A\x00B\x08C\x0bD\tE\n\nF\rG"

        normalized = normalize_text(dirty)

        self.assertEqual(normalized, "ABCD\tE\n\nF\rG")

    def test_normalize_padded_quoted_terms_is_idempotent(self) -> None:
        text = (
            '" Escrow Agent " means Colorado Business Bank.\n\n'
            '" Excluded Assets " means office furniture.'
        )

        first = normalize_padded_quoted_terms(text)
        second = normalize_padded_quoted_terms(first)

        self.assertEqual(
            first,
            (
                '"Escrow Agent" means Colorado Business Bank.\n\n'
                '"Excluded Assets" means office furniture.'
            ),
        )
        self.assertEqual(second, first)

    def test_normalize_padded_quoted_terms_preserves_outside_spaces_for_straight_quotes(self) -> None:
        text = (
            '(the " WARN Act ") and the Merger shall qualify as a " reorganization " '
            'within the meaning of Section 368(a).'
        )

        normalized = normalize_padded_quoted_terms(text)

        self.assertEqual(
            normalized,
            (
                '(the "WARN Act") and the Merger shall qualify as a "reorganization" '
                'within the meaning of Section 368(a).'
            ),
        )


if __name__ == "__main__":
    _ = unittest.main()
