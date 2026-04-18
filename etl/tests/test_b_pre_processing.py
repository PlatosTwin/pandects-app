import unittest
from typing import cast
from unittest.mock import Mock, patch

import requests

import etl.domain.b_pre_processing as b_pre_processing
from etl.domain.b_pre_processing import (
    AgreementRow,
    ContextProtocol,
    format_content,
    normalize_padded_quoted_terms,
    normalize_text,
    pre_process,
    strip_rendered_navigation_chrome,
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

    def test_format_content_drops_display_none_text(self) -> None:
        body_text = " ".join(["agreement clause"] * 2000)
        html = f"<div style='display:none'>{body_text}</div><p>Visible heading</p>"

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn("Visible heading", text)
        self.assertNotIn("agreement clause", text)
        self.assertLess(len(text), 20000)

    def test_format_content_drops_semantically_hidden_text(self) -> None:
        body_text = " ".join(["agreement clause"] * 2000)
        html = f"<div hidden>{body_text}</div><p>Visible heading</p>"

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn("Visible heading", text)
        self.assertNotIn("agreement clause", text)
        self.assertLess(len(text), 20000)

    def test_format_content_drops_visibility_hidden_text(self) -> None:
        html = (
            "<table>"
            "<tr style='visibility:hidden; line-height:0pt; color:white'>"
            "<td>September 30,</td>"
            "</tr>"
            "<tr><td>Joint Proxy Statement/Prospectus</td></tr>"
            "</table>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn("Joint Proxy Statement/Prospectus", text)
        self.assertNotIn("September 30,", text)

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

    def test_format_content_preserves_order_for_top_level_inline_section_heads(self) -> None:
        html = (
            "<html><body>"
            "<p>(a) Attached as Exhibit A.</p>"
            "<p>(b) All notes and accounts receivable.</p>"
            '<a name="_Toc449965823"></a>'
            '<font style="display:inline;font-weight:bold">4.7</font>'
            '<font style="display:inline;text-decoration:underline">Abse</font>'
            '<a name="ARTICLE4_7"></a>'
            '<font style="display:inline;text-decoration:underline">'
            "nce of Undisclosed Liabilities"
            "</font>"
            "<p>. Seller does not have any material Liabilities.</p>"
            '<a name="_Toc449965824"></a>'
            '<font style="display:inline;font-weight:bold">4.8</font>'
            '<font style="display:inline;text-decoration:underline">Absence o</font>'
            '<a name="ARTICLE4_8"></a>'
            '<font style="display:inline;text-decoration:underline">f Changes or Events</font>'
            "<p>. Since September 30, 2015, no Event has occurred.</p>"
            "</body></html>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn("4.7 Absence of Undisclosed Liabilities", text)
        self.assertIn("4.8 Absence of Changes or Events", text)
        self.assertNotIn("Abse\n\nnce", text)
        self.assertNotIn("Absence o\n\nf", text)
        self.assertLess(
            text.index("(b) All notes and accounts receivable."),
            text.index("4.7 Absence of Undisclosed Liabilities"),
        )
        self.assertLess(
            text.index("4.7 Absence of Undisclosed Liabilities"),
            text.index("4.8 Absence of Changes or Events"),
        )

    def test_format_content_preserves_br_separated_inline_lines(self) -> None:
        html = (
            "<br/>"
            '<p><br/><a name=\"02WLA1440_2\">QuickLinks</a><br/></p>'
            '<font size=\"2\"><a href=\"#toc_ke1440_1\">'
            "AGREEMENT AND PLAN OF REORGANIZATION by and among UNIVISION COMMUNICATIONS INC., "
            "UNIVISION ACQUISITION CORPORATION and HISPANIC BROADCASTING CORPORATION "
            "dated as of June 11, 2002"
            "</a></font><br/>"
            '<font size=\"2\"><a href=\"#toc_ke1440_2\">TABLE OF CONTENTS</a></font><br/>'
            '<font size=\"2\"><a href=\"#toc_kg1440_1\">AGREEMENT AND PLAN OF REORGANIZATION</a></font><br/>'
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn("QuickLinks", text)
        self.assertIn("TABLE OF CONTENTS", text)
        self.assertIn(
            "AGREEMENT AND PLAN OF REORGANIZATION by and among UNIVISION COMMUNICATIONS INC.",
            text,
        )
        self.assertIn("\n\nTABLE OF CONTENTS\n\n", text)
        self.assertNotIn(
            "dated as of June 11, 2002 TABLE OF CONTENTS AGREEMENT AND PLAN OF REORGANIZATION",
            text,
        )

    def test_format_content_preserves_dt_dd_boundaries(self) -> None:
        html = (
            "<dl>"
            "<dt>(a)</dt>"
            "<dd>First clause text.</dd>"
            "<dt>(b)</dt>"
            "<dd>Second clause text.</dd>"
            "</dl>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn("(a)\n\nFirst clause text.\n\n(b)\n\nSecond clause text.", text)
        self.assertNotIn("(a)First clause text.", text)
        self.assertNotIn("(b)Second clause text.", text)

    def test_format_content_moves_leading_quicklinks_to_footer(self) -> None:
        html = (
            "<br/>"
            '<p><br/><a name="q1">QuickLinks</a><br/></p>'
            '<font size="2"><a href="#toc1">TABLE OF CONTENTS</a></font><br/>'
            '<font size="2"><a href="#toc2">ARTICLE I MERGER</a></font><br/>'
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn("TABLE OF CONTENTS", text)
        self.assertIn("ARTICLE I MERGER", text)
        self.assertTrue(text.endswith("QuickLinks"))
        self.assertLess(text.index("TABLE OF CONTENTS"), text.index("QuickLinks"))

    def test_format_content_preserves_nbsp_only_inline_tag_between_words(self) -> None:
        html = (
            "<p>"
            '<font style="display:inline;color:#000000;">since</font>'
            '<font style="display:inline;font-weight:bold;color:#000000;">\u00a0</font>'
            '<font style="display:inline;color:#000000;">Reference Date, Seller has not</font>'
            "</p>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn("since Reference Date, Seller has not", text)
        self.assertNotIn("sinceReference Date", text)

    def test_format_content_preserves_break_wrapped_by_formatting_tag(self) -> None:
        html = (
            "<p>"
            "<font>Article I</font>"
            "<font><br/></font>"
            "<font>Definitions</font>"
            "</p>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertEqual(text, "Article I Definitions")

    def test_format_content_preserves_heading_space_for_inline_br_within_formatting_tag(self) -> None:
        html = (
            "<div style='margin-bottom:12pt;text-align:center'>"
            "<font style='font-weight:bold'>EXHIBIT A<br/><br/>DEFINED TERMS</font>"
            "</div>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertEqual(text, "EXHIBIT A DEFINED TERMS")

    def test_format_content_preserves_section_spacing_for_inline_br_within_formatting_tag(self) -> None:
        html = (
            "<div style='text-align:center'>"
            "<font style='font-weight:bold'>Section 1.2<br/>Representations</font>"
            "</div>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertEqual(text, "Section 1.2 Representations")

    def test_format_content_preserves_article_heading_space_across_adjacent_inline_tags(self) -> None:
        html = "<p>ARTICLE <b>II</b><b>THE MERGERS</b></p>"

        text = format_content(html, is_txt=False, is_html=True)

        self.assertEqual(text, "ARTICLE II THE MERGERS")

    def test_format_content_preserves_article_heading_space_when_next_tag_starts_with_br(self) -> None:
        html = (
            "<div>"
            "<font>ARTICLE II</font>"
            "<font><br/>THE MERGERS</font>"
            "</div>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertEqual(text, "ARTICLE II THE MERGERS")

    def test_format_content_collapses_source_newlines_inside_inline_formatting_tag(self) -> None:
        html = (
            "<div style='text-align:justify;text-indent:36pt'>"
            "<font>1.1.</font> <font><u>The\n\n\n\n          Merger</u></font>. Upon the terms."
            "</div>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn("1.1. The Merger. Upon the terms.", text)
        self.assertNotIn("The\n\nMerger", text)

    def test_format_content_preserves_article_heading_space_for_nested_malformed_inline_tags(self) -> None:
        html = (
            "<p>"
            "<font><b>Article&nbsp;I</b></font>"
            "<b><font style='text-transform: uppercase'><br/></font>"
            "THE MERGER AND THE HOLDCO MERGER</b>"
            "</p>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertEqual(text, "Article I THE MERGER AND THE HOLDCO MERGER")

    def test_format_content_does_not_leave_trailing_space_before_heading_line_break(self) -> None:
        html = (
            "<div>"
            "<font>Article XI</font>"
            "<a name='x'></a><font style='display:inline-block;visibility:hidden;width:0pt;'>\u200b</font><br/>"
            "<font style='display:inline-block;visibility:hidden;width:0pt;'>\u200b</font><br/>"
            "<b>Miscellaneous</b>"
            "</div>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertEqual(text, "Article XI\n\nMiscellaneous")

    def test_format_content_does_not_add_space_before_closing_quote_with_empty_inline_tags(self) -> None:
        html = (
            "<p>"
            '<font style="display:inline;color:#000000;">. Since September 30, 2015 (the “</font>'
            '<font style="display:inline;font-weight:bold;color:#000000;">Reference Date</font>'
            '<font style="display:inline;font-weight:bold;color:#000000;"></font>'
            '<font style="display:inline;color:#000000;"></font>'
            '<font style="display:inline;color:#000000;"></font>'
            '<font style="display:inline;color:#000000;">”), no Event has occurred that</font>'
            "</p>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertIn('the “Reference Date”), no Event has occurred', text)
        self.assertNotIn('the “Reference Date ”', text)

    def test_normalize_text_strips_disallowed_control_chars(self) -> None:
        dirty = "A\x00B\x08C\x0bD\tE\n\nF\rG"

        normalized = normalize_text(dirty)

        self.assertEqual(normalized, "ABCD E\n\nF G")

    def test_normalize_text_strips_zero_width_separators(self) -> None:
        dirty = "A\u200bB\u200cC\u200dD\u2060E\ufeffF"

        normalized = normalize_text(dirty)

        self.assertEqual(normalized, "ABCDEF")

    def test_normalize_text_strips_bidi_format_controls(self) -> None:
        dirty = "A\u200eB\u200fC\u202aD\u202bE\u202cF\u202dG\u202eH"

        normalized = normalize_text(dirty)

        self.assertEqual(normalized, "ABCDEFGH")

    def test_normalize_text_normalizes_unicode_horizontal_spaces(self) -> None:
        dirty = "2.2\u00a0\u202f\u2007Rights of Former VBI Shareholders."

        normalized = normalize_text(dirty)

        self.assertEqual(normalized, "2.2 Rights of Former VBI Shareholders.")

    def test_normalize_text_strips_line_final_spaces(self) -> None:
        dirty = "Alpha. \n\nBeta\t \n\nGamma"

        normalized = normalize_text(dirty)

        self.assertEqual(normalized, "Alpha.\n\nBeta\n\nGamma")

    def test_normalize_text_normalizes_crlf_and_tab_runs(self) -> None:
        dirty = "Alpha\t\tBeta\r\n\r\nGamma\rDelta"

        normalized = normalize_text(dirty)

        self.assertEqual(normalized, "Alpha Beta\n\nGamma Delta")

    def test_format_content_strips_trailing_space_before_paragraph_break(self) -> None:
        html = (
            "<p>“Accounts Receivable” means all in accordance with GAAP.\n</p> "
            "<p>“Action” means any claim.</p>"
        )

        text = format_content(html, is_txt=False, is_html=True)

        self.assertEqual(
            text,
            "“Accounts Receivable” means all in accordance with GAAP.\n\n“Action” means any claim.",
        )

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

    def test_strip_rendered_navigation_chrome_removes_quicklinks_banner(self) -> None:
        text = (
            "EX-2.1 3 a2090849zex-2_1.htm EXHIBIT 2.1\n"
            "QuickLinks -- Click here to rapidly navigate through this document\n\n"
            "Execution Copy"
        )

        cleaned = strip_rendered_navigation_chrome(text)

        self.assertEqual(
            cleaned,
            "EX-2.1 3 a2090849zex-2_1.htm EXHIBIT 2.1\n\nExecution Copy",
        )

    def test_strip_rendered_navigation_chrome_removes_split_quicklinks_banner(self) -> None:
        text = "QuickLinks\n-- Click here to rapidly navigate through this document\n\nExhibit 2.1"

        cleaned = strip_rendered_navigation_chrome(text)

        self.assertEqual(cleaned, "Exhibit 2.1")

    def test_strip_rendered_navigation_chrome_removes_back_to_contents_near_top(self) -> None:
        text = "EX-2.1 2 foo.htm EXHIBIT 2.1\nBack to Contents\n\nARTICLE I\n\nTHE MERGER"

        cleaned = strip_rendered_navigation_chrome(text)

        self.assertEqual(cleaned, "EX-2.1 2 foo.htm EXHIBIT 2.1\n\nARTICLE I\n\nTHE MERGER")

    def test_strip_rendered_navigation_chrome_keeps_table_of_contents_heading(self) -> None:
        text = "EX-2.1 2 foo.htm EXHIBIT 2.1\n\nTABLE OF CONTENTS\n\nARTICLE I"

        cleaned = strip_rendered_navigation_chrome(text)

        self.assertEqual(cleaned, text)

    def test_format_content_uses_legacy_html_formatter_by_default(self) -> None:
        html = "<div>ARTICLE I</div><div>THE MERGER</div>"

        with patch.object(
            b_pre_processing,
            "_format_html_content_legacy",
            return_value="legacy-result",
        ) as mock_legacy, patch.object(
            b_pre_processing,
            "_format_html_content_rendered",
            return_value="rendered-result",
        ) as mock_rendered, patch.dict("os.environ", {}, clear=False):
            text = format_content(html, is_txt=False, is_html=True)

        self.assertEqual(text, "legacy-result")
        mock_legacy.assert_called_once_with(html)
        mock_rendered.assert_not_called()

    def test_format_content_uses_rendered_html_formatter_when_gated_on(self) -> None:
        html = "<div>ARTICLE I</div><div>THE MERGER</div>"

        with patch.object(
            b_pre_processing,
            "_format_html_content_legacy",
            return_value="legacy-result",
        ) as mock_legacy, patch.object(
            b_pre_processing,
            "_format_html_content_rendered",
            return_value="rendered-result",
        ) as mock_rendered, patch.dict(
            "os.environ",
            {"ETL_HTML_FORMATTER_MODE": "rendered"},
            clear=False,
        ):
            text = format_content(html, is_txt=False, is_html=True)

        self.assertEqual(text, "rendered-result")
        mock_rendered.assert_called_once_with(html)
        mock_legacy.assert_not_called()

    def test_rendered_html_text_cleanup_strips_navigation_chrome(self) -> None:
        dumped_dom = (
            "<html><body><pre id=\"rendered-text\">"
            "EX-2.1 3 a2090849zex-2_1.htm EXHIBIT 2.1\n"
            "QuickLinks -- Click here to rapidly navigate through this document\n\n"
            "Execution Copy"
            "</pre></body></html>"
        )

        with patch.object(
            b_pre_processing.subprocess,
            "run",
            return_value=Mock(stdout=dumped_dom),
        ):
            text = b_pre_processing._render_visible_html_text("<div>ignored</div>")

        self.assertEqual(
            text,
            "EX-2.1 3 a2090849zex-2_1.htm EXHIBIT 2.1\n\nExecution Copy",
        )

    def test_pull_agreement_content_retries_read_timeout(self) -> None:
        class _SuccessResponse:
            text = "<html>ok</html>"

            def raise_for_status(self) -> None:
                return None

        with (
            patch.object(
                b_pre_processing,
                "requests_get",
                side_effect=[requests.exceptions.ReadTimeout("slow"), _SuccessResponse()],
            ) as mock_get,
            patch.object(b_pre_processing, "_wait_for_sec_rate_limit_slot"),
            patch("etl.domain.b_pre_processing.time.sleep") as mock_sleep,
        ):
            content = b_pre_processing.pull_agreement_content(
                "https://www.sec.gov/Archives/test.html",
                timeout=1.0,
                max_attempts=2,
                backoff_seconds=0.5,
            )

        self.assertEqual(content, "<html>ok</html>")
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once_with(0.5)

    def test_pre_process_skips_agreement_after_fetch_retries_exhausted(self) -> None:
        class _FakeLogger:
            def __init__(self) -> None:
                self.messages: list[str] = []

            def info(self, msg: str) -> None:
                self.messages.append(msg)

        class _FakeContext:
            def __init__(self) -> None:
                self.log = _FakeLogger()

        context = _FakeContext()
        agreements: list[AgreementRow] = [
            AgreementRow(
                agreement_uuid="agreement-timeout",
                url="https://www.sec.gov/Archives/bad.html",
            ),
            AgreementRow(
                agreement_uuid="agreement-ok",
                url="https://www.sec.gov/Archives/good.html",
            ),
        ]
        split_pages = [{"content": f"page {i}"} for i in range(11)]

        with (
            patch.object(
                b_pre_processing,
                "pull_agreement_content",
                side_effect=[
                    requests.exceptions.ReadTimeout("still slow"),
                    "<html>good</html>",
                ],
            ),
            patch.object(b_pre_processing, "split_to_pages", return_value=split_pages),
            patch.object(b_pre_processing, "_format_page", return_value="Clause text"),
            patch.object(b_pre_processing, "_count_alpha_tokens", return_value=3),
            patch.object(b_pre_processing, "classify", return_value=[]),
            patch.object(b_pre_processing, "_attach_preds_to_pages"),
            patch.object(b_pre_processing, "_attach_review_predictions_to_pages"),
        ):
            staged_pages, pagination_statuses = pre_process(
                cast(ContextProtocol, cast(object, context)),
                agreements,
                classifier_model=Mock(),
                review_model=Mock(),
            )

        self.assertEqual(len(staged_pages), 11)
        self.assertNotIn("agreement-timeout", pagination_statuses)
        self.assertTrue(pagination_statuses["agreement-ok"])
        self.assertTrue(
            any(
                "agreement-timeout" in message and "Skipping for now" in message
                for message in context.log.messages
            )
        )


if __name__ == "__main__":
    _ = unittest.main()
