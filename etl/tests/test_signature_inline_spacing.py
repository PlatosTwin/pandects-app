# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
"""Inline-element boundaries on HTML signature pages add no space next to punctuation.

The HTML in these tests is cut from the raw sig pages of served agreements whose
sigPages read "/ S /", "BRISTOL -MYERS", "STOCKHOLDERS ’", "10. 9" and
"(“ PROVIDER ”)" while the page prints them tight: 2f99114a (page_order 73),
01d86e89 (52), 0ed3301e (80), 60a6a850 (95), 17c43000 (65).
"""
import unittest

from etl.domain import f_xml


def _lines(raw_html: str) -> list[str]:
    return [line for line in f_xml.format_signature_html_like_screen(raw_html).splitlines() if line]


class SignatureInlineSpacingTests(unittest.TestCase):
    def test_small_caps_conformed_signature(self) -> None:
        # 2f99114a
        raw_html = (
            '<table><tr><td valign="top"><font face="Times New Roman" size="2">By:</font></td>'
            '<td valign="bottom"><font size="1"> </font></td>'
            '<td valign="bottom"> <p style="margin-bottom:1px;border-bottom:1px solid #000000">'
            '<font face="Times New Roman" size="2">/<small>S</small>/ L<small>ESLIE</small> '
            "H<small>UDSON</small>, P<small>H</small>.D.</font></p></td></tr></table>"
        )
        self.assertEqual(_lines(raw_html), ["By: /S/ LESLIE HUDSON, PH.D."])

    def test_hyphen_in_a_small_caps_name(self) -> None:
        # 01d86e89
        raw_html = (
            '<table><tr><td colspan="3" valign="top"><b>B<small>RISTOL</small>-M<small>YERS</small> '
            "S<small>QUIBB</small> C<small>OMPANY</small></b></td></tr></table>"
        )
        self.assertEqual(_lines(raw_html), ["BRISTOL-MYERS SQUIBB COMPANY"])

    def test_apostrophe_in_its_own_font_run(self) -> None:
        # 0ed3301e
        f11 = "<font style=\"color:#000000;font-family:'Times New Roman',sans-serif;font-size:11pt;font-weight:400;line-height:120%\">"
        f9 = "<font style=\"color:#000000;font-family:'Times New Roman',sans-serif;font-size:9pt;font-weight:400;line-height:120%\">"
        raw_html = (
            '<table><tr><td colspan="6" style="padding:2px 1pt;text-align:left;vertical-align:bottom"><div>'
            f"{f11}S</font>{f9}TOCKHOLDERS</font>{f11}’ A</font>{f9}GENT</font></div></td></tr></table>"
        )
        self.assertEqual(_lines(raw_html), ["STOCKHOLDERS’ AGENT"])

    def test_section_number_split_across_runs(self) -> None:
        # 60a6a850
        raw_html = (
            '<p style="margin:0in 0in .0001pt;"><font face="Times New Roman" size="2" style="font-size:10.0pt;">'
            "Solely   and exclusively for the limited purpose of <u>Sections 10.</u>9 and <u>11.13</u>:</font></p>"
        )
        self.assertEqual(_lines(raw_html), ["Solely and exclusively for the limited purpose of Sections 10.9 and 11.13:"])

    def test_quotes_around_a_bold_defined_term(self) -> None:
        # 17c43000
        raw_html = (
            "<div>by and between Leon Medical Centers Health Plans, Inc., a Florida corporation "
            "(“<b>PLAN</b>”), and Leon Medical Centers, Inc., a Florida corporation (“<b>PROVIDER</b>”).</div>"
        )
        self.assertEqual(
            _lines(raw_html),
            [
                "by and between Leon Medical Centers Health Plans, Inc., a Florida corporation (“PLAN”), "
                "and Leon Medical Centers, Inc., a Florida corporation (“PROVIDER”)."
            ],
        )

    def test_label_colon_and_underscore_blank_keep_their_gap(self) -> None:
        raw_html = "<p><font>Name:</font><font>Steven D. Barnhart</font></p><p>DiModica<u>__________</u></p>"
        self.assertEqual(_lines(raw_html), ["Name: Steven D. Barnhart", "DiModica __________"])

    def test_conformed_signature_keeps_the_gap_before_the_name(self) -> None:
        # 8066d292 (page_order 48) prints "By: /s/ Steve Rosenfield"
        self.assertEqual(_lines("<p>By:<font>/s/</font><u>Steve Rosenfield</u></p>"), ["By: /s/ Steve Rosenfield"])
        self.assertEqual(_lines("<p>By: /<small>S</small>/<u>Jane Doe</u></p>"), ["By: /S/ Jane Doe"])
        self.assertEqual(_lines("<p>ACME A/<b>S</b>/<i>B</i></p>"), ["ACME A/S/B"])

    def test_whitespace_in_the_source_still_separates(self) -> None:
        raw_html = "<p>By: <font>/s/</font> <font>Jane Doe</font></p>"
        self.assertEqual(_lines(raw_html), ["By: /s/ Jane Doe"])


if __name__ == "__main__":
    unittest.main()
