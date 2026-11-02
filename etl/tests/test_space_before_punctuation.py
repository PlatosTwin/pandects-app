# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
"""Body headings and text carry no space before a closing mark at an inline-tag boundary.

The HTML page text joins "<u>Counterparts</u><font></font>." as "Counterparts .", so
served section titles read "Section 11.13 Counterparts ." (d76248fd), "Section 2.07 .
Withholding Rights." (b606d2c0) and article titles "Article V. Non-Competition ;
Non-Disclosure" (c2228ba9). The text also opened ".In the event" after a heading whose
own period sat after the tag (b73800c6 8.8, 2021bbf1 11.20).
"""
from datetime import date
import unittest
import xml.etree.ElementTree as ET

from etl.domain import f_xml

AGREEMENT = "11111111-1111-1111-1111-111111111111"


def _root(tagged_text: str) -> ET.Element:
    return ET.fromstring(f_xml.convert_to_xml(tagged_text, AGREEMENT, date(2024, 1, 1), "https://example.com", "html"))


class TightenSpaceBeforePunctuationTests(unittest.TestCase):
    def test_closing_marks_join_the_word_before(self) -> None:
        t = f_xml.tighten_space_before_punctuation
        self.assertEqual(t("Section 11.13 Counterparts ."), "Section 11.13 Counterparts.")
        self.assertEqual(t("Section 2.07 . Withholding Rights."), "Section 2.07. Withholding Rights.")
        self.assertEqual(t("Non-Competition ; Non-Disclosure"), "Non-Competition; Non-Disclosure")
        self.assertEqual(t("prior to the Escrow Release Date , Mine 8"), "prior to the Escrow Release Date, Mine 8")
        self.assertEqual(t("as follows :"), "as follows:")
        self.assertEqual(t("the “Company” ."), "the “Company”.")
        self.assertEqual(t("ARTICLE IX . MISCELLANEOUS"), "ARTICLE IX. MISCELLANEOUS")

    def test_spaced_marks_that_are_printed_that_way_stay(self) -> None:
        t = f_xml.tighten_space_before_punctuation
        for text in (
            "interest at a .5% rate",
            "1 ,000 shares",
            "at 10 : 30 a.m.",
            "Section 3 . 2",
            "the parties . . . agree",
            "the parties ... agree",
            ". (a) The Company",
            "( a ) the Company",
            "built on .NET",
            "W I T N E S S E T H :",
            "the Company. W I T N E S S E T H :",
        ):
            self.assertEqual(t(text), text)

    def test_section_and_article_titles_are_tightened_and_ids_do_not_move(self) -> None:
        tagged = (
            "<article>Article V. Non-Competition ; Non-Disclosure</article>\n"
            "The parties agree as follows .\n"
            "<section>Section 5.1 Counterparts .</section> This Agreement may be executed in counterparts .\n"
        )
        old = f_xml.tighten_space_before_punctuation
        try:
            f_xml.tighten_space_before_punctuation = lambda s: s
            before = _root(tagged)
        finally:
            f_xml.tighten_space_before_punctuation = old
        after = _root(tagged)
        article, section = after.find(".//article"), after.find(".//section")
        assert article is not None and section is not None
        self.assertEqual(article.get("title"), "Article V. Non-Competition; Non-Disclosure")
        self.assertEqual(section.get("title"), "Section 5.1 Counterparts.")
        self.assertEqual([t.text for t in article.findall("text")], ["The parties agree as follows."])
        self.assertEqual(section.findtext("text"), "This Agreement may be executed in counterparts.")
        old_article, old_section = before.find(".//article"), before.find(".//section")
        assert old_article is not None and old_section is not None
        self.assertEqual(article.get("uuid"), old_article.get("uuid"))
        self.assertEqual(section.get("uuid"), old_section.get("uuid"))

    def test_heading_period_glued_to_the_first_word_is_dropped(self) -> None:
        # b73800c6 8.8: "</section> .In the event"
        root = _root("<section>8.8 Recovery of Common Stock.</section> .In the event Seller is liable.")
        self.assertEqual(root.findtext(".//section/text"), "In the event Seller is liable.")

    def test_leading_decimal_and_sub_number_stay(self) -> None:
        root = _root("<section>2.3 Rate</section>\n.5% per annum shall accrue.")
        self.assertEqual(root.findtext(".//section/text"), ".5% per annum shall accrue.")
        root = _root("<section>3.23 Products</section>.1.1 Except as set forth")
        self.assertEqual(root.findtext(".//section/text"), ".1.1 Except as set forth")


if __name__ == "__main__":
    unittest.main()
