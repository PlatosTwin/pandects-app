# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
"""Ordinals and possessives that an inline-tag boundary split are rejoined when the raw HTML confirms it.

The HTML page text joins "15<sup>th</sup>" as "15 th" and "<u>Broker</u><u>’</u><u>s Fees</u>"
as "Broker’ s Fees", so served titles read "3.7 Broker’ s Fees." (5ccd90be) and text reads
"prior to the 46 th day" (f7481a80) and "Four Embarcadero Center, 22 nd Floor" (f7481a80).
"""
from datetime import date
import unittest
import xml.etree.ElementTree as ET

import pandas as pd

from etl.domain import f_xml
from etl.domain.b_pre_processing import format_content
from etl.domain.inline_tag_splits import InlineSplitEvidence, raw_inline_plain

AGREEMENT = "11111111-1111-1111-1111-111111111111"

BROKER_RAW = (
    '<p style="margin-top:6pt; text-indent:8%">3.7 <u>Broker</u><u>’</u><u>s Fees</u>. Neither the Company '
    "nor any Company Subsidiary has employed any broker, finder or financial advisor or incurred any "
    "liability for any broker’s fees.</p>"
)
ORDINAL_RAW = (
    "<p>the Buyer shall not be obligated to consummate the Closing prior to the 46<sup>th</sup> day "
    "following the date of this Agreement.</p><p>Four Embarcadero Center, 22<sup>nd</sup> Floor</p>"
    "<p>through December 31<font style='vertical-align:super'>st</font> of the year</p>"
)


def _root(tagged_text: str, raw_pages: list[str] | None) -> ET.Element:
    evidence = InlineSplitEvidence(raw_pages) if raw_pages is not None else None
    return ET.fromstring(
        f_xml.convert_to_xml(
            tagged_text, AGREEMENT, date(2024, 1, 1), "https://example.com", "html", inline_splits=evidence
        )
    )


class InlineSplitEvidenceTests(unittest.TestCase):
    def test_raw_plain_drops_inline_tags_without_a_space(self) -> None:
        self.assertIn("3.7 Broker's Fees.", raw_inline_plain(BROKER_RAW))
        self.assertIn("the 46th day", raw_inline_plain(ORDINAL_RAW))
        self.assertIn("22nd Floor", raw_inline_plain(ORDINAL_RAW))
        # Block tags and <br> are breaks the reader sees.
        self.assertEqual(raw_inline_plain("<p>15</p><p>th</p>").strip(), "15 th")
        self.assertEqual(raw_inline_plain("15<br/>th").strip(), "15 th")

    def test_split_ordinals_and_possessives_rejoin(self) -> None:
        e = InlineSplitEvidence([BROKER_RAW, ORDINAL_RAW])
        self.assertEqual(e.join("3.7 Broker’ s Fees."), "3.7 Broker’s Fees.")
        self.assertEqual(e.join("prior to the 46 th day following"), "prior to the 46th day following")
        self.assertEqual(e.join("Four Embarcadero Center, 22 nd Floor"), "Four Embarcadero Center, 22nd Floor")
        self.assertEqual(e.join("through December 31 st of the year"), "through December 31st of the year")

    def test_space_before_the_apostrophe_rejoins(self) -> None:
        raw = "<p>the <b>Company</b>’s Subsidiaries and the Company<u>’s</u> Board</p>"
        e = InlineSplitEvidence([raw])
        self.assertEqual(e.join("the Company ’s Subsidiaries"), "the Company’s Subsidiaries")

    def test_upper_case_possessives_rejoin(self) -> None:
        raw = "<p>BUYER<u>’</u><u>S GUARANTOR</u><u>’</u><u>S OBLIGATIONS</u></p>"
        e = InlineSplitEvidence([raw])
        self.assertEqual(e.join("BUYER’ S GUARANTOR’ S OBLIGATIONS"), "BUYER’S GUARANTOR’S OBLIGATIONS")

    def test_a_space_the_raw_prints_stays(self) -> None:
        # The raw has the space, or a break, between the parts.
        for raw, text in (
            ("<p>the 15 <sup>th</sup> day</p>", "the 15 th day"),
            ("<p>the 15<sup> th</sup> day</p>", "the 15 th day"),
            ("<p>the 15&nbsp;th day</p>", "the 15 th day"),
            ("<p>Party’ s Fees</p>", "Party’ s Fees"),
            ("<p>Exhibit A’ s terms</p>", "Exhibit A’ s terms"),
            ("<p>Company ’s</p>", "Company ’s"),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(InlineSplitEvidence([raw]).join(text), text)

    def test_the_spaced_form_printed_anywhere_blocks_the_join(self) -> None:
        raw = ["<p>the 15<sup>th</sup> day</p>", "<p>on the 15 th of May</p>"]
        self.assertEqual(InlineSplitEvidence(raw).join("the 15 th day"), "the 15 th day")

    def test_unconfirmed_or_unlike_splits_stay(self) -> None:
        e = InlineSplitEvidence([ORDINAL_RAW, BROKER_RAW])
        for text in (
            "the 47 th day",  # not printed joined anywhere
            "the 46 st day",  # the suffix does not fit the number
            "Section 4.1 0",  # a split number is not an ordinal
            "1,000 th",
            "the Lenders’ S&P rating",  # "Lenders'S" is not printed
            "the Seller’ s Fees",  # "Seller's" is not printed
        ):
            with self.subTest(text=text):
                self.assertEqual(e.join(text), text)
        # The raw prints "broker’s" joined in the same text, so the lower-case form rejoins too.
        self.assertEqual(e.join("the broker’ s fees"), "the broker’s fees")

    def test_no_raw_no_change(self) -> None:
        self.assertEqual(InlineSplitEvidence([]).join("the 46 th day"), "the 46 th day")


class ConvertToXmlTests(unittest.TestCase):
    TAGGED = (
        "<article>ARTICLE III REPRESENTATIONS</article>"
        "<section>3.7 Broker’ s Fees.</section> Neither the Company nor any Company Subsidiary has "
        "employed any broker.\nprior to the 46 th day following the date of this Agreement.\n"
        "<pageUUID>p1</pageUUID>"
    )

    def test_titles_and_text_rejoin_and_ids_stay(self) -> None:
        old = _root(self.TAGGED, None)
        new = _root(self.TAGGED, [BROKER_RAW, ORDINAL_RAW])
        old_section, new_section = old.find(".//section"), new.find(".//section")
        assert old_section is not None and new_section is not None
        self.assertEqual(old_section.get("title"), "3.7 Broker’ s Fees.")
        self.assertEqual(new_section.get("title"), "3.7 Broker’s Fees.")
        self.assertEqual(old_section.get("uuid"), new_section.get("uuid"))
        texts = [t.text for t in new_section.iter("text")]
        self.assertIn("prior to the 46th day following the date of this Agreement.", texts)

    def test_generate_xml_uses_the_html_body_pages(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "agreement_uuid": AGREEMENT,
                    "page_uuid": "p1",
                    "page_order": 1,
                    "raw_page_content": BROKER_RAW + ORDINAL_RAW,
                    "source_page_type": "body",
                    "tagged_output": self.TAGGED.replace("<pageUUID>p1</pageUUID>", ""),
                    "url": "https://example.com",
                    "filing_date": date(2024, 1, 1),
                    "source_is_txt": False,
                    "source_is_html": True,
                }
            ]
        )
        xml_rows, failures = f_xml.generate_xml(df)
        self.assertEqual(failures, [])
        root = ET.fromstring(xml_rows[0].xml)
        section = root.find(".//section")
        assert section is not None
        self.assertEqual(section.get("title"), "3.7 Broker’s Fees.")
        # A .txt page is not evidence.
        df.loc[0, "source_is_html"] = False
        df.loc[0, "source_is_txt"] = True
        xml_rows, _ = f_xml.generate_xml(df)
        section = ET.fromstring(xml_rows[0].xml).find(".//section")
        assert section is not None
        self.assertEqual(section.get("title"), "3.7 Broker’ s Fees.")


class PreProcessingSourceTests(unittest.TestCase):
    """New ingests: the HTML formatter adds no space inside an ordinal or a possessive."""

    def test_formatter_keeps_ordinals_and_possessives_whole(self) -> None:
        self.assertEqual(format_content(BROKER_RAW, is_txt=False, is_html=True)[:20], "3.7 Broker’s Fees. N")
        text = format_content(ORDINAL_RAW, is_txt=False, is_html=True)
        self.assertIn("the 46th day", text)
        self.assertIn("22nd Floor", text)
        self.assertIn("December 31st of", text)
        self.assertEqual(
            format_content("<p><b>15</b><sup>th</sup> day</p>", is_txt=False, is_html=True), "15th day"
        )
        self.assertEqual(
            format_content("<p>BUYER<u>’</u><u>S GUARANTOR</u></p>", is_txt=False, is_html=True),
            "BUYER’S GUARANTOR",
        )

    def test_formatter_still_spaces_other_tag_boundaries(self) -> None:
        self.assertEqual(format_content("<p>Section 1.1<u>Defined Terms</u></p>", is_txt=False, is_html=True),
                         "Section 1.1 Defined Terms")
        self.assertEqual(format_content("<p>Section 5<u>Seller</u></p>", is_txt=False, is_html=True),
                         "Section 5 Seller")
        # "th" that opens a word is not an ordinal suffix.
        self.assertEqual(format_content("<p>Article 1<u>the Merger</u></p>", is_txt=False, is_html=True),
                         "Article 1 the Merger")
        # A suffix that does not fit the number is left as it was.
        self.assertEqual(format_content("<p>5<sup>st</sup></p>", is_txt=False, is_html=True), "5 st")


if __name__ == "__main__":
    unittest.main()
