# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
"""A heading's own period left outside the heading tag opened the section text.

The tagged text in these tests is copied from served agreement
3c4efa67-9a46-57d0-8bff-60f856e3e856 (page_order 69 and 70), where 113 of 136
sections started with ". ", for example 7.02 ". (39) From the date of this
Agreement".
"""
from datetime import date
import unittest
import xml.etree.ElementTree as ET

from etl.domain import f_xml

PAGE_69_TAIL = (
    "(38) enter into any commitment with respect to, or otherwise agree or commit to, any of the foregoing.\n\n"
    "<section>7.02 Access to Information</section>\n\n.\n\n<page>69</page><pageUUID>p69</pageUUID>\n"
)
PAGE_70_HEAD = (
    "(39) From the date of this Agreement until the Closing Date, the Seller shall, and shall cause the Company, "
    "the Company Subsidiary and, with respect to the Reinsured Business, ELIC, to give the Buyer and its "
    "authorized Representatives\n<pageUUID>p70</pageUUID>"
)


def _section_texts(tagged_text: str) -> dict[str, list[str]]:
    root = ET.fromstring(
        f_xml.convert_to_xml(tagged_text, "11111111-1111-1111-1111-111111111111", date(2024, 1, 1), "https://example.com", "html")
    )
    return {s.get("title") or "": [t.text or "" for t in s.iter("text")] for s in root.iter("section")}


class OrphanHeadingPeriodTests(unittest.TestCase):
    def test_lone_period_line_after_a_section_heading_is_dropped(self) -> None:
        texts = _section_texts("<article>ARTICLE VII</article>\n" + PAGE_69_TAIL + PAGE_70_HEAD)
        self.assertEqual(texts["7.02 Access to Information"][0][:24], "(39) From the date of th")

    def test_period_on_the_heading_line_is_dropped(self) -> None:
        texts = _section_texts("<section>9.1 Termination</section>. This Agreement may be terminated at any time.")
        self.assertEqual(texts["9.1 Termination"], ["This Agreement may be terminated at any time."])

    def test_period_after_a_page_break_is_dropped_and_the_markers_stay(self) -> None:
        # 45184584: the heading ends page 61 and its period opens page 62.
        tagged = (
            "<section>Section 6.18. Post-Closing Assurances</section><pageUUID>p61</pageUUID>\n"
            ". After the Closing, each Party shall execute and deliver, upon the reasonable request of the other Party, "
            "any and all further instruments or documents."
        )
        root = ET.fromstring(
            f_xml.convert_to_xml(tagged, "11111111-1111-1111-1111-111111111111", date(2024, 1, 1), "https://example.com", "html")
        )
        section = root.find(".//section")
        assert section is not None
        self.assertEqual([child.tag for child in section], ["pageUUID", "text"])
        self.assertTrue((section.findtext("text") or "").startswith("After the Closing, each Party"))

    def test_period_glued_to_the_first_enumerator_is_dropped(self) -> None:
        # 1d6c6132
        texts = _section_texts(
            "<section>6.13 Tax Matters</section>\n\n.(a) The Shareholders’ Agent, the Company Shareholders and Parent "
            "shall cooperate fully."
        )
        self.assertEqual(texts["6.13 Tax Matters"][0][:30], "(a) The Shareholders’ Agent, t")

    def test_text_that_only_starts_with_a_period_like_mark_is_kept(self) -> None:
        cases = {
            "<section>1.1 Rate</section>.5% per annum applies.": ".5% per annum applies.",
            "<section>1.2 Omitted</section>... (intentionally omitted)": "... (intentionally omitted)",
        }
        for tagged, first in cases.items():
            with self.subTest(tagged=tagged):
                [texts] = _section_texts(tagged).values()
                self.assertEqual(texts[0], first)

    def test_unit_helper(self) -> None:
        self.assertEqual(f_xml._drop_orphan_heading_period("."), "")
        self.assertEqual(f_xml._drop_orphan_heading_period(".\n\n<page>69</page>"), "\n\n<page>69</page>")
        self.assertEqual(f_xml._drop_orphan_heading_period(". From the date"), "From the date")
        self.assertEqual(f_xml._drop_orphan_heading_period("From the date."), "From the date.")


if __name__ == "__main__":
    unittest.main()
