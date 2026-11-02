# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
"""Page numbers and body furniture on signature and back-matter pages.

The page edges in these tests are copied from served agreements whose sigPages
or backMatter kept printed page numbers or navigation links: 45b17e53 ("55"),
0ce8fcc0 ("- 97 -" .. "- 100 -"), fc335a3c ("-39-" .. "-44-"), 99fd5b55
("74", "A-1" .. "A-4"), bce24dbb ("Table of Contents", "2", "3").
"""
from datetime import date
import unittest
import xml.etree.ElementTree as ET

import pandas as pd

from etl.domain import f_xml
from etl.domain.page_furniture import RemovedFurnitureLine, strip_nonbody_page_furniture


def _strip(texts: list[str], orders: list[int], body_texts: list[str], body_orders: list[int], removed=None) -> list[str]:
    return strip_nonbody_page_furniture(
        texts,
        orders,
        body_page_texts=body_texts,
        body_page_orders=body_orders,
        body_removed=removed or [],
    )


SIG_45B17E53 = (
    "IN WITNESS WHEREOF, the parties hereto have caused this Agreement to be executed as of the date first written above.\n"
    "SELLER:\n"
    "By: /s/ Michael Sember\n"
    "Its: Governor\n"
    "55"
)


class StripNonbodyPageFurnitureTests(unittest.TestCase):
    def test_sig_page_number_that_continues_the_tagged_body_numbers(self) -> None:
        body = ["... the Closing.\n<page>53</page>", "Counterparts.\n<page>54</page>"]
        [sig] = _strip([SIG_45B17E53], [59], body, [57, 58])
        self.assertEqual(sig, SIG_45B17E53.rsplit("\n", 1)[0])

    def test_dashed_numbers_on_consecutive_sig_pages(self) -> None:
        body = ["text\n<page>- 95 -</page>", "text\n<page>- 96 -</page>"]
        sigs = [
            "IN WITNESS WHEREOF, the Parties have executed this Agreement.\nTitle: Authorized Signatory\n- 97 -",
            "HMT INC. EDDYFI HOLDING INC.\nTitle: Authorized Signatory Title: Trustee\n- 98 -",
            "/s/ Andre Levesque\nTitle: President and Chief Executive Officer\n- 99 -",
        ]
        out = _strip(sigs, [100, 101, 102], body, [98, 99])
        self.assertTrue(all(not text.endswith("-") for text in out), out)
        self.assertEqual(out[1], "HMT INC. EDDYFI HOLDING INC.\nTitle: Authorized Signatory Title: Trustee")

    def test_leading_numbers_on_sig_and_back_pages_of_a_txt_filing(self) -> None:
        # fc335a3c: the body's own numbers were bare lines the body rules removed.
        removed = [RemovedFurnitureLine(page_index=0, line="-37-", kind="page_number"),
                   RemovedFurnitureLine(page_index=1, line="-38-", kind="page_number")]
        body = ["-37-\nTelecopier No.: (514) 937-2554", "-38-\n[INTENTIONALLY LEFT BLANK]"]
        pages = ["-39-\nIN WITNESS WHEREOF the parties hereto have duly executed this Agreement.",
                 "-40-\nSCHEDULE A", "-41-\n(b) those government or regulatory approvals"]
        out = _strip(pages, [38, 39, 40], body, [36, 37], removed)
        self.assertEqual(out, ["IN WITNESS WHEREOF the parties hereto have duly executed this Agreement.",
                               "SCHEDULE A", "(b) those government or regulatory approvals"])

    def test_exhibit_page_numbers_that_restart(self) -> None:
        pages = [
            "EXHIBIT A\nCERTIFICATE OF MERGER\nSIXTH: That the executed Merger Agreement is on file.\nA-1",
            "SEVENTH: That a copy of the Merger Agreement will be furnished.\nA-2",
            "(e) No Merger Consideration shall be paid.\nA-3",
        ]
        out = _strip(pages, [84, 85, 86], ["x\n<page>73</page>"], [82])
        self.assertEqual([t.splitlines()[-1] for t in out],
                         ["SIXTH: That the executed Merger Agreement is on file.",
                          "SEVENTH: That a copy of the Merger Agreement will be furnished.",
                          "(e) No Merger Consideration shall be paid."])

    def test_body_toc_link_and_chained_numbers_in_back_matter(self) -> None:
        removed = [RemovedFurnitureLine(page_index=i, line="Table of Contents", kind="toc_link") for i in range(3)]
        pages = [
            "Table of Contents\nEXHIBIT F\nClosing Working Capital",
            "Table of Contents\nAccounts Payable: To be determined.\n2",
            "Table of Contents\nNote #2- Prepaid expenses.\n3",
        ]
        out = _strip(pages, [101, 102, 103], ["a\n<page>89</page>"], [93], removed)
        self.assertEqual(out, ["EXHIBIT F\nClosing Working Capital", "Accounts Payable: To be determined.",
                               "Note #2- Prepaid expenses."])

    def test_body_running_title_on_a_sig_page(self) -> None:
        # 6fd9c8c2: the body's running footer "Share Purchase Agreement" closes the sig page too.
        removed = [RemovedFurnitureLine(page_index=i, line="Share Purchase Agreement", kind="running_title") for i in range(5)]
        sig = "BAYTEX ENERGY TRUST, by its Administrator, BAYTEX ENERGY LTD.\nBy: (signed) “Raymond T. Chan”\nShare Purchase Agreement"
        [out] = _strip([sig], [89], ["x"], [88], removed)
        self.assertEqual(out, "BAYTEX ENERGY TRUST, by its Administrator, BAYTEX ENERGY LTD.\nBy: (signed) “Raymond T. Chan”")

    def test_numbers_that_fit_nothing_stay(self) -> None:
        # A lone "2" on one exhibit page with no neighbor in sequence, far from the body numbers.
        pages = ["EXHIBIT B\nForm of Transition Services Agreement\n2", "EXHIBIT C\nForm of Supply Agreement"]
        out = _strip(pages, [97, 98], ["a\n<page>89</page>"], [93])
        self.assertEqual(out, pages)

    def test_signer_text_and_repeated_body_lines_stay(self) -> None:
        # A repeated body header ("ACME CORP") is not learned furniture on sig pages.
        removed = [RemovedFurnitureLine(page_index=i, line="ACME CORP", kind="repeated_line") for i in range(5)]
        sig = "ACME CORP\nBy: /s/ Jane Doe\nName: Jane Doe\nTitle: 54"
        out = _strip([sig], [59], ["x\n<page>54</page>"], [58], removed)
        self.assertEqual(out, [sig])

    def test_numbers_off_the_page_edge_stay(self) -> None:
        sig = "IN WITNESS WHEREOF\n55\nBy: /s/ Jane Doe\nName: Jane Doe"
        self.assertEqual(_strip([sig], [59], ["x\n<page>54</page>"], [58]), [sig])

    def test_a_page_that_is_only_a_number_is_kept(self) -> None:
        self.assertEqual(_strip(["55"], [59], ["x\n<page>54</page>"], [58]), ["55"])


class GenerateXmlNonbodyFurnitureTests(unittest.TestCase):
    def test_sig_pages_lose_page_numbers_in_the_served_xml(self) -> None:
        agreement_uuid = "33333333-3333-3333-3333-333333333333"
        pages = [
            ("body", "<article>ARTICLE I GENERAL</article>\n<section>1.1 Notices.</section>\nAll notices.\n<page>53</page>"),
            ("body", "<section>1.2 Counterparts.</section>\nIn counterparts.\n<page>54</page>"),
            ("sig", SIG_45B17E53),
        ]
        df = pd.DataFrame(
            [
                {
                    "agreement_uuid": agreement_uuid,
                    "page_uuid": f"p-{i}",
                    "page_order": 57 + i,
                    "raw_page_content": "",
                    "source_page_type": page_type,
                    "gold_label": None,
                    "tagged_output": text,
                    "url": "https://example.com/agreement",
                    "filing_date": date(2024, 1, 1),
                    "source_is_txt": True,
                    "source_is_html": False,
                }
                for i, (page_type, text) in enumerate(pages)
            ]
        )
        xml_rows, failures = f_xml.generate_xml(df)
        self.assertEqual(failures, [])
        root = ET.fromstring(xml_rows[0].xml)
        sig = root.find("sigPages")
        assert sig is not None
        texts = [t.text for t in sig.iter("text")]
        self.assertEqual(texts[-1], "Its: Governor")
        self.assertNotIn("55", texts)

    def test_html_sig_page_reformatted_from_raw_html_loses_its_page_number(self) -> None:
        raw_html = (
            "<div><font>IN WITNESS WHEREOF, the parties have executed this Agreement.</font></div>"
            "<div><font>By: /s/ Michael Sember</font></div><div><font>Its: Governor</font></div>"
            "<div style=\"text-align:center\"><font>55</font></div>"
        )
        row = {"source_page_type": "sig", "source_is_html": True, "raw_page_content": raw_html,
               "tagged_output": "", "source_is_txt": False, "page_order": 59}
        text = f_xml._xml_page_text(row)
        self.assertTrue(text.endswith("55"))
        [out] = _strip([text], [59], ["x\n<page>54</page>"], [58])
        self.assertTrue(out.endswith("Its: Governor"), out)


if __name__ == "__main__":
    unittest.main()
