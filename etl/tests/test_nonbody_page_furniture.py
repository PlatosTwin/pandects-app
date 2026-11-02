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

    def test_exhibit_table_of_contents_heading_stays(self) -> None:
        # f0d679c8: Exhibit G prints its own "Table of Contents" heading right under the back-link.
        removed = [RemovedFurnitureLine(page_index=i, line="Table of Contents", kind="toc_link") for i in range(3)]
        pages = [
            "Table of Contents\nTable of Contents\nPage\nArticle I DEFINITIONS G-5\nG-1",
            "Table of Contents\nARTICLE IV CAPITAL STOCK 8 4.1 Issuance of Stock 8 4.2 Transfers 9\nG-2",
            "Table of Contents\nSection 1.01. Definitions. As used in this Agreement:\nG-3",
        ]
        out = _strip(pages, [207, 208, 209], ["a\n<page>89</page>"], [93], removed)
        # Page 2 continues the table: its back-link is furniture.
        self.assertEqual([t.splitlines()[0] for t in out],
                         ["Table of Contents", "ARTICLE IV CAPITAL STOCK 8 4.1 Issuance of Stock 8 4.2 Transfers 9",
                          "Section 1.01. Definitions. As used in this Agreement:"])
        self.assertEqual(out[0].splitlines()[1], "Page")

    def test_back_link_above_a_page_number_goes(self) -> None:
        removed = [RemovedFurnitureLine(page_index=i, line="Table of Contents", kind="toc_link") for i in range(3)]
        pages = ["Table of Contents\n\nPage 91\n\nEXHIBIT A\nForm of Escrow Agreement",
                 "Table of Contents\n\nPage 3 of 40\n\nThe Escrow Agent shall hold the funds."]
        out = _strip(pages, [95, 99], ["a\n<page>89</page>", "b\n<page>90</page>"], [93, 94], removed)
        self.assertNotIn("Table of Contents", out[0])
        self.assertNotIn("Table of Contents", out[1])

    def test_continuation_page_with_numbered_entries_keeps_no_back_link(self) -> None:
        # a1c23200: "Page 5.6.5 Treasurer 15" is a column header, not a page number.
        removed = [RemovedFurnitureLine(page_index=i, line="Table of Contents", kind="toc_link") for i in range(3)]
        pages = ["Table of Contents\nTABLE OF CONTENTS\nPage ARTICLE 1 Identification 1\ni",
                 "Table of Contents\nPage 5.6.5 Treasurer 15 5.6.6 Assistant Secretaries 15\nii",
                 "Table of Contents\nPage Section 9.4 Seal 21 Section 9.5 Fiscal Year 21\niii"]
        out = _strip(pages, [95, 96, 97], ["a\n<page>89</page>"], [93], removed)
        self.assertEqual([t.splitlines()[0] for t in out],
                         ["TABLE OF CONTENTS", "Page 5.6.5 Treasurer 15 5.6.6 Assistant Secretaries 15",
                          "Page Section 9.4 Seal 21 Section 9.5 Fiscal Year 21"])

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


def _strip_typed(texts, orders, types, body_texts, body_orders, removed=None):
    return strip_nonbody_page_furniture(
        texts,
        orders,
        body_page_texts=body_texts,
        body_page_orders=body_orders,
        body_removed=removed or [],
        page_types=types,
    )


def _last_lines(out: list[str]) -> list[str]:
    return [t.splitlines()[-1] for t in out]


class ExhibitAndSigRunNumberTests(unittest.TestCase):
    """Page layouts from the 2026-11-01 audit of served sigPages and backMatter."""

    BODY = ["x\n<page>53</page>", "y\n<page>54</page>", "z\n<page>55</page>"]

    def test_labels_with_a_space_after_the_hyphen_chain(self) -> None:
        # 5db52097: Exhibit A prints "A- 2" .. "A- 8" (the body ends at 55).
        pages = ["EXHIBIT A\nDEFINED TERMS\n“Code” means the Internal Revenue Code of 1986, as amended.",
                 "or for which the Company has any direct or indirect liability.\nA- 2",
                 "shutdown or suspension of operations at any power plant.\nA- 3",
                 "“Good Utility Practice” means any of the practices.\nA- 4"]
        out = _strip(pages, [60, 61, 62, 63], self.BODY, [56, 57, 58])
        self.assertEqual(_last_lines(out)[1:], ["or for which the Company has any direct or indirect liability.",
                                                "shutdown or suspension of operations at any power plant.",
                                                "“Good Utility Practice” means any of the practices."])

    def test_label_on_the_only_page_of_an_exhibit(self) -> None:
        # 966adaf4 "B-1"; 1d34d08f "-1-" (Exhibit B is one page; Exhibit C's page 1 is unnumbered).
        pages = ["“WARN Act” means the Worker Adjustment and Retraining Notification Act of 1988.\nA-24",
                 "EXHIBIT B\nPERSONS ENTERING INTO VOTING AGREEMENTS\nRyan Nabil Finley\nB-1",
                 "EXHIBIT C\nAMENDED AND RESTATED\n(c) Any repeal or amendment of the foregoing provisions.\n-1-",
                 "EXHIBIT D\nAMENDED AND RESTATED BYLAWS\nSection 3. Special Meetings."]
        out = _strip(pages, [116, 117, 118, 119], self.BODY, [56, 57, 58])
        self.assertEqual(_last_lines(out)[1:], ["Ryan Nabil Finley",
                                                "(c) Any repeal or amendment of the foregoing provisions.",
                                                "Section 3. Special Meetings."])

    def test_exhibit_numbering_that_restarts_without_neighbors(self) -> None:
        # a4aef3cb: Annex A prints "2" on its second page only.
        pages = ["ANNEX A\nCONDITIONS TO THE OFFER\n(a) there shall have been instituted or be pending any action",
                 "(f) since the date of the Agreement there shall have occurred any change\n2",
                 "The foregoing conditions are for the sole benefit of Parent."]
        out = _strip(pages, [50, 51, 52], self.BODY, [46, 47, 48])
        self.assertEqual(_last_lines(out)[1], "(f) since the date of the Agreement there shall have occurred any change")

    def test_numbering_after_a_cover_sheet(self) -> None:
        # f1a55a5e: "Schedule 2" is a one-line cover; the letter behind it prints "- 1 -".
        pages = ["Schedule 2 Letter Agreement with Founders",
                 "Omaha Acquisition Corp.\nc/o Thomas H. Lee Partners, L.P.\nLetter from Omaha Acq. Corp to West\n- 1 -"]
        out = _strip(pages, [67, 68], self.BODY, [56, 57, 58])
        self.assertEqual(_last_lines(out), ["Schedule 2 Letter Agreement with Founders", "Letter from Omaha Acq. Corp to West"])

    def test_exhibit_heading_under_a_final_form_banner(self) -> None:
        # 37660870: "FINAL FORM" sits above "EXHIBIT B-1"; the second page prints "2".
        pages = ["FINAL FORM\nEXHIBIT B-1\nINSTRUMENT OF ASSUMPTION\n3 NOTE TO DRAFT: Use this bracketed text.",
                 "This Instrument may be signed in any number of counterparts.\n[Remainder of page intentionally left blank.]\n2"]
        out = _strip(pages, [144, 145], self.BODY, [56, 57, 58])
        self.assertEqual(_last_lines(out)[1], "[Remainder of page intentionally left blank.]")

    def test_a_number_that_does_not_match_its_place_stays(self) -> None:
        # The third page of an exhibit ending in "2" (a list or table entry) keeps it.
        pages = ["EXHIBIT A\nFORM OF JOINDER\nThe undersigned agrees as follows.",
                 "1. Joinder. The undersigned is a party.",
                 "Number of Shares held by the Holder:\n2"]
        out = _strip(pages, [80, 81, 82], self.BODY, [56, 57, 58])
        self.assertEqual(out, pages)

    def test_roman_numbers_on_an_exhibit_table_of_contents(self) -> None:
        # 44b10133: one TOC page "i"; e0533518: "i" .. "iii" on consecutive TOC pages.
        single = ["TABLE OF CONTENTS\nArticle I DEFINITIONS 1 1.1 Certain Definitions 1\ni"]
        self.assertEqual(_strip(single, [133], self.BODY, [56, 57, 58]),
                         ["TABLE OF CONTENTS\nArticle I DEFINITIONS 1 1.1 Certain Definitions 1"])
        run = ["ANNEX B\nAMENDED AND RESTATED PARTNERSHIP AGREEMENT\nB-1",
               "TABLE OF CONTENTS\nARTICLE I\nSection 6.1 Tax Allocations 15\ni",
               "Section 6.2 Preparation of Tax Returns 15\nSection 10.1 Initial Issuance of LP Certificates 25\nii",
               "Section 10.5 Prohibited Transfers 28\nSection 15.1 Amendments 36\niii"]
        out = _strip(run, [60, 61, 62, 63], self.BODY, [56, 57, 58])
        self.assertEqual(_last_lines(out), ["AMENDED AND RESTATED PARTNERSHIP AGREEMENT", "Section 6.1 Tax Allocations 15",
                                            "Section 10.1 Initial Issuance of LP Certificates 25",
                                            "Section 15.1 Amendments 36"])

    def test_sig_pages_numbered_on_their_own(self) -> None:
        # 6f611efe: the second sig page prints "2"; 83ded211: "S-1"; b5303a2f: "B-2".
        sigs = ["IN WITNESS WHEREOF, each of the undersigned has executed this Agreement.\nName: Erik J. Fagan\nTitle: President",
                "MOTOROLA SOLUTIONS, INC.\nBy: /s/ Michael D. Annes\nTitle: Senior Vice President\n2"]
        out = _strip_typed(sigs, [122, 123], ["sig", "sig"], self.BODY, [119, 120, 121])
        self.assertEqual(out[1], "MOTOROLA SOLUTIONS, INC.\nBy: /s/ Michael D. Annes\nTitle: Senior Vice President")
        [one] = _strip_typed(["BUYER\nBy: /s/ Peter W. Clifford\nS-1"], [76], ["sig"], self.BODY, [73, 74, 75])
        self.assertEqual(one, "BUYER\nBy: /s/ Peter W. Clifford")

    def test_signer_lines_that_look_like_numbers_stay(self) -> None:
        # A signer block whose last line is a share count or a title stays whole.
        sig = "HOLDER:\nBy: /s/ Jane Doe\nNumber of Shares:\n2,500"
        self.assertEqual(_strip_typed([sig], [80], ["sig"], self.BODY, [56, 57, 58]), [sig])
        sig = "HOLDER:\nBy: /s/ Jane Doe\nTitle: Manager\n3"
        self.assertEqual(_strip_typed([sig], [80], ["sig"], self.BODY, [56, 57, 58]), [sig])

    def test_a_footnote_marker_at_the_top_stays(self) -> None:
        # d196ed31: a footnote carried over to the next page opens with its marker "1".
        pages = ["EXHIBIT C\nSECOND AMENDED AND RESTATED LIMITED LIABILITY COMPANY AGREEMENT\nThis Agreement is made.",
                 "1\nNote to Form\n: All rollover sellers who acquire Units will execute this Agreement."]
        out = _strip(pages, [120, 121], self.BODY, [56, 57, 58])
        self.assertEqual(out[1].splitlines()[0], "1")

    def test_a_second_body_numbering_continues_on_the_sig_page(self) -> None:
        # 7ba39c9e: body pages carry a sheet counter at the top and the printed number at the bottom.
        body = ["<page>63</page>\nwaives and covenants\n<page>-58-</page>", "<page>64</page>\nthe Parties\n<page>-59-</page>"]
        sig = "65\nIN WITNESS WHEREOF, the parties hereto have executed this Agreement.\nName: Title:\n-60-"
        [out] = _strip([sig], [65], body, [63, 64])
        self.assertEqual(out, "IN WITNESS WHEREOF, the parties hereto have executed this Agreement.\nName: Title:")

    def test_a_number_under_a_document_id(self) -> None:
        # d7f2069e: each sig page ends "62" then "4892-0619-9513".
        sigs = [f"COMPANY:\nBy: /s/ Signer {n}\n{n}\n4892-0619-9513" for n in (62, 63, 64)]
        out = _strip_typed(sigs, [66, 67, 68], ["sig"] * 3, ["x\n<page>61</page>"], [65])
        self.assertEqual(out, [f"COMPANY:\nBy: /s/ Signer {n}" for n in (62, 63, 64)])

    def test_two_level_labels_chain(self) -> None:
        # df3adb62: Exhibit A-3 prints "A-3-1" .. "A-3-3".
        pages = ["EXHIBIT A-3\nMISCELLANEOUS CONTRACTS\n16. Consulting Agreement, dated April 20, 2000.\nA-3-1",
                 "17. Contract Service Agreement, dated October 22, 2001.\nA-3-2",
                 "42. Almedica Agreement\nA-3-3"]
        out = _strip(pages, [41, 42, 43], self.BODY, [35, 36, 37])
        self.assertEqual(_last_lines(out), ["16. Consulting Agreement, dated April 20, 2000.",
                                            "17. Contract Service Agreement, dated October 22, 2001.",
                                            "42. Almedica Agreement"])


class LawFirmFooterTests(unittest.TestCase):
    def test_body_footer_printed_nowhere_else_goes_on_sig_pages(self) -> None:
        # 82eab2ce: "CLIFFE DEKKER HOFMEYR" closes every body page and every sig page.
        body = [f"clause {i} text\nCLIFFE DEKKER HOFMEYR" for i in range(6)]
        removed = [RemovedFurnitureLine(page_index=i, line="CLIFFE DEKKER HOFMEYR", kind="repeated_line") for i in range(6)]
        sig = "SIGNED at Cape Town on 31 October 2021\nName of Signatory: Pieter Erasmus\nDesignation of Signatory: Director\nCLIFFE DEKKER HOFMEYR"
        [out] = _strip([sig], [80], body, list(range(10, 16)), removed)
        self.assertEqual(out, sig.rsplit("\n", 1)[0])

    def test_running_party_name_stays_on_sig_pages(self) -> None:
        # A party name repeated as a running line also appears in the preamble: never learned.
        body = ["This Agreement is made by ACME HOLDINGS INC. and Buyer.\nACME HOLDINGS INC."] + [
            f"clause {i}\nACME HOLDINGS INC." for i in range(5)]
        removed = [RemovedFurnitureLine(page_index=i, line="ACME HOLDINGS INC.", kind="repeated_line") for i in range(6)]
        sig = "IN WITNESS WHEREOF\nBy: /s/ Jane Doe\nTitle: CEO\nACME HOLDINGS INC."
        self.assertEqual(_strip([sig], [80], body, list(range(10, 16)), removed), [sig])

    def test_footer_only_on_the_edge_the_body_printed_it(self) -> None:
        body = [f"clause {i} text\nCLIFFE DEKKER HOFMEYR" for i in range(6)]
        removed = [RemovedFurnitureLine(page_index=i, line="CLIFFE DEKKER HOFMEYR", kind="repeated_line") for i in range(6)]
        page = "CLIFFE DEKKER HOFMEYR\nAnnexure A\nLeased Premises"
        self.assertEqual(_strip([page], [82], body, list(range(10, 16)), removed), [page])


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
