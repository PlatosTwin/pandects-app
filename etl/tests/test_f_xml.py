# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
from datetime import date
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import pandas as pd

from etl.domain import f_xml


class XMLGenerationTests(unittest.TestCase):
    def test_format_toc_html_like_screen_preserves_table_rows(self) -> None:
        raw_html = """
        <div><p>TABLE OF CONTENTS Page</p></div>
        <table>
          <tr><td>ARTICLE I THE MERGER</td><td>6</td></tr>
          <tr><td>1.1</td><td>The Merger</td><td>6</td></tr>
          <tr><td>1.2</td><td>The Closing</td><td>6</td></tr>
        </table>
        """

        formatted = f_xml.format_toc_html_like_screen(raw_html, line_width=80)

        self.assertIn("TABLE OF CONTENTS\n\nPage", formatted)
        self.assertIn("ARTICLE I THE MERGER", formatted)
        self.assertIn("1.1     The Merger", formatted)
        self.assertIn("1.2     The Closing", formatted)
        self.assertNotIn("ARTICLE I THE MERGER 6 1.1", formatted)

    def test_format_toc_html_like_screen_renders_rows_nested_in_unclosed_cell_once(
        self,
    ) -> None:
        # html.parser nests the rows after an unclosed <td> inside that cell.
        raw_html = """
        <table>
          <tr><td>ARTICLE I THE MERGER</td><td>6</td></tr>
          <tr><td>1.1</td><td>The Merger
          <tr><td>1.2</td><td>The Closing</td><td>7</td></tr>
          <tr><td>1.3</td><td>Effective Time</td><td>8</td></tr>
          </td></tr>
        </table>
        """

        formatted = f_xml.format_toc_html_like_screen(raw_html, line_width=80)

        self.assertEqual(formatted.count("The Closing"), 1)
        self.assertEqual(formatted.count("Effective Time"), 1)
        self.assertEqual(
            [line.split() for line in formatted.splitlines()],
            [
                ["ARTICLE", "I", "THE", "MERGER", "6"],
                ["1.1", "The", "Merger"],
                ["1.2", "The", "Closing", "7"],
                ["1.3", "Effective", "Time", "8"],
            ],
        )

    def test_format_toc_html_like_screen_ignores_nested_hidden_nodes(self) -> None:
        raw_html = """
        <div style="display: none"><span>hidden section</span></div>
        <table>
          <tr><td>1.1</td><td>The Merger</td><td>6</td></tr>
        </table>
        """

        formatted = f_xml.format_toc_html_like_screen(raw_html, line_width=80)

        self.assertIn("1.1     The Merger", formatted)
        self.assertNotIn("hidden section", formatted)

    def test_generate_xml_reformats_html_toc_raw_page_content(self) -> None:
        agreement_uuid = "11111111-1111-1111-1111-111111111111"
        raw_toc_html = """
        <div><p>TABLE OF CONTENTS Page</p></div>
        <table>
          <tr><td>ARTICLE I THE MERGER</td><td>6</td></tr>
          <tr><td>1.1</td><td>The Merger</td><td>6</td></tr>
          <tr><td>1.2</td><td>The Closing</td><td>6</td></tr>
          <tr><td>Exhibit A\x0e</td><td>7</td></tr>
        </table>
        """
        df = pd.DataFrame(
            [
                {
                    "agreement_uuid": agreement_uuid,
                    "page_uuid": "toc-page",
                    "page_order": 1,
                    "raw_page_content": raw_toc_html,
                    "source_page_type": "toc",
                    "gold_label": None,
                    "tagged_output": "TABLE OF CONTENTS Page ARTICLE I THE MERGER 6 1.1 The Merger 6 1.2 The Closing 6",
                    "url": "https://example.com/agreement",
                    "filing_date": date(2024, 1, 1),
                    "source_is_txt": False,
                    "source_is_html": True,
                },
                {
                    "agreement_uuid": agreement_uuid,
                    "page_uuid": "body-page",
                    "page_order": 2,
                    "raw_page_content": "<p>ARTICLE I</p>",
                    "source_page_type": "body",
                    "gold_label": None,
                    "tagged_output": "<article>ARTICLE I</article>Body text",
                    "url": "https://example.com/agreement",
                    "filing_date": date(2024, 1, 1),
                    "source_is_txt": False,
                    "source_is_html": True,
                },
            ]
        )

        generated, failures = f_xml.generate_xml(df)

        self.assertEqual(failures, [])
        self.assertEqual(len(generated), 1)
        root = ET.fromstring(generated[0].xml)
        toc = root.find("tableOfContents")
        self.assertIsNotNone(toc)
        assert toc is not None
        toc_texts = [el.text or "" for el in toc.findall("text")]
        self.assertIn("TABLE OF CONTENTS", toc_texts)
        self.assertIn("Page", toc_texts)
        self.assertTrue(any(text.startswith("1.1     The Merger") for text in toc_texts))
        self.assertTrue(any(text.startswith("1.2     The Closing") for text in toc_texts))
        self.assertTrue(any(text.startswith("Exhibit A") for text in toc_texts))
        self.assertNotIn("\x0e", generated[0].xml)
        self.assertNotIn(
            "TABLE OF CONTENTS Page ARTICLE I THE MERGER 6 1.1 The Merger 6 1.2 The Closing 6",
            toc_texts,
        )

    def test_generate_xml_uses_gold_label_for_toc_reformatting(self) -> None:
        agreement_uuid = "11111111-1111-1111-1111-111111111111"
        df = pd.DataFrame(
            [
                {
                    "agreement_uuid": agreement_uuid,
                    "page_uuid": "toc-page",
                    "page_order": 1,
                    "raw_page_content": """
                    <table>
                      <tr><td>1.1</td><td>The Merger</td><td>6</td></tr>
                    </table>
                    """,
                    "source_page_type": "body",
                    "gold_label": "toc",
                    "tagged_output": "1.1 The Merger 6",
                    "url": "https://example.com/agreement",
                    "filing_date": date(2024, 1, 1),
                    "source_is_txt": False,
                    "source_is_html": True,
                },
            ]
        )

        generated, failures = f_xml.generate_xml(df)

        self.assertEqual(failures, [])
        self.assertEqual(len(generated), 1)
        root = ET.fromstring(generated[0].xml)
        toc = root.find("tableOfContents")
        self.assertIsNotNone(toc)
        assert toc is not None
        toc_texts = [el.text or "" for el in toc.findall("text")]
        self.assertTrue(any(text.startswith("1.1     The Merger") for text in toc_texts))

    def test_format_signature_text_like_screen_joins_field_values(self) -> None:
        rendered_signature = """
        STANLEY BLACK & DECKER, INC.

        By:
         /s/ Donald Allan, Jr.

        Name:
        Donald Allan, Jr.

        Title:
        Senior Vice President and Chief Financial Officer

        SPECTRUM BRANDS,
        INC.

        by
        /s/ Nathan Fagre
        """

        formatted = f_xml.format_signature_text_like_screen(rendered_signature)

        self.assertIn("By: /s/ Donald Allan, Jr.", formatted)
        self.assertIn("Name: Donald Allan, Jr.", formatted)
        self.assertIn(
            "Title: Senior Vice President and Chief Financial Officer",
            formatted,
        )
        self.assertIn("SPECTRUM BRANDS, INC.", formatted)
        # The source prints a lowercase "by" with no colon; it is served as printed.
        self.assertIn("by /s/ Nathan Fagre", formatted)
        self.assertNotIn("By:\n\n/s/ Donald Allan, Jr.", formatted)

    def test_format_signature_text_like_screen_preserves_blank_form_fields(self) -> None:
        rendered_signature = """
        SUBSCRIBER:

        By:

        Name:
        Title:

        (Please print. Please indicate name and capacity of person signing above)
        """

        formatted = f_xml.format_signature_text_like_screen(rendered_signature)

        self.assertIn("By:\n\nName:\nTitle:", formatted)
        self.assertIn(
            "(Please print. Please indicate name and capacity of person signing above)",
            formatted,
        )
        self.assertNotIn(
            "Title: (Please print. Please indicate name and capacity of person signing above)",
            formatted,
        )

    def test_format_signature_html_like_screen_preserves_table_rows(self) -> None:
        raw_html = """
        <div><font>In Witness Whereof, the parties hereto have entered into and signed this Agreement.</font></div>
        <div>
          <table>
            <tr><td></td><td>PARENT:</td></tr>
            <tr><td></td><td>HawkEye 360, Inc.</td></tr>
            <tr style="height:14pt"><td></td><td></td></tr>
            <tr><td></td><td>By:</td><td>/s/ John Serafini</td></tr>
            <tr><td></td><td>Name:</td><td>John Serafini</td></tr>
            <tr><td></td><td>Its:</td><td>Chief Executive Officer</td></tr>
            <tr style="height:14pt"><td></td><td></td></tr>
            <tr><td></td><td>MERGER SUB:</td></tr>
            <tr><td></td><td>Forestal Merger Sub, Inc.</td></tr>
            <tr><td></td><td>By:</td><td>/s/ Todd Probert</td></tr>
            <tr><td></td><td>Name:</td><td>Todd Probert</td></tr>
            <tr><td></td><td>Its:</td><td>President</td></tr>
          </table>
        </div>
        <div><font>[SIGNATURE PAGE TO AGREEMENT AND PLAN OF MERGER]</font></div>
        """

        formatted = f_xml.format_signature_html_like_screen(raw_html)

        self.assertIn("PARENT:", formatted)
        self.assertIn("HawkEye 360, Inc.", formatted)
        self.assertIn(
            "In Witness Whereof, the parties hereto have entered into and signed this Agreement.",
            formatted,
        )
        self.assertNotIn("Whereof ,", formatted)
        self.assertIn("By: /s/ John Serafini", formatted)
        self.assertIn("Name: John Serafini", formatted)
        self.assertIn("Its: Chief Executive Officer", formatted)
        self.assertIn("MERGER SUB:", formatted)
        self.assertIn("Forestal Merger Sub, Inc.", formatted)
        self.assertIn("By: /s/ Todd Probert", formatted)
        self.assertIn("Its: President", formatted)
        self.assertNotIn(
            "PARENT: HawkEye 360, Inc. By: /s/ John Serafini Name: John Serafini",
            formatted,
        )

    def test_format_signature_html_like_screen_renders_rows_nested_in_unclosed_cell_once(
        self,
    ) -> None:
        # html.parser nests the rows after an unclosed <td> inside that cell.
        raw_html = """
        <table>
          <tr><td></td><td>PUBCO</td></tr>
          <tr><td></td><td colspan="2">[___]<p></p>
          <tr><td>By:</td><td>/s/ Jane Doe</td></tr>
          <tr><td>Name:</td><td>Jane Doe</td></tr>
          </td></tr>
        </table>
        """

        lines = [
            line for line in f_xml.format_signature_html_like_screen(raw_html).splitlines() if line
        ]

        self.assertEqual(lines, ["PUBCO", "[___]", "By: /s/ Jane Doe", "Name: Jane Doe"])

    def test_format_signature_html_like_screen_keeps_rows_after_unclosed_row(self) -> None:
        raw_html = """
        <table>
          <tr><td>ACME CORP.</td>
          <tr><td>By:</td><td>/s/ Jane Doe</td></tr>
          </tr>
        </table>
        """

        lines = [
            line for line in f_xml.format_signature_html_like_screen(raw_html).splitlines() if line
        ]

        self.assertEqual(lines, ["ACME CORP.", "By: /s/ Jane Doe"])

    def test_format_signature_html_like_screen_normalizes_bracket_spacing(self) -> None:
        raw_html = """
        <div><font>( Signature Page )</font></div>
        <div><font>[ SIGNATURE PAGE TO AGREEMENT AND PLAN OF MERGER ]</font></div>
        """

        formatted = f_xml.format_signature_html_like_screen(raw_html)

        self.assertIn("(Signature Page)", formatted)
        self.assertIn("[SIGNATURE PAGE TO AGREEMENT AND PLAN OF MERGER]", formatted)
        self.assertNotIn("( Signature Page )", formatted)
        self.assertNotIn("[ SIGNATURE PAGE TO AGREEMENT AND PLAN OF MERGER ]", formatted)

    def test_format_signature_html_like_screen_keeps_text_before_table_in_unclosed_paragraph(
        self,
    ) -> None:
        # EDGAR leaves the <p> unclosed, so html.parser nests the tables inside
        # it and the opening sentence is loose text in that paragraph.
        raw_html = """
        <div>
        <p align="left">IN WITNESS WHEREOF the parties have hereunto caused this Agreement
        to be duly executed as of the date first above written.
        <table>
          <tr><td></td><td colspan="3">ENCORE MEDICAL CORPORATION<br/></td></tr>
          <tr><td></td><td>By:</td><td>/s/ Harry L. Zimmerman</td></tr>
          <tr><td></td><td colspan="3">Name: Harry L. Zimmerman</td></tr>
        </table>
        <table>
          <tr><td></td><td colspan="3">EMPI, INC.<br/></td></tr>
          <tr><td></td><td>By:</td><td>/s/ H. Philip Vierling</td></tr>
        </table>
        </div>
        """

        lines = f_xml.format_signature_html_like_screen(raw_html).splitlines()

        self.assertEqual(
            lines[0],
            (
                "IN WITNESS WHEREOF the parties have hereunto caused this Agreement "
                "to be duly executed as of the date first above written."
            ),
        )
        self.assertIn("ENCORE MEDICAL CORPORATION", lines)
        self.assertIn("By: /s/ Harry L. Zimmerman", lines)
        self.assertIn("EMPI, INC.", lines)
        self.assertIn("By: /s/ H. Philip Vierling", lines)

    def test_format_signature_html_like_screen_keeps_loose_text_between_tables(self) -> None:
        raw_html = """
        <p>
        <table><tr><td>ACME CORP.</td></tr><tr><td>By:</td><td>/s/ Jane Doe</td></tr></table>
        The undersigned <b>Principal Shareholders</b> are executing this Agreement solely for Section 14.15.
        <table><tr><td>MPI HOLDINGS, LLC</td></tr></table>
        """

        lines = f_xml.format_signature_html_like_screen(raw_html).splitlines()

        self.assertEqual(
            [line for line in lines if line],
            [
                "ACME CORP.",
                "By: /s/ Jane Doe",
                "The undersigned Principal Shareholders are executing this Agreement solely for Section 14.15.",
                "MPI HOLDINGS, LLC",
            ],
        )

    def test_format_signature_html_like_screen_joins_small_caps(self) -> None:
        raw_html = """
        <p>I<small>N</small> W<small>ITNESS</small> W<small>HEREOF</small>, the Parties
        hereto have executed this Agreement.</p>
        <div><table>
          <tr><td colspan="3"><p><b>B<small>UYER</small></b></p><p>Thor Industries, Inc.</p></td></tr>
          <tr><td>By:</td><td></td><td>/s/ Robert W. Martin</td></tr>
          <tr><td>P<small>RINTED</small>&#160;N<small>AME</small>:</td><td></td>
              <td>R<small>OBERT</small> W. M<small>ARTIN</small></td></tr>
          <tr><td><font>T</font><font style="font-size:9pt">ITLE</font>:</td><td></td><td>CEO</td></tr>
          <tr><td>Na<font style="DISPLAY: inline">me:   Jordan Hitch</font></td></tr>
        </table></div>
        """

        formatted = f_xml.format_signature_html_like_screen(raw_html)

        self.assertIn(
            "IN WITNESS WHEREOF, the Parties hereto have executed this Agreement.",
            formatted,
        )
        self.assertIn("BUYER Thor Industries, Inc.", formatted)
        self.assertIn("PRINTED NAME: ROBERT W. MARTIN", formatted)
        self.assertIn("TITLE: CEO", formatted)
        self.assertIn("Name: Jordan Hitch", formatted)
        self.assertNotRegex(formatted, r"\b[B-Z] [A-Z]{2,}")

    def test_format_signature_html_like_screen_keeps_labels_as_printed(self) -> None:
        # 65406fe5 (page d03a2622): the label cell prints "By" with no colon.
        # 8717e0e7 (page f9f7cc04): a schedule table's "Name" header is no label.
        raw_html = """
        <table><tr>
        <td width="3%"><p><font face="Times New Roman" size="2">By</font></p> </td>
        <td style="border-bottom:solid windowtext 1.0pt;" width="46%">
        <p><font face="Times New Roman" size="2">/s/   Geoffrey T. Martin</font></p> </td> </tr>
        </table>
        <table><tr bgcolor="#cceeff">
        <td><div><font style="DISPLAY: inline">Name</font></div></td>
        <td><font style="DISPLAY: inline">\u00a0</font></td>
        <td colspan="2"><div><font style="DISPLAY: inline">FDVE shares</font></div></td>
        </tr></table>
        """

        lines = [line for line in f_xml.format_signature_html_like_screen(raw_html).splitlines() if line]

        self.assertEqual(lines, ["By /s/ Geoffrey T. Martin", "Name FDVE shares"])

    def test_label_cell_without_colon_still_ends_an_empty_field(self) -> None:
        # 60074974: an empty "Its:" row is followed by a "By" | "/s/ ..." row; the
        # By row must stay its own field, not become the value of "Its".
        raw_html = """
        <table>
        <tr><td>Its:</td><td></td></tr>
        <tr><td>By</td><td>/s/ i.v. Peter Nussbaum</td></tr>
        <tr><td>Name</td><td>Jane Doe</td></tr>
        </table>
        <table>
        <tr><td>by</td><td></td></tr>
        <tr><td>/s/ George Kotlarz</td><td></td></tr>
        </table>
        """

        lines = [line for line in f_xml.format_signature_html_like_screen(raw_html).splitlines() if line]

        # 04db8f82: a "by" cell alone on its row still takes the "/s/" row below.
        self.assertEqual(
            lines, ["Its:", "By /s/ i.v. Peter Nussbaum", "Name Jane Doe", "by /s/ George Kotlarz"]
        )

    def test_format_signature_html_like_screen_ignores_doctype_and_comments(self) -> None:
        raw_html = """<!DOCTYPE html><!-- page 54 -->
        <p>IN WITNESS WHEREOF, the parties have signed.<!-- hidden -->
        <table><tr><td>ACME CORP.</td></tr></table>
        """

        formatted = f_xml.format_signature_html_like_screen(raw_html)

        self.assertEqual(
            [line for line in formatted.splitlines() if line],
            ["IN WITNESS WHEREOF, the parties have signed.", "ACME CORP."],
        )

    def test_format_signature_html_like_screen_keeps_gap_between_adjacent_inline_fields(
        self,
    ) -> None:
        raw_html = """
        <p><font style="padding-right:107pt"><font>By:</font></font><font><font>/s/ Steven D. Barnhart</font></font></p>
        <p><font>Name:</font><font>Steven D. Barnhart</font></p>
        <p><font style="display:inline-block;width:36pt">By:</font>Mason Wells Buyout Partners IV, LLC</p>
        <p>DiModica<u>__________</u></p>
        """

        lines = [
            line
            for line in f_xml.format_signature_html_like_screen(raw_html).splitlines()
            if line
        ]

        self.assertEqual(
            lines,
            [
                "By: /s/ Steven D. Barnhart",
                "Name: Steven D. Barnhart",
                "By: Mason Wells Buyout Partners IV, LLC",
                "DiModica __________",
            ],
        )

    def test_format_signature_html_like_screen_still_separates_block_elements(self) -> None:
        raw_html = """
        <table><tr>
          <td><p>ACME CORP.</p><p>By:</p></td>
          <td><div>/s/ Jane Doe</div><br/><div>Jane Doe</div></td>
        </tr></table>
        """

        formatted = f_xml.format_signature_html_like_screen(raw_html)

        self.assertIn("ACME CORP. By: /s/ Jane Doe Jane Doe", formatted)

    def test_format_signature_html_like_screen_drops_back_navigation_links(self) -> None:
        raw_html = """
        <div style="text-align:center"><a href="#toc">BACK</a></div>
        <div><a href="#TOC">Back
        </a></div>
        <p><a href="#top">Return to top</a></p>
        <p>IN WITNESS WHEREOF, the parties have signed.</p>
        <table><tr><td>ACME CORP.</td></tr><tr><td>By:</td><td>/s/ Jane Doe</td></tr></table>
        """

        lines = [
            line
            for line in f_xml.format_signature_html_like_screen(raw_html).splitlines()
            if line
        ]

        self.assertEqual(
            lines,
            ["IN WITNESS WHEREOF, the parties have signed.", "ACME CORP.", "By: /s/ Jane Doe"],
        )

    def test_format_signature_html_like_screen_keeps_other_back_links(self) -> None:
        # Only an internal anchor whose sole text is the navigation label is dropped.
        raw_html = """
        <p><a href="https://www.sec.gov/">BACK</a></p>
        <p><a href="#s1">Back Bay Capital LLC</a></p>
        <p><a>BACK</a></p>
        """

        lines = [
            line
            for line in f_xml.format_signature_html_like_screen(raw_html).splitlines()
            if line
        ]

        self.assertEqual(lines, ["BACK", "Back Bay Capital LLC", "BACK"])

    def test_format_signature_html_like_screen_joins_letters_across_unstyled_float_none(
        self,
    ) -> None:
        # 98b575c0: the "ç" sits in a span styled "display: inline ! important; float: none".
        raw_html = """
        <table><tr>
          <td><div>By:</div></td>
          <td>/s/ <font>F<font style="font-size: 10pt;">ran<font style="color: rgb(0, 0, 0);
            word-spacing: 0px; display: inline ! important; float: none;">çois Locoh-Donou<br/>
          </font></font></font></td>
        </tr><tr>
          <td><div>Name:</div></td>
          <td><font>F<font style="font-size: 10pt;">ran<font style="display: inline ! important;
            float: none;">çois L</font>ocoh-Donou</font></font></td>
        </tr></table>
        <p><font style="float: left">By:</font>Jane Doe</p>
        """

        lines = [
            line
            for line in f_xml.format_signature_html_like_screen(raw_html).splitlines()
            if line
        ]

        self.assertEqual(
            lines,
            [
                "By: /s/ François Locoh-Donou",
                "Name: François Locoh-Donou",
                "By: Jane Doe",
            ],
        )

    def test_format_signature_text_like_screen_keeps_by_empty_before_next_signer_block(
        self,
    ) -> None:
        # 0904d542: each empty "By:" is followed by the next signer entity and its own "By:".
        text = "\n\n".join(
            [
                "SUN MICROSYSTEMS, INC.",
                "By:",
                "STANFORD ACQUISITION CORPORATION",
                "By:",
                "STORAGE TECHNOLOGY CORPORATION",
                "By:",
                "Patrick J. Martin",
                "PURCHASER:",
                "RYERSON TULL, INC.",
                "By:",
                "51",
            ]
        )

        lines = [
            line for line in f_xml.format_signature_text_like_screen(text).splitlines() if line
        ]

        self.assertEqual(
            lines,
            [
                "SUN MICROSYSTEMS, INC.",
                "By:",
                "STANFORD ACQUISITION CORPORATION",
                "By:",
                "STORAGE TECHNOLOGY CORPORATION",
                "By: Patrick J. Martin",
                "PURCHASER:",
                "RYERSON TULL, INC.",
                "By:",
                "51",
            ],
        )

    def test_format_signature_text_like_screen_keeps_empty_title_before_sentence_or_entity(
        self,
    ) -> None:
        # c5ac8b6f: an empty Title: was followed by a sentence or the next entity.
        sentence = (
            "The following executes this Agreement as the Management Representative "
            "solely to obtain the benefit of the indemnities provided to him in Section 15.23."
        )
        entity = (
            "EDGESTONE CAPITAL EQUITY PARTNERS, INC., as general partner or agent for "
            "and on behalf of EDGESTONE CAPITAL EQUITY FUND, L.P."
        )
        text = "\n".join(
            [
                "EDGESTONE CAPITAL PARTNERS INC.",
                "By:",
                "Name:",
                "Title:",
                "",
                sentence,
                "",
                "Name:",
                "Title:",
                "",
                "CAMERON CAPITAL CORPORATION",
                "By:",
                "Name: Steve Hudson",
                "Title: President",
                "",
                "Name:",
                "Title:",
                "",
                entity,
            ]
        )

        lines = [
            line for line in f_xml.format_signature_text_like_screen(text).splitlines() if line
        ]

        self.assertEqual(
            lines,
            [
                "EDGESTONE CAPITAL PARTNERS INC.",
                "By:",
                "Name:",
                "Title:",
                sentence,
                "Name:",
                "Title:",
                "CAMERON CAPITAL CORPORATION",
                "By:",
                "Name: Steve Hudson",
                "Title: President",
                "Name:",
                "Title:",
                entity,
            ],
        )

    def test_format_signature_text_like_screen_keeps_empty_labels_as_printed(self) -> None:
        text = "\n".join(
            ["BY:", "PRINTED NAME:", "TITLE", "", "KM INVESTMENT CORPORATION", "Its", "46"]
        )

        lines = [
            line for line in f_xml.format_signature_text_like_screen(text).splitlines() if line
        ]

        self.assertEqual(
            lines,
            ["BY:", "PRINTED NAME:", "TITLE", "KM INVESTMENT CORPORATION", "Its", "46"],
        )

    def test_format_signature_text_like_screen_fills_title_and_capacity_values(self) -> None:
        # 9576169c page b3df3f20: a capacity line is the Title value, not a sentence.
        icore = (
            "in his capacity as the representative for the Unitholders of "
            "ICORE HealthCare, LLC"
        )
        long_title = (
            "Executive Vice President, General Counsel and Secretary of the Company "
            "and its Subsidiaries"
        )
        text = "\n".join(
            [
                "Name: Raju Mantena",
                "Title:",
                icore,
                "",
                "Name: Jane Doe",
                "Title:",
                long_title,
                "",
                "Name: John Roe",
                "Title:",
                "GENERAL PARTNER OF ACME HOLDINGS LP",
                "",
                "By: /s/ Ann Poe",
                "Its:",
                "MANAGING MEMBER OF ACME CAPITAL LLC",
                "",
                "ACME FUND, L.P.",
                "By:",
                "ACME FUND GP, LLC, ITS GENERAL PARTNER",
                "By:",
                "/s/ Max Moe",
                "",
                "By:",
                "GENERAL ELECTRIC CAPITAL CORPORATION",
                "By: /s/ [Illegible]",
            ]
        )

        lines = [
            line for line in f_xml.format_signature_text_like_screen(text).splitlines() if line
        ]

        self.assertEqual(
            lines,
            [
                "Name: Raju Mantena",
                f"Title: {icore}",
                "Name: Jane Doe",
                f"Title: {long_title}",
                "Name: John Roe",
                "Title: GENERAL PARTNER OF ACME HOLDINGS LP",
                "By: /s/ Ann Poe",
                "Its: MANAGING MEMBER OF ACME CAPITAL LLC",
                "ACME FUND, L.P.",
                "By: ACME FUND GP, LLC, ITS GENERAL PARTNER",
                "By: /s/ Max Moe",
                "By:",
                "GENERAL ELECTRIC CAPITAL CORPORATION",
                "By: /s/ [Illegible]",
            ],
        )

    def test_format_signature_text_like_screen_splits_entities_with_office_words(self) -> None:
        # Entity names that open with an office word are no Title value.
        entities = [
            "DIRECTOR'S TRUST INC.",
            "PRINCIPAL FINANCIAL GROUP, INC.",
            "PRESIDENTIAL REALTY CORPORATION",
        ]
        text_lines: list[str] = []
        for entity in entities:
            text_lines.extend(["Name:", "Title:", "", entity, "By: /s/ Jane Doe"])

        lines = [
            line
            for line in f_xml.format_signature_text_like_screen("\n".join(text_lines)).splitlines()
            if line
        ]

        expected: list[str] = []
        for entity in entities:
            expected.extend(["Name:", "Title:", entity, "By: /s/ Jane Doe"])
        self.assertEqual(lines, expected)

    def test_format_signature_html_like_screen_keeps_gap_at_script_change(self) -> None:
        raw_html = "<p>By: /s/ Wang<font>王</font>小<font>明</font> Li<font>ly</font></p>"

        formatted = f_xml.format_signature_html_like_screen(raw_html)

        self.assertEqual(formatted, "By: /s/ Wang 王小明 Lily")

    def test_format_signature_text_like_screen_still_fills_ordinary_continuations(
        self,
    ) -> None:
        text = "\n".join(
            [
                "ACME FUND, L.P.",
                "By:",
                "ACME GP, LLC",
                "Its: General Partner",
                "By:",
                "/s/ Jane Doe",
                "Name:",
                "JANE DOE",
                "Title:",
                "CO-CHIEF EXECUTIVE OFFICER",
                "",
                "BETA HOLDINGS, INC.",
                "By:",
                "/s/ John Roe",
                "Title:",
                "President of the Company",
            ]
        )

        lines = [
            line for line in f_xml.format_signature_text_like_screen(text).splitlines() if line
        ]

        self.assertEqual(
            lines,
            [
                "ACME FUND, L.P.",
                "By: ACME GP, LLC",
                "Its: General Partner",
                "By: /s/ Jane Doe",
                "Name: JANE DOE",
                "Title: CO-CHIEF EXECUTIVE OFFICER",
                "BETA HOLDINGS, INC.",
                "By: /s/ John Roe",
                "Title: President of the Company",
            ],
        )

    def test_generate_xml_reformats_html_signature_raw_page_content(self) -> None:
        agreement_uuid = "11111111-1111-1111-1111-111111111111"
        df = pd.DataFrame(
            [
                {
                    "agreement_uuid": agreement_uuid,
                    "page_uuid": "sig-page",
                    "page_order": 1,
                    "raw_page_content": """
                    <div><font>IN WITNESS WHEREOF</font></div>
                    <div>
                      <table>
                        <tr><td></td><td>ACME CORP.</td></tr>
                        <tr><td></td><td>By:</td><td>/s/ Jane Doe</td></tr>
                        <tr><td></td><td>Name:</td><td>Jane Doe</td></tr>
                        <tr><td></td><td>Its:</td><td>Chief Executive Officer</td></tr>
                      </table>
                    </div>
                    """,
                    "source_page_type": "body",
                    "gold_label": "sig",
                    "tagged_output": (
                        "IN WITNESS WHEREOF\n\n"
                        "ACME CORP. By: /s/ Jane Doe Name: Jane Doe Its: Chief Executive Officer"
                    ),
                    "url": "https://example.com/agreement",
                    "filing_date": date(2024, 1, 1),
                    "source_is_txt": False,
                    "source_is_html": True,
                },
            ]
        )

        generated, failures = f_xml.generate_xml(df)

        self.assertEqual(failures, [])
        self.assertEqual(len(generated), 1)
        root = ET.fromstring(generated[0].xml)
        sig_pages = root.find("sigPages")
        self.assertIsNotNone(sig_pages)
        assert sig_pages is not None
        sig_texts = [el.text or "" for el in sig_pages.findall("text")]
        self.assertIn("ACME CORP.", sig_texts)
        self.assertIn("By: /s/ Jane Doe", sig_texts)
        self.assertIn("Name: Jane Doe", sig_texts)
        self.assertIn("Its: Chief Executive Officer", sig_texts)

    def test_generate_xml_skips_agreement_with_invalid_xml_token(self) -> None:
        valid_uuid = "11111111-1111-1111-1111-111111111111"
        invalid_uuid = "22222222-2222-2222-2222-222222222222"
        df = pd.DataFrame(
            [
                {
                    "agreement_uuid": valid_uuid,
                    "page_uuid": "page-valid",
                    "page_order": 1,
                    "source_page_type": "body",
                    "tagged_output": "<article>ARTICLE I</article>Valid text",
                    "url": "https://example.com/valid",
                    "filing_date": date(2024, 1, 1),
                    "source_is_html": True,
                },
                {
                    "agreement_uuid": invalid_uuid,
                    "page_uuid": "page-invalid",
                    "page_order": 1,
                    "source_page_type": "body",
                    "tagged_output": "<article>ARTICLE I</article>Bad text",
                    "url": "https://example.com/invalid",
                    "filing_date": date(2024, 1, 1),
                    "source_is_html": True,
                },
            ]
        )

        original_convert_to_xml = f_xml.convert_to_xml

        def mock_convert_to_xml(
            tagged_text: str,
            agreement_uuid: str,
            filing_date: date,
            url: str,
            source_format: str,
            **kwargs: object,
        ) -> str:
            if agreement_uuid == invalid_uuid:
                return "<document><broken></document>"
            return original_convert_to_xml(
                tagged_text,
                agreement_uuid,
                filing_date,
                url,
                source_format,
                **kwargs,  # pyright: ignore[reportArgumentType]
            )

        with patch(
            "etl.domain.f_xml.convert_to_xml", side_effect=mock_convert_to_xml
        ):
            generated, failures = f_xml.generate_xml(df)

        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0].agreement_uuid, valid_uuid)
        valid_root = ET.fromstring(generated[0].xml)
        valid_metadata = valid_root.find("metadata")
        self.assertIsNotNone(valid_metadata)
        assert valid_metadata is not None
        self.assertEqual(valid_metadata.findtext("agreementUuid"), valid_uuid)
        self.assertIsNone(valid_metadata.find("target"))
        self.assertIsNone(valid_metadata.find("acquirer"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].agreement_uuid, invalid_uuid)

    def test_convert_to_xml_metadata_uses_agreement_uuid_not_party_names(self) -> None:
        agreement_uuid = "11111111-1111-1111-1111-111111111111"

        xml_str = f_xml.convert_to_xml(
            tagged_text="<article>ARTICLE I</article>Body text",
            agreement_uuid=agreement_uuid,
            filing_date=date(2024, 1, 1),
            url="https://example.com",
            source_format="html",
        )
        root = ET.fromstring(xml_str)
        metadata = root.find("metadata")
        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertEqual(metadata.findtext("agreementUuid"), agreement_uuid)
        self.assertIsNone(metadata.find("target"))
        self.assertIsNone(metadata.find("acquirer"))

    def test_convert_to_xml_sets_heading_page_uuid_attribute(self) -> None:
        agreement_uuid = "11111111-1111-1111-1111-111111111111"
        tagged_text = (
            "<article>ARTICLE I</article>"
            "<section>1.01 First section.</section>"
            "Body text on page one."
            "<pageUUID>page-1</pageUUID>\n"
            "<section>1.02 Second section.</section>"
            "Body text on page two."
            "<pageUUID>page-2</pageUUID>"
        )

        xml_str = f_xml.convert_to_xml(
            tagged_text=tagged_text,
            agreement_uuid=agreement_uuid,
            filing_date=date(2024, 1, 1),
            url="https://example.com",
            source_format="html",
        )
        root = ET.fromstring(xml_str)

        body = root.find("body")
        self.assertIsNotNone(body)
        assert body is not None
        sections = list(body.iter("section"))
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].attrib.get("pageUUID"), "page-1")
        self.assertEqual(sections[1].attrib.get("pageUUID"), "page-2")

    def test_convert_to_xml_uses_normalized_section_uuid_formula(self) -> None:
        agreement_uuid = "11111111-1111-1111-1111-111111111111"
        tagged_text = (
            "<article>ARTICLE I DEFINITIONS</article>"
            "<section>Section 1.01 . Definitions .</section>"
            "Alpha body."
            "<section>Section 1.02 The Merger .</section>"
            "Beta body."
        )

        xml_str = f_xml.convert_to_xml(
            tagged_text=tagged_text,
            agreement_uuid=agreement_uuid,
            filing_date=date(2024, 1, 1),
            url="https://example.com",
            source_format="html",
        )
        root = ET.fromstring(xml_str)
        body = root.find("body")
        self.assertIsNotNone(body)
        assert body is not None
        sections = list(body.iter("section"))
        self.assertEqual(len(sections), 2)
        self.assertEqual(
            sections[0].attrib["uuid"],
            f_xml.make_section_uuid(
                agreement_uuid,
                "definitions",
                "definitions",
                1,
                1,
            ),
        )
        self.assertEqual(
            sections[1].attrib["uuid"],
            f_xml.make_section_uuid(
                agreement_uuid,
                "definitions",
                "the merger",
                1,
                2,
            ),
        )

    def test_make_section_uuid_is_stable_across_heading_spacing_noise(self) -> None:
        agreement_uuid = "11111111-1111-1111-1111-111111111111"
        normalized_uuid = f_xml.make_section_uuid(
            agreement_uuid,
            "definitions",
            "definitions",
            1,
            1,
        )
        self.assertEqual(
            normalized_uuid,
            f_xml.make_section_uuid(
                agreement_uuid,
                "definitions",
                "definitions",
                1,
                1,
            ),
        )

    def test_convert_to_xml_keeps_section_text_across_inline_pageuuid_markers(self) -> None:
        agreement_uuid = "11111111-1111-1111-1111-111111111111"
        tagged_text = (
            "<article>ARTICLE V CERTAIN AGREEMENTS</article>"
            "<section>Section 5.5 Cooperation.</section>"
            "(a) At the expense of Purchaser, the Equityholders shall cooperate "
            "in obtaining the Financing necessary<pageUUID>page-11</pageUUID>\n"
            "to complete the transactions contemplated hereby.<pageUUID>page-12</pageUUID>\n"
            "<page>11</page>\n"
            "<section>Section 5.6 D&O Insurance.</section>"
            "Purchaser shall obtain tail coverage."
        )

        xml_str = f_xml.convert_to_xml(
            tagged_text=tagged_text,
            agreement_uuid=agreement_uuid,
            filing_date=date(2024, 1, 1),
            url="https://example.com",
            source_format="html",
        )
        root = ET.fromstring(xml_str)
        body = root.find("body")
        self.assertIsNotNone(body)
        assert body is not None
        sections = list(body.iter("section"))
        self.assertEqual(len(sections), 2)

        section_55 = sections[0]
        section_55_xml = ET.tostring(section_55, encoding="unicode")
        self.assertIn("obtaining the Financing necessary", section_55_xml)
        self.assertIn("to complete the transactions contemplated hereby.", section_55_xml)
        self.assertIn("<pageUUID>page-11</pageUUID>", section_55_xml)
        self.assertIn("<pageUUID>page-12</pageUUID>", section_55_xml)
        self.assertIn("<page>11</page>", section_55_xml)

    def test_strip_subsection_tags_keeps_section_that_mentions_subsection_number(self) -> None:
        text = (
            "<section>Section 11.10. Escrow Agent.</section> Body. "
            "<section>Section 11.11. Disbursements from Section 11.1.4 Escrow Account</section> Body."
        )
        self.assertEqual(f_xml.strip_subsection_section_tags(text), text)

    def test_strip_subsection_tags_unwraps_real_subsection_headings(self) -> None:
        cases = {
            "<section>11.1.4 Escrow Account</section> Body.": "11.1.4 Escrow Account Body.",
            "<section>Section 2.3.1. Closing Deliveries</section> Body.": "Section 2.3.1. Closing Deliveries Body.",
            "<section>  § 4.2.1.3 Notices</section>": "  § 4.2.1.3 Notices",
        }
        for tagged, expected in cases.items():
            with self.subTest(tagged=tagged):
                self.assertEqual(f_xml.strip_subsection_section_tags(tagged), expected)

    def test_convert_to_xml_restores_space_in_glued_heading_titles(self) -> None:
        tagged_text = (
            "<article>ARTICLE VIDEFINITIONS</article>"
            "<section>Section 6.1Closing.</section>Body."
            "<article>ARTICLE VIIINDEMNIFICATION</article>"
            "<section>7.1 Survival.</section>Body."
        )
        xml_str = f_xml.convert_to_xml(
            tagged_text=tagged_text,
            agreement_uuid="11111111-1111-1111-1111-111111111111",
            filing_date=date(2024, 1, 1),
            url="https://example.com",
            source_format="html",
        )
        root = ET.fromstring(xml_str)
        titles = [elem.attrib["title"] for elem in root.iter() if elem.tag in {"article", "section"}]
        self.assertEqual(
            titles,
            ["ARTICLE VI DEFINITIONS", "Section 6.1 Closing.", "ARTICLE VII INDEMNIFICATION", "7.1 Survival."],
        )


if __name__ == "__main__":
    _ = unittest.main()
