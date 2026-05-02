# pyright: reportAny=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownMemberType=false
import json
import re
import unittest
import xml.etree.ElementTree as ET
from datetime import date
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pandas as pd
from dagster import AssetExecutionContext
from openai import OpenAI
from etl.defs.resources import DBResource, PipelineConfig
from etl.defs.f_xml_asset import (
    XML_REASON_BODY_STARTS_NON_ARTICLE,
    XML_REASON_SECTION_ARTICLE_MISMATCH,
    XML_REASON_SECTION_NON_SEQUENTIAL,
    XML_REASON_SECTION_TITLE_INVALID_NUMBERING,
    XML_REASON_LLM_INVALID,
    XML_REASON_TOO_FEW_ARTICLES,
    XML_REASON_TOO_MANY_EMPTY_ARTICLES,
    XML_VERIFY_INSTRUCTIONS,
    _apply_safe_xml_tag_repairs_to_df,
    _build_xml_verify_batch_request_body,
    _build_xml_verify_toc_context,
    _extract_article_number,
    _hard_rule_result_for_df,
    _render_tag_tree_from_root,
    _reason_rows_changed,
    _apply_xml_verify_batch_output,
    find_hard_rule_violations,
    xml_verify_asset,
)
from etl.domain.xml_tag_repairs import (
    SectionGap,
    split_combined_missing_section_tags,
)


class XMLVerifyAssetTests(unittest.TestCase):
    def _xml_build_df(self, tagged_output: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "agreement_uuid": "11111111-1111-1111-1111-111111111111",
                    "page_uuid": "body-page",
                    "page_order": 1,
                    "raw_page_content": "",
                    "source_page_type": "body",
                    "gold_label": None,
                    "tagged_output": tagged_output,
                    "url": "https://example.com/agreement",
                    "filing_date": date(2024, 1, 1),
                    "source_is_txt": True,
                    "source_is_html": False,
                }
            ]
        )

    def _xml_build_df_rows(self, tagged_outputs: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "agreement_uuid": "11111111-1111-1111-1111-111111111111",
                    "page_uuid": f"body-page-{idx}",
                    "page_order": idx,
                    "raw_page_content": "",
                    "source_page_type": "body",
                    "gold_label": None,
                    "tagged_output": tagged_output,
                    "url": "https://example.com/agreement",
                    "filing_date": date(2024, 1, 1),
                    "source_is_txt": True,
                    "source_is_html": False,
                }
                for idx, tagged_output in enumerate(tagged_outputs, start=1)
            ]
        )

    def _five_article_text(self, article_one: str) -> str:
        return (
            "<article>ARTICLE I</article>"
            + article_one
            + "<article>ARTICLE II</article><section>2.1 Second</section>"
            + "<article>ARTICLE III</article><section>3.1 Third</section>"
            + "<article>ARTICLE IV</article><section>4.1 Fourth</section>"
            + "<article>ARTICLE V</article><section>5.1 Fifth</section>"
        )

    def _normalized_text_without_heading_tags(self, tagged_output: str) -> str:
        return re.sub(
            r"\s+",
            "",
            re.sub(r"</?(?:article|section)>", "", tagged_output, flags=re.IGNORECASE),
        )

    def test_safe_xml_tag_repairs_split_combined_missing_section_tags(self) -> None:
        tagged_output = self._five_article_text(
            "".join(
                [
                    "<section>1.1 First</section>",
                    "<section>1.2 Omitted. 1.3 Third</section>",
                    "<section>1.4 Fourth</section>",
                ]
            )
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(sum(before_counts.values()), 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn("<section>1.3 Third</section>", repaired_text)
        self.assertEqual(
            self._normalized_text_without_heading_tags(tagged_output),
            self._normalized_text_without_heading_tags(repaired_text),
        )

    def test_split_combined_missing_section_tags_ignores_orphan_section_label_fragments(self) -> None:
        cases = [
            (
                "<section>Sections 7.2 Indemnification of the Purchaser Indemnitees.</section>",
                SectionGap(article_num=7, expected=2, found=3),
            ),
            (
                "<section>Secton 6.11 R&W Insurance Policy.</section>",
                SectionGap(article_num=6, expected=11, found=12),
            ),
        ]

        for tagged_text, gap in cases:
            with self.subTest(tagged_text=tagged_text):
                repaired, stats = split_combined_missing_section_tags(tagged_text, [gap])

                self.assertEqual(repaired, tagged_text)
                self.assertEqual(stats.applied, {})

    def test_split_combined_missing_section_tags_still_splits_real_combined_headings(self) -> None:
        tagged_text = (
            "<section>Section 5.02 Cooper Filings Section 5.03 Non-Solicitation</section>"
        )

        repaired, stats = split_combined_missing_section_tags(
            tagged_text,
            [SectionGap(article_num=5, expected=3, found=4)],
        )

        self.assertEqual(
            repaired,
            "".join(
                [
                    "<section>Section 5.02 Cooper Filings</section>\n\n",
                    "<section>Section 5.03 Non-Solicitation</section>",
                ]
            ),
        )
        self.assertEqual(stats.applied["split_combined_missing_section_tags"], 1)

    def test_safe_xml_tag_repairs_do_not_create_orphan_section_label_sections(self) -> None:
        cases = [
            (
                "".join(
                    [
                        "<section>7.1 Survival</section>",
                        "<section>Sections 7.2 Indemnification</section>",
                        "<section>7.3 Seller Indemnification</section>",
                    ]
                ),
                "<section>Sections</section>",
            ),
            (
                "".join(
                    [
                        "<section>6.10 Attorney-Client Privilege</section>",
                        "<section>Secton 6.11 R&W Insurance Policy.</section>",
                        "<section>6.12 Sellers Representative</section>",
                    ]
                ),
                "<section>Secton</section>",
            ),
        ]

        for article_body, orphan_tag in cases:
            with self.subTest(orphan_tag=orphan_tag):
                tagged_output = self._five_article_text(article_body)
                df = self._xml_build_df(tagged_output)

                repaired_df = _apply_safe_xml_tag_repairs_to_df(df)

                repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
                self.assertNotIn(orphan_tag, repaired_text)
                self.assertIn(article_body, repaired_text)

    def test_safe_xml_tag_repairs_split_combined_quoted_definition_tags(self) -> None:
        tagged_output = self._five_article_text(
            "".join(
                [
                    "<section>1.1 First</section>",
                    "<section>1.2 Agreement</section>",
                    '<section>1.3 "ABCA" means the Arkansas Business Corporation Act. 1.4 "BCL" means business corporation law.</section>',
                    "<section>1.5 Fifth</section>",
                ]
            )
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(before_counts[XML_REASON_SECTION_NON_SEQUENTIAL], 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn('<section>1.4 "BCL" means business corporation law.</section>', repaired_text)
        self.assertEqual(
            self._normalized_text_without_heading_tags(tagged_output),
            self._normalized_text_without_heading_tags(repaired_text),
        )

    def test_safe_xml_tag_repairs_split_double_omitted_section_tags(self) -> None:
        tagged_output = self._five_article_text(
            "".join(
                [
                    "<section>1.1 First</section>",
                    "<section>1.2 [Reserved.]. Section 1.3 [Reserved.]. Section 1.4 Public Announcements.</section>",
                    "<section>1.5 Fifth</section>",
                ]
            )
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(sum(before_counts.values()), 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn("<section>Section 1.3 [Reserved.].</section>", repaired_text)
        self.assertIn("<section>Section 1.4 Public Announcements.</section>", repaired_text)
        self.assertEqual(
            self._normalized_text_without_heading_tags(tagged_output),
            self._normalized_text_without_heading_tags(repaired_text),
        )

    def test_safe_xml_tag_repairs_split_combined_omitted_article_tags(self) -> None:
        tagged_output = (
            "<article>ARTICLE I</article><section>1.1 First</section>"
            "<article>ARTICLE II RESERVED</article>"
            "<article>ARTICLE III</article><section>3.1 Third</section>"
            "<article>Article IV RESERVED Article V REPRESENTATIONS AND WARRANTIES</article>"
            "<section>5.1 Organization</section>"
            "<article>ARTICLE VI</article><section>6.1 Sixth</section>"
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(before_counts[XML_REASON_SECTION_ARTICLE_MISMATCH], 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn("<article>Article IV RESERVED</article>", repaired_text)
        self.assertIn("<article>Article V REPRESENTATIONS AND WARRANTIES</article>", repaired_text)
        self.assertEqual(
            self._normalized_text_without_heading_tags(tagged_output),
            self._normalized_text_without_heading_tags(repaired_text),
        )

    def test_safe_xml_tag_repairs_insert_missing_section_heading_tags(self) -> None:
        tagged_output = self._five_article_text(
            "".join(
                [
                    "<section>1.1 First</section>",
                    "Interim text. Section 1.2 Missing Section. More text.",
                    "<section>1.3 Third</section>",
                ]
            )
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(sum(before_counts.values()), 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn("<section>Section 1.2 Missing Section. More text.</section>", repaired_text)
        self.assertEqual(
            self._normalized_text_without_heading_tags(tagged_output),
            self._normalized_text_without_heading_tags(repaired_text),
        )

    def test_safe_xml_tag_repairs_fix_body_start_article_heading_tags(self) -> None:
        tagged_output = (
            "<section>1. Purchase and Sale. 1.1 Sale</section>"
            "<section>1.2 Closing</section>"
            "<article>ARTICLE II</article><section>2.1 Second</section>"
            "<article>ARTICLE III</article><section>3.1 Third</section>"
            "<article>ARTICLE IV</article><section>4.1 Fourth</section>"
            "<article>ARTICLE V</article><section>5.1 Fifth</section>"
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(before_counts[XML_REASON_BODY_STARTS_NON_ARTICLE], 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn("<article>1. Purchase and Sale.</article>", repaired_text)
        self.assertIn("<section>1.1 Sale</section>", repaired_text)
        self.assertEqual(
            self._normalized_text_without_heading_tags(tagged_output),
            self._normalized_text_without_heading_tags(repaired_text),
        )

    def test_safe_xml_tag_repairs_promotes_cross_page_article_heading(self) -> None:
        df = self._xml_build_df_rows(
            [
                (
                    "<article>ARTICLE I</article><section>1.1 First</section>"
                    "<section>2. Representations and Warranties</section>"
                ),
                (
                    "<section>2.1 Organization</section>"
                    "<article>ARTICLE III</article><section>3.1 Third</section>"
                    "<article>ARTICLE IV</article><section>4.1 Fourth</section>"
                    "<article>ARTICLE V</article><section>5.1 Fifth</section>"
                ),
            ]
        )

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(before_counts[XML_REASON_SECTION_ARTICLE_MISMATCH], 0)
        self.assertEqual(after_counts, {})
        repaired_text = "\n".join(cast(str, value) for value in repaired_df["tagged_output"])
        self.assertIn("<article>2. Representations and Warranties</article>", repaired_text)

    def test_safe_xml_tag_repairs_promotes_article_only_heading_by_sequence(self) -> None:
        tagged_output = (
            "<article>ARTICLE I</article><section>1.1 First</section>"
            "<article>ARTICLE II</article><section>2.1 Second</section>"
            "<article>ARTICLE III</article><section>3.1 Third</section>"
            "<section>4. COSTS</section>Each party shall pay its own costs."
            "<article>ARTICLE V</article><section>5.1 Fifth</section>"
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(sum(before_counts.values()), 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn("<article>4. COSTS</article>", repaired_text)
        self.assertEqual(
            self._normalized_text_without_heading_tags(tagged_output),
            self._normalized_text_without_heading_tags(repaired_text),
        )

    def test_safe_xml_tag_repairs_promotes_article_heading_before_sequence_gap(self) -> None:
        tagged_output = (
            "<article>ARTICLE I</article><section>1.1 First</section>"
            "<article>ARTICLE II</article><section>2.1 Second</section>"
            "<article>ARTICLE III</article><section>3.1 Third</section>"
            "<section>4. CONTINUING EFFECT</section>Continuing effect text. 5. INVALIDITY Invalidity text."
            "<article>ARTICLE VI</article><section>6.1 Sixth</section>"
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(sum(before_counts.values()), 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn("<article>4. CONTINUING EFFECT</article>", repaired_text)
        self.assertEqual(
            self._normalized_text_without_heading_tags(tagged_output),
            self._normalized_text_without_heading_tags(repaired_text),
        )

    def test_safe_xml_tag_repairs_promotes_body_start_whole_number_article_by_sequence(self) -> None:
        tagged_output = (
            "<section>1. DEFINITIONS.</section>Definitions apply."
            "<article>2. PURCHASE AND SALE</article><section>2.1 Sale</section>"
            "<article>3. CLOSING</article><section>3.1 Closing</section>"
            "<article>4. COVENANTS</article><section>4.1 Covenants</section>"
            "<article>5. MISCELLANEOUS</article><section>5.1 Notices</section>"
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(before_counts[XML_REASON_BODY_STARTS_NON_ARTICLE], 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn("<article>1. DEFINITIONS.</article>", repaired_text)
        self.assertEqual(
            self._normalized_text_without_heading_tags(tagged_output),
            self._normalized_text_without_heading_tags(repaired_text),
        )

    def test_safe_xml_tag_repairs_promotes_no_dot_article_heading_by_sequence(self) -> None:
        tagged_output = (
            "<article>1 Definitions</article><section>1.1 Definitions</section>"
            "<article>2 Completion Statement</article><section>2.1 Statement</section>"
            "<section>3 Tax Indemnity</section>Tax covenant text."
            "<article>4 Specific business matters</article><section>4.1 Release</section>"
            "<article>5 Specific Indemnities</article>Specific indemnity text."
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(before_counts[XML_REASON_SECTION_TITLE_INVALID_NUMBERING], 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn("<article>3 Tax Indemnity</article>", repaired_text)
        self.assertEqual(
            self._normalized_text_without_heading_tags(tagged_output),
            self._normalized_text_without_heading_tags(repaired_text),
        )

    def test_safe_xml_tag_repairs_splits_embedded_article_with_bare_first_section(self) -> None:
        tagged_output = (
            "<article>ARTICLE I</article><section>1.1 First</section>"
            "<article>ARTICLE II</article><section>2.1 Second</section>"
            "<article>ARTICLE III</article><section>3.1 Third</section>"
            "<article>ARTICLE IV</article><section>4.1 Fourth</section>"
            "<section>5. TAXES AND EXPENSES.\n\n5.1</section>Tax allocation text."
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(before_counts[XML_REASON_SECTION_TITLE_INVALID_NUMBERING], 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn("<article>5. TAXES AND EXPENSES.</article>", repaired_text)
        self.assertIn("<section>5.1</section>", repaired_text)
        self.assertEqual(
            self._normalized_text_without_heading_tags(tagged_output),
            self._normalized_text_without_heading_tags(repaired_text),
        )

    def test_safe_xml_tag_repairs_joins_split_section_number_digit(self) -> None:
        tagged_output = (
            "<article>ARTICLE I</article><section>1.1 First</section>"
            "<article>ARTICLE II</article><section>2.1 Second</section>"
            "<article>ARTICLE III</article>"
            "<section>Section 3.28 Prior Section</section>"
            "<section>Section 3.2 9 Representations and Warranties Relating to Captive Subsidiary.</section>"
            "<section>Section 3.30 Following Section</section>"
            "<article>ARTICLE IV</article><section>4.1 Fourth</section>"
            "<article>ARTICLE V</article><section>5.1 Fifth</section>"
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(before_counts[XML_REASON_SECTION_NON_SEQUENTIAL], 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn(
            "<section>Section 3.29 Representations and Warranties Relating to Captive Subsidiary.</section>",
            repaired_text,
        )

    def test_safe_xml_tag_repairs_joins_split_article_number_digit_in_section_title(self) -> None:
        tagged_output = (
            "<article>ARTICLE I</article><section>1.1 First</section>"
            "<article>ARTICLE II</article><section>2.1 Second</section>"
            "<article>ARTICLE III</article><section>3.1 Third</section>"
            "<article>ARTICLE IV</article><section>4.1 Fourth</section>"
            "<article>ARTICLE XIV</article>"
            "<section>Section 14.4 Prior Section</section>"
            "<section>Section 1 4.5 Middle Section</section>"
            "<section>Section 14.6 Following Section</section>"
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(sum(before_counts.values()), 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn("<section>Section 14.5 Middle Section</section>", repaired_text)

    def test_safe_xml_tag_repairs_joins_split_section_number_with_next_neighbor(self) -> None:
        tagged_output = (
            "<article>ARTICLE I</article><section>1.1 First</section>"
            "<article>ARTICLE II</article><section>2.1 Second</section>"
            "<article>ARTICLE III</article><section>3.1 Third</section>"
            "<article>ARTICLE IV</article>"
            "<section>Section 4.11 Prior Section</section>"
            "<section>Section 4.1 3 Middle Section</section>"
            "<section>Section 4.14 Following Section</section>"
            "<article>ARTICLE V</article><section>5.1 Fifth</section>"
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(sum(before_counts.values()), 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn("<section>Section 4.13 Middle Section</section>", repaired_text)

    def test_safe_xml_tag_repairs_unwraps_bare_page_number_section_tag(self) -> None:
        tagged_output = self._five_article_text(
            "".join(
                [
                    "<section>1.1 First</section>",
                    "<section>3.</section>",
                    "<section>1.2 Second</section>",
                ]
            )
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(before_counts[XML_REASON_SECTION_TITLE_INVALID_NUMBERING], 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn("3.", repaired_text)
        self.assertNotIn("<section>3.</section>", repaired_text)
        self.assertEqual(
            self._normalized_text_without_heading_tags(tagged_output),
            self._normalized_text_without_heading_tags(repaired_text),
        )

    def test_safe_xml_tag_repairs_splits_leading_page_number_from_section_tag(self) -> None:
        tagged_output = (
            "<article>ARTICLE I</article><section>1.1 First</section>"
            "<article>ARTICLE II</article><section>10 2.1 Sale</section>"
            "<article>ARTICLE III</article><section>3.1 Third</section>"
            "<article>ARTICLE IV</article><section>4.1 Fourth</section>"
            "<article>ARTICLE V</article><section>5.1 Fifth</section>"
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(before_counts[XML_REASON_SECTION_TITLE_INVALID_NUMBERING], 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn("10 <section>2.1 Sale</section>", repaired_text)
        self.assertEqual(
            self._normalized_text_without_heading_tags(tagged_output),
            self._normalized_text_without_heading_tags(repaired_text),
        )

    def test_safe_xml_tag_repairs_unwraps_standalone_section_label_tag(self) -> None:
        tagged_output = self._five_article_text(
            "".join(
                [
                    "<section>1.1 First</section>",
                    "<section>Section</section>",
                    "<section>§</section>",
                    "<section>1.2 Second</section>",
                ]
            )
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(before_counts[XML_REASON_SECTION_TITLE_INVALID_NUMBERING], 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn("Section", repaired_text)
        self.assertIn("§", repaired_text)
        self.assertNotIn("<section>Section</section>", repaired_text)
        self.assertNotIn("<section>§</section>", repaired_text)
        self.assertEqual(
            self._normalized_text_without_heading_tags(tagged_output),
            self._normalized_text_without_heading_tags(repaired_text),
        )

    def test_safe_xml_tag_repairs_adds_space_after_section_prefix(self) -> None:
        tagged_output = (
            "<article>ARTICLE I</article><section>Section1.1 First</section>"
            "<article>ARTICLE II</article><section>Section2.1 Second</section>"
            "<article>ARTICLE III</article><section>Section3.14 Agreements with Regulatory Agencies.</section>"
            "<article>ARTICLE IV</article><section>Section4.1 Fourth</section>"
            "<article>ARTICLE V</article><section>Section5.1 Fifth</section>"
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(before_counts[XML_REASON_SECTION_TITLE_INVALID_NUMBERING], 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn("<section>Section 3.14 Agreements with Regulatory Agencies.</section>", repaired_text)
        self.assertNotIn("Section3.14", repaired_text)

    def test_safe_xml_tag_repairs_unwraps_inline_section_reference_tag(self) -> None:
        tagged_output = (
            "<article>ARTICLE I</article><section>1.1 First</section>"
            "<article>ARTICLE II</article><section>2.1 Second</section>"
            "<article>ARTICLE III</article><section>3.1 Third</section>"
            "<article>ARTICLE IV</article>"
            "<section>4.1 Fourth</section>"
            "<section>Section 4.13, or under applicable Legal Requirements.</section>"
            "<section>4.2 Next</section>"
            "<article>ARTICLE V</article><section>5.1 Fifth</section>"
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(sum(before_counts.values()), 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn("Section 4.13, or under applicable Legal Requirements.", repaired_text)
        self.assertNotIn(
            "<section>Section 4.13, or under applicable Legal Requirements.</section>",
            repaired_text,
        )
        self.assertEqual(
            self._normalized_text_without_heading_tags(tagged_output),
            self._normalized_text_without_heading_tags(repaired_text),
        )

    def test_safe_xml_tag_repairs_unwraps_decimal_comma_reference_tag(self) -> None:
        tagged_output = (
            "<article>ARTICLE I</article><section>1.1 First</section>"
            "<article>ARTICLE II</article><section>2.1 Second</section>"
            "<article>ARTICLE III</article><section>3.1 Third</section>"
            "<article>ARTICLE IV</article>"
            "<section>4.1 Fourth</section>"
            "<section>4.6, no representation</section>"
            "<section>4.2 Next</section>"
            "<article>ARTICLE V</article><section>5.1 Fifth</section>"
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(sum(before_counts.values()), 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn("4.6, no representation", repaired_text)
        self.assertNotIn("<section>4.6, no representation</section>", repaired_text)
        self.assertEqual(
            self._normalized_text_without_heading_tags(tagged_output),
            self._normalized_text_without_heading_tags(repaired_text),
        )

    def test_safe_xml_tag_repairs_merges_cross_page_split_article_title(self) -> None:
        df = self._xml_build_df_rows(
            [
                (
                    "<article>ARTICLE I</article><section>1.1 First</section>"
                    "<article>ARTICLE II</article><section>2.1 Second</section>"
                    "<article>ARTICLE III.</article>"
                ),
                (
                    "<article>REPRESENTATIONS AND WARRANTIES OF BUYERS</article>"
                    "<section>3.1 Organization</section>"
                    "<section>3.2 Authorization</section>"
                    "<article>ARTICLE IV</article><section>4.1 Fourth</section>"
                    "<article>ARTICLE V</article><section>5.1 Fifth</section>"
                ),
            ]
        )

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(sum(before_counts.values()), 0)
        self.assertEqual(after_counts, {})
        repaired_text = "\n".join(cast(str, value) for value in repaired_df["tagged_output"])
        self.assertIn(
            "<article>ARTICLE III. REPRESENTATIONS AND WARRANTIES OF BUYERS</article>",
            repaired_text,
        )
        self.assertNotIn(
            "<article>REPRESENTATIONS AND WARRANTIES OF BUYERS</article>",
            repaired_text,
        )

    def test_safe_xml_tag_repairs_wraps_untagged_article_heading_before_first_section(self) -> None:
        tagged_output = (
            "<article>ARTICLE I</article><section>1.1 First</section>"
            "<article>ARTICLE II</article><section>2.1 Second</section>"
            "<article>ARTICLE III</article><section>3.1 Third</section>"
            "<article>ARTICLE IV</article><section>4.1 Fourth</section>"
            "SECTION 5. MISCELLANEOUS.\n\n"
            "<section>5.1 Notices</section>"
        )
        df = self._xml_build_df(tagged_output)

        before_counts, _ = _hard_rule_result_for_df(df)
        repaired_df = _apply_safe_xml_tag_repairs_to_df(df)
        after_counts, _ = _hard_rule_result_for_df(repaired_df)

        self.assertGreater(before_counts[XML_REASON_SECTION_ARTICLE_MISMATCH], 0)
        self.assertEqual(after_counts, {})
        repaired_text = cast(str, repaired_df.iloc[0]["tagged_output"])
        self.assertIn("<article>SECTION 5. MISCELLANEOUS.</article>", repaired_text)
        self.assertEqual(
            self._normalized_text_without_heading_tags(tagged_output),
            self._normalized_text_without_heading_tags(repaired_text),
        )

    def test_extract_article_number_accepts_numeric_formats_without_roman_false_positive(self) -> None:
        self.assertEqual(_extract_article_number("ARTICLE IV Representations"), 4)
        self.assertEqual(_extract_article_number("1. Purchase and Sale"), 1)
        self.assertEqual(_extract_article_number("Section 2. Consideration"), 2)
        self.assertIsNone(_extract_article_number("Indemnification"))

    def test_find_hard_rule_violations_accepts_numeric_article_titles(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="1. Purchase and Sale"><section title="1.1 Sale" pageUUID="page-1" /></article>
                <article title="Section 2. Consideration"><section title="2.1 Purchase Price" pageUUID="page-2" /></article>
                <article title="3. Closing"><section title="3.1 Closing" pageUUID="page-3" /></article>
                <article title="4. Covenants"><section title="4.1 Covenants" pageUUID="page-4" /></article>
                <article title="5. Miscellaneous"><section title="5.1 Notices" pageUUID="page-5" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        self.assertFalse(
            any(
                v.reason_code
                in {XML_REASON_SECTION_TITLE_INVALID_NUMBERING, XML_REASON_TOO_FEW_ARTICLES}
                for v in violations
            )
        )
        self.assertEqual(violations, [])

    def test_find_hard_rule_violations_rejects_section_prefix_without_space(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE I"><section title="Section1.1 First" pageUUID="page-1" /></article>
                <article title="ARTICLE II"><section title="Section2.1 Second" pageUUID="page-2" /></article>
                <article title="ARTICLE III"><section title="Section3.1 Third" pageUUID="page-3" /></article>
                <article title="ARTICLE IV"><section title="Section4.1 Fourth" pageUUID="page-4" /></article>
                <article title="ARTICLE V"><section title="Section5.1 Fifth" pageUUID="page-5" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        self.assertTrue(
            any(
                violation.reason_code == XML_REASON_SECTION_TITLE_INVALID_NUMBERING
                for violation in violations
            )
        )

    def test_section_article_mismatch_targets_previous_page_with_missed_article_heading(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE I"><section title="Section 1.1 First" pageUUID="page-1" /></article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-2" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-3" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-4" /></article>
                <article title="Section 5. Agreements" pageUUID="page-5-start">
                  <section title="5.11 Resale Registration" pageUUID="page-5-start">
                    <text>Final Section 5 text.</text>
                    <text>Section 6. Conditions Precedent.</text>
                    <page>59</page>
                    <pageUUID>page-6-heading</pageUUID>
                    <text>Introductory Article 6 text.</text>
                  </section>
                  <section title="6.1 No Restraints" pageUUID="page-6-body" />
                  <section title="6.2 Stockholder Approval" pageUUID="page-6-body" />
                </article>
              </body>
            </document>
            """
        )

        violations = [
            violation
            for violation in find_hard_rule_violations(root)
            if violation.reason_code == XML_REASON_SECTION_ARTICLE_MISMATCH
        ]

        self.assertEqual(len(violations), 2)
        self.assertTrue(all(violation.page_uuids == ("page-6-heading",) for violation in violations))

    def test_section_article_mismatch_targets_current_page_without_missed_article_heading(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE I"><section title="Section 1.1 First" pageUUID="page-1" /></article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-2" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-3" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-4" /></article>
                <article title="Section 5. Agreements" pageUUID="page-5-start">
                  <section title="5.11 Resale Registration" pageUUID="page-5-start">
                    <text>Final Section 5 text.</text>
                  </section>
                  <section title="6.1 No Restraints" pageUUID="page-6-body" />
                </article>
              </body>
            </document>
            """
        )

        violations = [
            violation
            for violation in find_hard_rule_violations(root)
            if violation.reason_code == XML_REASON_SECTION_ARTICLE_MISMATCH
        ]

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].page_uuids, ("page-6-body",))

    def test_find_hard_rule_violations_uses_section_page_uuid_attribute(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE III">
                  <section title="AND" pageUUID="page-22">
                    <text>Body text</text>
                    <pageUUID>page-99</pageUUID>
                  </section>
                </article>
              </body>
            </document>
            """
        )
        violations = find_hard_rule_violations(root)
        target = next(
            v for v in violations if v.reason_code == XML_REASON_SECTION_TITLE_INVALID_NUMBERING
        )
        self.assertEqual(target.page_uuids, ("page-22",))

    def test_find_hard_rule_violations_rejects_fewer_than_five_articles(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE I"><section title="Section 1.1 First" pageUUID="page-1" /></article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-2" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-3" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-4" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        target = next(v for v in violations if v.reason_code == XML_REASON_TOO_FEW_ARTICLES)
        self.assertEqual(
            target.reason_detail,
            "Too few articles: found 4, minimum required is 5.",
        )

    def test_find_hard_rule_violations_allows_five_articles(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE I"><section title="Section 1.1 First" pageUUID="page-1" /></article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-2" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-3" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-4" /></article>
                <article title="ARTICLE V"><section title="Section 5.1 Fifth" pageUUID="page-5" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        self.assertFalse(any(v.reason_code == XML_REASON_TOO_FEW_ARTICLES for v in violations))

    def test_find_hard_rule_violations_ignores_omitted_empty_articles(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE I"><section title="Section 1.1 First" pageUUID="page-1" /></article>
                <article title="ARTICLE II RESERVED" pageUUID="page-2" />
                <article title="ARTICLE III INTENTIONALLY OMITTED" pageUUID="page-3" />
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-4" /></article>
                <article title="ARTICLE V"><section title="Section 5.1 Fifth" pageUUID="page-5" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        self.assertFalse(
            any(v.reason_code == XML_REASON_TOO_MANY_EMPTY_ARTICLES for v in violations)
        )

    def test_find_hard_rule_violations_does_not_count_article_text_as_empty(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE I"><section title="Section 1.1 First" pageUUID="page-1" /></article>
                <article title="ARTICLE II"><text>Standalone article text.</text><pageUUID>page-2</pageUUID></article>
                <article title="ARTICLE III"><text>More standalone article text.</text></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-4" /></article>
                <article title="ARTICLE V"><section title="Section 5.1 Fifth" pageUUID="page-5" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        self.assertFalse(
            any(v.reason_code == XML_REASON_TOO_MANY_EMPTY_ARTICLES for v in violations)
        )

    def test_find_hard_rule_violations_counts_untagged_sections_as_empty_article(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE I"><section title="Section 1.1 First" pageUUID="page-1" /></article>
                <article title="ARTICLE II">
                  <text>Section 2.1 Missing Section Heading.</text>
                  <text>Section 2.2 Also Missing.</text>
                </article>
                <article title="ARTICLE III">
                  <text>3.1 Missing Decimal Section Heading.</text>
                </article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-4" /></article>
                <article title="ARTICLE V"><section title="Section 5.1 Fifth" pageUUID="page-5" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        target = next(v for v in violations if v.reason_code == XML_REASON_TOO_MANY_EMPTY_ARTICLES)
        self.assertEqual(
            target.reason_detail,
            "Too many empty articles: found 2, maximum allowed is 1.",
        )

    def test_find_hard_rule_violations_suppresses_forward_gap_when_missing_number_absent(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE I">
                  <section title="Section 1.1 First" pageUUID="page-1" />
                  <section title="Section 1.2 Second" pageUUID="page-2" />
                  <section title="Section 1.4 Closing" pageUUID="page-3">
                    <pageUUID>page-4</pageUUID>
                  </section>
                </article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-5" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-6" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-7" /></article>
                <article title="ARTICLE V"><section title="Section 5.1 Fifth" pageUUID="page-8" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        self.assertFalse(any(v.reason_code == XML_REASON_SECTION_NON_SEQUENTIAL for v in violations))

    def test_find_hard_rule_violations_targets_previous_section_when_it_mentions_missing_heading(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE I">
                  <section title="Section 1.1 First" pageUUID="page-1" />
                  <section title="Section 1.2 Second" pageUUID="page-2">
                    <text>Section 1.3 Interim Covenants.</text>
                  </section>
                  <section title="Section 1.4 Closing" pageUUID="page-3" />
                </article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-5" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-6" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-7" /></article>
                <article title="ARTICLE V"><section title="Section 5.1 Fifth" pageUUID="page-8" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        target = next(v for v in violations if v.reason_code == XML_REASON_SECTION_NON_SEQUENTIAL)
        self.assertEqual(target.page_uuids, ("page-2",))

    def test_find_hard_rule_violations_targets_both_adjacent_pages_when_forward_gap_is_ambiguous(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE I">
                  <section title="Section 1.1 First" pageUUID="page-1" />
                  <section title="Section 1.2 Second" pageUUID="page-2" />
                  <section title="Section 1.4 Closing" pageUUID="page-3" />
                  <text>See Section 1.3 elsewhere.</text>
                </article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-5" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-6" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-7" /></article>
                <article title="ARTICLE V"><section title="Section 5.1 Fifth" pageUUID="page-8" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        target = next(v for v in violations if v.reason_code == XML_REASON_SECTION_NON_SEQUENTIAL)
        self.assertEqual(target.page_uuids, ("page-2", "page-3"))

    def test_find_hard_rule_violations_suppresses_duplicate_when_expected_number_absent(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE I">
                  <section title="Section 1.1 First" pageUUID="page-1" />
                  <section title="Section 1.2 Second" pageUUID="page-2" />
                  <section title="Section 1.2 Duplicate Source Number" pageUUID="page-3" />
                  <section title="Section 1.4 Closing" pageUUID="page-4" />
                </article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-5" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-6" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-7" /></article>
                <article title="ARTICLE V"><section title="Section 5.1 Fifth" pageUUID="page-8" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        self.assertFalse(any(v.reason_code == XML_REASON_SECTION_NON_SEQUENTIAL for v in violations))

    def test_find_hard_rule_violations_suppresses_forward_gap_when_toc_matches_body(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <tableOfContents>
                <text>1.1 First</text>
                <text>1.2 Second</text>
                <text>1.4 Closing</text>
              </tableOfContents>
              <body>
                <article title="ARTICLE I">
                  <section title="Section 1.1 First" pageUUID="page-1" />
                  <section title="Section 1.2 Second" pageUUID="page-2" />
                  <section title="Section 1.4 Closing" pageUUID="page-3" />
                  <text>See Section 1.3 elsewhere.</text>
                </article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-5" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-6" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-7" /></article>
                <article title="ARTICLE V"><section title="Section 5.1 Fifth" pageUUID="page-8" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        self.assertFalse(any(v.reason_code == XML_REASON_SECTION_NON_SEQUENTIAL for v in violations))

    def test_find_hard_rule_violations_suppresses_gap_when_missing_number_only_in_toc(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <tableOfContents>
                <text>1.1 First</text>
                <text>1.2 Second</text>
                <text>1.3 Third</text>
                <text>1.4 Closing</text>
              </tableOfContents>
              <body>
                <article title="ARTICLE I">
                  <section title="Section 1.1 First" pageUUID="page-1" />
                  <section title="Section 1.2 Second" pageUUID="page-2" />
                  <section title="Section 1.4 Closing" pageUUID="page-3" />
                </article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-5" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-6" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-7" /></article>
                <article title="ARTICLE V"><section title="Section 5.1 Fifth" pageUUID="page-8" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        self.assertFalse(any(v.reason_code == XML_REASON_SECTION_NON_SEQUENTIAL for v in violations))

    def test_find_hard_rule_violations_suppresses_local_toc_gap_when_article_differs_elsewhere(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <tableOfContents>
                <text>1.1 First</text>
                <text>1.2 Second</text>
                <text>1.4 Closing</text>
                <text>1.6 Post-Closing</text>
              </tableOfContents>
              <body>
                <article title="ARTICLE I">
                  <section title="Section 1.1 First" pageUUID="page-1" />
                  <section title="Section 1.2 Second" pageUUID="page-2" />
                  <section title="Section 1.4 Closing" pageUUID="page-3" />
                  <section title="Section 1.5 Extra Body Section" pageUUID="page-4" />
                  <section title="Section 1.6 Post-Closing" pageUUID="page-5" />
                </article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-6" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-7" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-8" /></article>
                <article title="ARTICLE V"><section title="Section 5.1 Fifth" pageUUID="page-9" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        self.assertFalse(any(v.reason_code == XML_REASON_SECTION_NON_SEQUENTIAL for v in violations))

    def test_find_hard_rule_violations_suppresses_duplicate_when_toc_matches_body(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <tableOfContents>
                <text>12.1 Intro</text>
                <text>12.1 Duplicate Intro</text>
                <text>12.2 Closing</text>
              </tableOfContents>
              <body>
                <article title="ARTICLE XII">
                  <section title="Section 12.1 Intro" pageUUID="page-1" />
                  <section title="Section 12.1 Duplicate Intro" pageUUID="page-2" />
                  <section title="Section 12.2 Closing" pageUUID="page-3" />
                </article>
                <article title="ARTICLE XIII"><section title="Section 13.1 Second" pageUUID="page-5" /></article>
                <article title="ARTICLE XIV"><section title="Section 14.1 Third" pageUUID="page-6" /></article>
                <article title="ARTICLE XV"><section title="Section 15.1 Fourth" pageUUID="page-7" /></article>
                <article title="ARTICLE XVI"><section title="Section 16.1 Fifth" pageUUID="page-8" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        self.assertFalse(any(v.reason_code == XML_REASON_SECTION_NON_SEQUENTIAL for v in violations))

    def test_find_hard_rule_violations_keeps_gap_when_toc_conflicts(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <tableOfContents>
                <text>1.1 First</text>
                <text>1.2 Second</text>
                <text>1.3 Third</text>
                <text>1.4 Closing</text>
              </tableOfContents>
              <body>
                <article title="ARTICLE I">
                  <section title="Section 1.1 First" pageUUID="page-1" />
                  <section title="Section 1.2 Second" pageUUID="page-2" />
                  <section title="Section 1.4 Closing" pageUUID="page-3" />
                  <text>See Section 1.3 elsewhere.</text>
                </article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-5" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-6" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-7" /></article>
                <article title="ARTICLE V"><section title="Section 5.1 Fifth" pageUUID="page-8" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        self.assertTrue(any(v.reason_code == XML_REASON_SECTION_NON_SEQUENTIAL for v in violations))

    def test_find_hard_rule_violations_keeps_gap_when_toc_is_partial(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <tableOfContents>
                <text>1.1 First</text>
                <text>1.2 Second</text>
              </tableOfContents>
              <body>
                <article title="ARTICLE I">
                  <section title="Section 1.1 First" pageUUID="page-1" />
                  <section title="Section 1.2 Second" pageUUID="page-2" />
                  <section title="Section 1.4 Closing" pageUUID="page-3" />
                  <text>See Section 1.3 elsewhere.</text>
                </article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-5" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-6" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-7" /></article>
                <article title="ARTICLE V"><section title="Section 5.1 Fifth" pageUUID="page-8" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        self.assertTrue(any(v.reason_code == XML_REASON_SECTION_NON_SEQUENTIAL for v in violations))

    def test_section_non_sequential_targets_absorbed_heading_page_after_boundary(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE I">
                  <section title="Section 1.1 First" pageUUID="page-1" />
                  <section title="Section 1.2 Second" pageUUID="page-1">
                    <text>Second section body.</text>
                    <page>1</page>
                    <pageUUID>page-1</pageUUID>
                    <text>Section 1.3 Missing heading absorbed into prior section.</text>
                    <page>2</page>
                    <pageUUID>page-absorbed-heading</pageUUID>
                  </section>
                  <section title="Section 1.4 Fourth" pageUUID="page-2" />
                </article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-3" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-4" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-5" /></article>
                <article title="ARTICLE V"><section title="Section 5.1 Fifth" pageUUID="page-6" /></article>
              </body>
            </document>
            """
        )

        violations = [
            violation
            for violation in find_hard_rule_violations(root)
            if violation.reason_code == XML_REASON_SECTION_NON_SEQUENTIAL
        ]

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].page_uuids, ("page-absorbed-heading",))

    def test_section_non_sequential_targets_zero_padded_absorbed_heading_page(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE VI"><section title="Section 6.1 First" pageUUID="page-6" /></article>
                <article title="ARTICLE VII"><section title="Section 7.1 First" pageUUID="page-7" /></article>
                <article title="ARTICLE VIII"><section title="Section 8.1 First" pageUUID="page-8" /></article>
                <article title="ARTICLE IX"><section title="Section 9.1 First" pageUUID="page-9" /></article>
                <article title="ARTICLE X">
                  <section title="Section 10.01 First" pageUUID="page-1" />
                  <section title="Section 10.02 Second" pageUUID="page-1">
                    <text>Second section body.</text>
                    <page>75</page>
                    <pageUUID>page-absorbed-heading</pageUUID>
                    <text>Section 10.03 Missing zero-padded heading absorbed into prior section.</text>
                  </section>
                  <section title="Section 10.04 Fourth" pageUUID="page-2" />
                </article>
              </body>
            </document>
            """
        )

        violations = [
            violation
            for violation in find_hard_rule_violations(root)
            if violation.reason_code == XML_REASON_SECTION_NON_SEQUENTIAL
        ]

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].page_uuids, ("page-absorbed-heading",))

    def test_section_non_sequential_targets_previous_page_when_missing_heading_is_before_page_boundary(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE I">
                  <section title="Section 1.1 First" pageUUID="page-1" />
                  <section title="Section 1.2 Second" pageUUID="page-1">
                    <text>Section 1.3 Missing heading absorbed into prior section.</text>
                    <page>1</page>
                    <pageUUID>page-1</pageUUID>
                  </section>
                  <section title="Section 1.4 Fourth" pageUUID="page-2" />
                </article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-3" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-4" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-5" /></article>
                <article title="ARTICLE V"><section title="Section 5.1 Fifth" pageUUID="page-6" /></article>
              </body>
            </document>
            """
        )

        violations = [
            violation
            for violation in find_hard_rule_violations(root)
            if violation.reason_code == XML_REASON_SECTION_NON_SEQUENTIAL
        ]

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].page_uuids, ("page-1",))

    def test_xml_verify_instructions_include_toc_exception(self) -> None:
        self.assertIn("tableOfContents", XML_VERIFY_INSTRUCTIONS)
        self.assertIn("same local section-number jump", XML_VERIFY_INSTRUCTIONS)
        self.assertIn("expected/skipped section number is absent from the body", XML_VERIFY_INSTRUCTIONS)

    def test_xml_verify_toc_context_summarizes_numbering_for_llm_input(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <tableOfContents>
                <text>5.20 Intellectual Property</text>
                <text>5.21 [Reserved]</text>
                <text>5.22 Customers and Suppliers</text>
                <text>5.23 Accounts Receivable and Payable; Loans</text>
                <text>6.1 Corporate Existence and Power</text>
              </tableOfContents>
              <body>
                <article title="ARTICLE V">
                  <section title="Section 5.20 Intellectual Property" pageUUID="page-1" />
                  <section title="Section 5.22 Customers and Suppliers" pageUUID="page-2" />
                  <section title="Section 5.23 Accounts Receivable and Payable; Loans" pageUUID="page-3" />
                </article>
              </body>
            </document>
            """
        )

        toc_context = _build_xml_verify_toc_context(root)

        self.assertEqual(
            toc_context,
            "\n".join(
                [
                    "XML tableOfContents numbering:",
                    "Article 5: 5.20, 5.21, 5.22, 5.23",
                    "Article 6: 6.1",
                ]
            ),
        )

    def test_xml_verify_request_body_includes_toc_context_when_available(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <tableOfContents>
                <text>5.20 Intellectual Property</text>
                <text>5.21 [Reserved]</text>
                <text>5.22 Customers and Suppliers</text>
              </tableOfContents>
              <body>
                <article title="ARTICLE V">
                  <section title="Section 5.20 Intellectual Property" pageUUID="page-1" />
                  <section title="Section 5.22 Customers and Suppliers" pageUUID="page-2" />
                </article>
              </body>
            </document>
            """
        )

        body = _build_xml_verify_batch_request_body(
            custom_id="agreement|1",
            tag_tree=_render_tag_tree_from_root(root),
            model="gpt-5.4-mini",
            toc_context=_build_xml_verify_toc_context(root),
        )["body"]

        self.assertIn("XML tag tree:", str(body["input"]))
        self.assertIn("XML tableOfContents numbering:", str(body["input"]))
        self.assertIn("Article 5: 5.20, 5.21, 5.22", str(body["input"]))

    def test_reason_rows_changed_ignores_order(self) -> None:
        existing = [
            {"reason_code": "section_non_sequential", "reason_detail": "gap", "page_uuid": "page-2"},
            {"reason_code": "section_article_mismatch", "reason_detail": "mismatch", "page_uuid": "page-3"},
        ]
        new = [
            {"reason_code": "section_article_mismatch", "reason_detail": "mismatch", "page_uuid": "page-3"},
            {"reason_code": "section_non_sequential", "reason_detail": "gap", "page_uuid": "page-2"},
        ]

        self.assertFalse(_reason_rows_changed(existing, new))

    def test_reason_rows_changed_detects_page_target_change(self) -> None:
        existing = [
            {"reason_code": "section_non_sequential", "reason_detail": "gap", "page_uuid": "page-2"},
        ]
        new = [
            {"reason_code": "section_non_sequential", "reason_detail": "gap", "page_uuid": "page-3"},
        ]

        self.assertTrue(_reason_rows_changed(existing, new))

    def test_apply_xml_verify_batch_output_sets_status_source_to_asset(self) -> None:
        class _FakeContent:
            def __init__(self, text_value: str) -> None:
                self._text_value = text_value

            def text(self) -> str:
                return self._text_value

        class _FakeFiles:
            def __init__(self, text_value: str) -> None:
                self._text_value = text_value

            def content(self, _file_id: str) -> _FakeContent:
                return _FakeContent(self._text_value)

        class _FakeClient:
            def __init__(self, text_value: str) -> None:
                self.files = _FakeFiles(text_value)

        class _FakeResult:
            def __init__(
                self,
                rowcount: int,
                rows: list[dict[str, object]] | None = None,
            ) -> None:
                self.rowcount = rowcount
                self._rows = rows or []

            class _Mappings:
                def __init__(self, rows: list[dict[str, object]]) -> None:
                    self._rows = rows

                def fetchall(self) -> list[dict[str, object]]:
                    return self._rows

            def mappings(self):
                return _FakeResult._Mappings(self._rows)

        class _FakeConn:
            def __init__(self) -> None:
                self.executed: list[tuple[str, dict[str, object]]] = []

            def execute(self, statement: object, params: dict[str, object]) -> _FakeResult:
                sql = str(statement)
                self.executed.append((sql, params))
                if "SELECT reason_code, reason_detail, page_uuid" in sql:
                    return _FakeResult(0, rows=[])
                return _FakeResult(1)

        class _BeginContext:
            def __init__(self, conn: _FakeConn) -> None:
                self._conn = conn

            def __enter__(self) -> _FakeConn:
                return self._conn

            def __exit__(self, *_exc: object) -> None:
                return None

        class _FakeEngine:
            def __init__(self, conn: _FakeConn) -> None:
                self._conn = conn

            def begin(self) -> _BeginContext:
                return _BeginContext(self._conn)

        response_payload = {
            "custom_id": "agreement-1|3",
            "response": {
                "status_code": 200,
                "body": {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"text": '{"status":"verified"}'}],
                        }
                    ]
                },
            },
        }
        out_text = json.dumps(response_payload)
        conn = _FakeConn()
        engine = _FakeEngine(conn)
        client = _FakeClient(out_text)

        class _FakeLog:
            def info(self, *_args: object, **_kwargs: object) -> None:
                return None

            def warning(self, *_args: object, **_kwargs: object) -> None:
                return None

        context = SimpleNamespace(
            log=_FakeLog()
        )
        batch = SimpleNamespace(output_file_id="file-1")

        updated, parse_errors = _apply_xml_verify_batch_output(
            context=cast(AssetExecutionContext, cast(object, context)),
            engine=engine,
            client=cast(OpenAI, cast(object, client)),
            xml_table="pdx.xml",
            xml_status_reasons_table="pdx.xml_status_reasons",
            batch=batch,
        )

        self.assertEqual(updated, 1)
        self.assertEqual(parse_errors, 0)
        self.assertEqual(len(conn.executed), 3)
        select_sql, _ = conn.executed[0]
        self.assertIn("SELECT reason_code, reason_detail, page_uuid", select_sql)
        executed_sql, params = conn.executed[1]
        self.assertIn("status_source = 'asset'", executed_sql)
        self.assertIn("status_reason_code = :reason_code", executed_sql)
        self.assertIn("status_reason_detail = :reason_detail", executed_sql)
        self.assertEqual(params["agreement_uuid"], "agreement-1")
        self.assertEqual(params["version"], 3)
        self.assertEqual(params["status"], "verified")
        self.assertIsNone(params["reason_code"])
        self.assertIsNone(params["reason_detail"])
        delete_sql, _ = conn.executed[2]
        self.assertIn("DELETE FROM pdx.xml_status_reasons", delete_sql)

    def test_apply_xml_verify_batch_output_sets_llm_invalid_reason_code(self) -> None:
        class _FakeContent:
            def __init__(self, text_value: str) -> None:
                self._text_value = text_value

            def text(self) -> str:
                return self._text_value

        class _FakeFiles:
            def __init__(self, text_value: str) -> None:
                self._text_value = text_value

            def content(self, _file_id: str) -> _FakeContent:
                return _FakeContent(self._text_value)

        class _FakeClient:
            def __init__(self, text_value: str) -> None:
                self.files = _FakeFiles(text_value)

        class _FakeResult:
            def __init__(
                self,
                rowcount: int,
                rows: list[dict[str, object]] | None = None,
            ) -> None:
                self.rowcount = rowcount
                self._rows = rows or []

            class _Mappings:
                def __init__(self, rows: list[dict[str, object]]) -> None:
                    self._rows = rows

                def fetchall(self) -> list[dict[str, object]]:
                    return self._rows

            def mappings(self):
                return _FakeResult._Mappings(self._rows)

        class _FakeConn:
            def __init__(self) -> None:
                self.executed: list[tuple[str, dict[str, object]]] = []

            def execute(self, statement: object, params: dict[str, object]) -> _FakeResult:
                sql = str(statement)
                self.executed.append((sql, params))
                if "SELECT reason_code, reason_detail, page_uuid" in sql:
                    return _FakeResult(0, rows=[])
                return _FakeResult(1)

        class _BeginContext:
            def __init__(self, conn: _FakeConn) -> None:
                self._conn = conn

            def __enter__(self) -> _FakeConn:
                return self._conn

            def __exit__(self, *_exc: object) -> None:
                return None

        class _FakeEngine:
            def __init__(self, conn: _FakeConn) -> None:
                self._conn = conn

            def begin(self) -> _BeginContext:
                return _BeginContext(self._conn)

        response_payload = {
            "custom_id": "agreement-2|4",
            "response": {
                "status_code": 200,
                "body": {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"text": '{"status":"invalid"}'}],
                        }
                    ]
                },
            },
        }
        out_text = json.dumps(response_payload)
        conn = _FakeConn()
        engine = _FakeEngine(conn)
        client = _FakeClient(out_text)

        class _FakeLog:
            def info(self, *_args: object, **_kwargs: object) -> None:
                return None

            def warning(self, *_args: object, **_kwargs: object) -> None:
                return None

        context = SimpleNamespace(log=_FakeLog())
        batch = SimpleNamespace(output_file_id="file-2")

        updated, parse_errors = _apply_xml_verify_batch_output(
            context=cast(AssetExecutionContext, cast(object, context)),
            engine=engine,
            client=cast(OpenAI, cast(object, client)),
            xml_table="pdx.xml",
            xml_status_reasons_table="pdx.xml_status_reasons",
            batch=batch,
        )

        self.assertEqual(updated, 1)
        self.assertEqual(parse_errors, 0)
        self.assertEqual(len(conn.executed), 5)
        select_sql, _ = conn.executed[0]
        self.assertIn("SELECT reason_code, reason_detail, page_uuid", select_sql)
        _, params = conn.executed[1]
        self.assertEqual(params["agreement_uuid"], "agreement-2")
        self.assertEqual(params["version"], 4)
        self.assertEqual(params["status"], "invalid")
        self.assertEqual(params["reason_code"], XML_REASON_LLM_INVALID)
        self.assertIsNone(params["reason_detail"])
        insert_sql, insert_params = conn.executed[3]
        self.assertIn("INSERT INTO pdx.xml_status_reasons", insert_sql)
        self.assertEqual(insert_params["reason_code"], XML_REASON_LLM_INVALID)
        reset_sql, _ = conn.executed[4]
        self.assertIn("SET ai_repair_attempted = 0", reset_sql)

    def test_xml_verify_asset_hard_invalid_sets_status_source_to_asset(self) -> None:
        class _FakeLog:
            def info(self, *_args: object, **_kwargs: object) -> None:
                return None

            def warning(self, *_args: object, **_kwargs: object) -> None:
                return None

        class _Result:
            def __init__(
                self,
                *,
                rows: list[dict[str, object]] | None = None,
                rowcount: int = 0,
            ) -> None:
                self._rows = rows or []
                self.rowcount = rowcount

            class _Scalars:
                def __init__(self, values: list[object]) -> None:
                    self._values = values

                def all(self) -> list[object]:
                    return self._values

            def mappings(self) -> "_Result":
                return self

            def fetchall(self) -> list[dict[str, object]]:
                return self._rows

            def scalars(self) -> "_Result._Scalars":
                values: list[object] = []
                for row in self._rows:
                    first = next(iter(row.values()), None)
                    values.append(first)
                return _Result._Scalars(values)

        class _FakeConn:
            def __init__(self) -> None:
                self.executed: list[tuple[str, dict[str, object]]] = []

            def execute(
                self,
                statement: object,
                params: dict[str, object] | None = None,
            ) -> _Result:
                sql = str(statement)
                query_params = params or {}
                self.executed.append((sql, query_params))
                if "FROM state_components" in sql and "latest_xml_status IS NULL" in sql:
                    return _Result(rows=[{"agreement_uuid": "agreement-hard-invalid"}])
                if "SELECT agreement_uuid, version, xml" in sql:
                    return _Result(
                        rows=[
                            {
                                "agreement_uuid": "agreement-hard-invalid",
                                "version": 1,
                                "xml": "<document><body><section title='bad'/></body></document>",
                            }
                        ]
                    )
                return _Result(rowcount=1)

        class _BeginContext:
            def __init__(self, conn: _FakeConn) -> None:
                self._conn = conn

            def __enter__(self) -> _FakeConn:
                return self._conn

            def __exit__(self, *_exc: object) -> None:
                return None

        class _FakeEngine:
            def __init__(self, conn: _FakeConn) -> None:
                self._conn = conn

            def begin(self) -> _BeginContext:
                return _BeginContext(self._conn)

        conn = _FakeConn()
        engine = _FakeEngine(conn)
        db = SimpleNamespace(database="pdx", get_engine=lambda: engine)
        pipeline_config = SimpleNamespace(
            xml_agreement_batch_size=10,
            resume_openai_batches=True,
        )
        context = SimpleNamespace(log=_FakeLog())

        with (
            patch("etl.defs.f_xml_asset._oai_client", return_value=SimpleNamespace()),
            patch("etl.defs.f_xml_asset.assert_tables_exist", return_value=None),
            patch("etl.defs.f_xml_asset._fetch_unpulled_xml_verify_batch", return_value=None),
            patch("etl.defs.f_xml_asset.run_post_asset_refresh", return_value=None),
        ):
            _ = xml_verify_asset.node_def.compute_fn.decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                cast(PipelineConfig, cast(object, pipeline_config)),
                ["agreement-hard-invalid"],
            )

        hard_invalid_update_sql, hard_invalid_params = next(
            (sql, params)
            for sql, params in conn.executed
            if "status_reason_code = :reason_code" in sql
        )
        self.assertIn("status_source = 'asset'", hard_invalid_update_sql)
        self.assertIn("status_source <=> 'asset'", hard_invalid_update_sql)
        self.assertIn("status_reason_code = :reason_code", hard_invalid_update_sql)
        self.assertIn("status_reason_detail = :reason_detail", hard_invalid_update_sql)
        self.assertEqual(
            hard_invalid_params["reason_code"], XML_REASON_BODY_STARTS_NON_ARTICLE
        )
        self.assertIn(
            "<body> must start with <article>.",
            str(hard_invalid_params["reason_detail"]),
        )
        self.assertTrue(
            any("INSERT INTO pdx.xml_status_reasons" in sql for sql, _ in conn.executed)
        )

    def test_xml_verify_asset_bypasses_llm_when_disabled_and_marks_verified(self) -> None:
        class _FakeLog:
            def info(self, *_args: object, **_kwargs: object) -> None:
                return None

            def warning(self, *_args: object, **_kwargs: object) -> None:
                return None

        class _Result:
            def __init__(
                self,
                *,
                rows: list[dict[str, object]] | None = None,
                rowcount: int = 0,
            ) -> None:
                self._rows = rows or []
                self.rowcount = rowcount

            class _Scalars:
                def __init__(self, values: list[object]) -> None:
                    self._values = values

                def all(self) -> list[object]:
                    return self._values

            def mappings(self) -> "_Result":
                return self

            def fetchall(self) -> list[dict[str, object]]:
                return self._rows

            def scalars(self) -> "_Result._Scalars":
                values: list[object] = []
                for row in self._rows:
                    first = next(iter(row.values()), None)
                    values.append(first)
                return _Result._Scalars(values)

        class _FakeConn:
            def __init__(self) -> None:
                self.executed: list[tuple[str, dict[str, object]]] = []

            def execute(
                self,
                statement: object,
                params: dict[str, object] | None = None,
            ) -> _Result:
                sql = str(statement)
                query_params = params or {}
                self.executed.append((sql, query_params))
                if "FROM state_components" in sql and "latest_xml_status IS NULL" in sql:
                    return _Result(rows=[{"agreement_uuid": "agreement-direct-verify"}])
                if "SELECT agreement_uuid, version, xml" in sql:
                    return _Result(
                        rows=[
                            {
                                "agreement_uuid": "agreement-direct-verify",
                                "version": 2,
                                "xml": (
                                    "<document><body>"
                                    "<article title='ARTICLE I'><section title='Section 1.1 First' pageUUID='page-1' /></article>"
                                    "<article title='ARTICLE II'><section title='Section 2.1 Second' pageUUID='page-2' /></article>"
                                    "<article title='ARTICLE III'><section title='Section 3.1 Third' pageUUID='page-3' /></article>"
                                    "<article title='ARTICLE IV'><section title='Section 4.1 Fourth' pageUUID='page-4' /></article>"
                                    "<article title='ARTICLE V'><section title='Section 5.1 Fifth' pageUUID='page-5' /></article>"
                                    "</body></document>"
                                ),
                            }
                        ]
                    )
                if "SELECT reason_code, reason_detail, page_uuid" in sql:
                    return _Result(rows=[])
                return _Result(rowcount=1)

        class _BeginContext:
            def __init__(self, conn: _FakeConn) -> None:
                self._conn = conn

            def __enter__(self) -> _FakeConn:
                return self._conn

            def __exit__(self, *_exc: object) -> None:
                return None

        class _FakeEngine:
            def __init__(self, conn: _FakeConn) -> None:
                self._conn = conn

            def begin(self) -> _BeginContext:
                return _BeginContext(self._conn)

        conn = _FakeConn()
        engine = _FakeEngine(conn)
        db = SimpleNamespace(database="pdx", get_engine=lambda: engine)
        pipeline_config = SimpleNamespace(
            xml_agreement_batch_size=10,
            resume_openai_batches=True,
            xml_enable_llm_verification=False,
        )
        context = SimpleNamespace(log=_FakeLog())

        with (
            patch(
                "etl.defs.f_xml_asset._oai_client",
                side_effect=AssertionError("LLM client should not be created when XML verification is disabled"),
            ),
            patch("etl.defs.f_xml_asset.assert_tables_exist", return_value=None),
            patch("etl.defs.f_xml_asset.run_post_asset_refresh", return_value=None),
        ):
            result = xml_verify_asset.node_def.compute_fn.decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                cast(PipelineConfig, cast(object, pipeline_config)),
                ["agreement-direct-verify"],
            )

        self.assertEqual(result, ["agreement-direct-verify"])
        verified_update_sql, verified_params = next(
            (sql, params)
            for sql, params in conn.executed
            if "status_reason_code = :reason_code" in sql and params.get("status") == "verified"
        )
        self.assertIn("status_source = 'asset'", verified_update_sql)
        self.assertEqual(verified_params["agreement_uuid"], "agreement-direct-verify")
        self.assertEqual(verified_params["version"], 2)
        self.assertEqual(verified_params["status"], "verified")
        self.assertIsNone(verified_params["reason_code"])
        self.assertIsNone(verified_params["reason_detail"])
        self.assertFalse(
            any("INSERT INTO pdx.xml_verify_batches" in sql for sql, _ in conn.executed)
        )


if __name__ == "__main__":
    _ = unittest.main()
