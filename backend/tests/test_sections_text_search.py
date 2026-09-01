from __future__ import annotations

import unittest
from typing import cast
from unittest.mock import MagicMock

from marshmallow import ValidationError
from sqlalchemy import Column, MetaData, Table, Text, select
from sqlalchemy.dialects import mysql

from backend.services.sections_service import (
    apply_section_text_search,
    build_section_text_match_expression,
    sections_total_count_metadata,
)
from backend.schemas.sections import SectionsArgsSchema
from backend.text_search import (
    MAX_ANY_TEXT_SEARCH_TERMS,
    MAX_TEXT_PREFIX_TERMS,
    MAX_TEXT_QUERY_CHARS,
    MAX_TEXT_SEARCH_TERMS,
    compile_boolean_text_query,
    compile_phrase_text_pattern,
    text_query_focus_terms,
)


class CompileBooleanTextQueryTests(unittest.TestCase):
    def test_compiles_each_supported_match_mode(self) -> None:
        self.assertEqual(
            compile_boolean_text_query("material adverse effect", "phrase"),
            "+material +adverse +effect",
        )
        self.assertEqual(
            compile_boolean_text_query("material adverse eff*", "all_terms"),
            "+material +adverse +eff*",
        )
        self.assertEqual(
            compile_boolean_text_query("material adverse eff*", "any_terms"),
            "material adverse eff*",
        )

    def test_normalizes_like_the_etl_and_does_not_forward_boolean_syntax(self) -> None:
        self.assertEqual(
            compile_boolean_text_query(
                '  ＭＡＴＥＲＩＡＬ   +(Adverse) -"EFFECT"  ',
                "all_terms",
            ),
            "+material +adverse +effect",
        )

    def test_keeps_phrase_duplicates_but_deduplicates_term_modes(self) -> None:
        self.assertEqual(
            compile_boolean_text_query("had had effect", "phrase"),
            "+had +effect",
        )
        self.assertEqual(
            compile_boolean_text_query("effect effect", "all_terms"),
            "+effect",
        )

    def test_rejects_blank_and_punctuation_only_queries(self) -> None:
        for query in ("", "   ", '+ - () "'):
            with self.subTest(query=query), self.assertRaises(ValueError):
                compile_boolean_text_query(query, "phrase")

    def test_rejects_invalid_or_overly_broad_prefixes(self) -> None:
        invalid_queries = (
            "*sandbag",
            "sand*bag",
            "sand**",
            "ab*",
            " ".join(f"prefix{index}*" for index in range(MAX_TEXT_PREFIX_TERMS + 1)),
        )
        for query in invalid_queries:
            with self.subTest(query=query), self.assertRaises(ValueError):
                compile_boolean_text_query(query, "any_terms")

    def test_enforces_query_and_term_limits(self) -> None:
        with self.assertRaises(ValueError):
            compile_boolean_text_query("x" * (MAX_TEXT_QUERY_CHARS + 1), "phrase")
        with self.assertRaises(ValueError):
            compile_boolean_text_query(
                " ".join(f"term{index}" for index in range(MAX_TEXT_SEARCH_TERMS + 1)),
                "any_terms",
            )
        with self.assertRaises(ValueError):
            compile_boolean_text_query(
                " ".join(
                    f"optional{index}"
                    for index in range(MAX_ANY_TEXT_SEARCH_TERMS + 1)
                ),
                "any_terms",
            )

    def test_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            compile_boolean_text_query("material", "natural_language")

    def test_phrase_pattern_preserves_order_and_supports_prefixes(self) -> None:
        self.assertEqual(
            compile_phrase_text_pattern("material adverse eff*"),
            "[[:<:]]material[^[:alnum:]_]+adverse"
            "[^[:alnum:]_]+eff[[:alnum:]_]*[[:>:]]",
        )

    def test_phrase_pattern_requires_whole_token_boundaries(self) -> None:
        pattern = compile_phrase_text_pattern("no shop")
        self.assertEqual(
            pattern,
            "[[:<:]]no[^[:alnum:]_]+shop[[:>:]]",
        )
        self.assertNotEqual(pattern, "no[^[:alnum:]_]+shop")

    def test_short_terms_require_phrase_context(self) -> None:
        self.assertEqual(
            compile_boolean_text_query("no shop", "phrase"),
            "+shop",
        )
        for mode in ("all_terms", "any_terms"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                compile_boolean_text_query("no shop", mode)
        with self.assertRaises(ValueError):
            compile_boolean_text_query("no go", "phrase")
        self.assertEqual(
            compile_boolean_text_query("the agreement", "phrase"),
            "+agreement",
        )
        with self.assertRaises(ValueError):
            compile_boolean_text_query("the agreement", "all_terms")

    def test_snippet_terms_match_the_sanitized_fulltext_query(self) -> None:
        self.assertEqual(
            text_query_focus_terms(' +(Adverse)  EFFECT* ', "phrase"),
            ["adverse effect"],
        )
        self.assertEqual(
            text_query_focus_terms("effect* adverse effect*", "all_terms"),
            ["effect", "adverse"],
        )


class SectionTextMatchExpressionTests(unittest.TestCase):
    def test_mysql_fulltext_query_is_parameterized(self) -> None:
        metadata = MetaData()
        text_search = Table(
            "section_text_search",
            metadata,
            Column("normalized_text", Text),
        )
        statement = select(text_search.c.normalized_text).where(
            build_section_text_match_expression(
                text_search.c.normalized_text,
                text_query='material" operator one=one -- eff*',
                match_mode="all_terms",
            )
        )

        compiled = statement.compile(dialect=mysql.dialect())
        sql = str(compiled)
        self.assertIn("MATCH (section_text_search.normalized_text)", sql)
        self.assertIn("AGAINST (%s IN BOOLEAN MODE)", sql)
        self.assertNotIn("material", sql)
        self.assertEqual(
            list(compiled.params.values()),
            ["+material +operator +one +eff*"],
        )

    def test_phrase_uses_indexed_candidates_and_parameterized_order_filter(self) -> None:
        metadata = MetaData()
        text_search = Table(
            "section_text_search",
            metadata,
            Column("normalized_text", Text),
        )
        statement = select(text_search.c.normalized_text).where(
            build_section_text_match_expression(
                text_search.c.normalized_text,
                text_query="no shop*",
                match_mode="phrase",
            )
        )

        compiled = statement.compile(dialect=mysql.dialect())
        sql = str(compiled)
        self.assertIn("MATCH (section_text_search.normalized_text)", sql)
        self.assertIn("REGEXP", sql)
        self.assertNotIn("no shop", sql)
        self.assertEqual(
            list(compiled.params.values()),
            [
                "+shop*",
                "[[:<:]]no[^[:alnum:]_]+shop[[:alnum:]_]*[[:>:]]",
            ],
        )

    def test_does_not_join_text_table_without_a_query(self) -> None:
        query = MagicMock()
        result = apply_section_text_search(
            query,
            latest=MagicMock(),
            section_text_search=MagicMock(),
            text_query=None,
            match_mode="phrase",
        )
        self.assertIs(result, query)
        query.join.assert_not_called()
        query.filter.assert_not_called()

    def test_rejects_blank_query_before_joining(self) -> None:
        query = MagicMock()
        with self.assertRaises(ValueError):
            apply_section_text_search(
                query,
                latest=MagicMock(),
                section_text_search=MagicMock(),
                text_query="   ",
                match_mode="phrase",
            )
        query.join.assert_not_called()


class SectionTextCountTests(unittest.TestCase):
    def test_text_auto_count_prefers_a_credible_planner_estimate(self) -> None:
        deps = MagicMock()
        deps._estimated_query_row_count.return_value = 125

        result = sections_total_count_metadata(
            deps,
            query=MagicMock(),
            page=1,
            page_size=10,
            item_count=10,
            has_next=True,
            has_filters=True,
            count_mode="auto",
            count_cache_key="text-search",
            prefer_estimate=True,
        )

        self.assertEqual(result, (125, True, "table_estimate"))
        deps._cached_exact_query_count.assert_not_called()

    def test_auto_count_uses_exact_count_instead_of_a_page_lower_bound(self) -> None:
        deps = MagicMock()
        deps._estimated_query_row_count.return_value = 10
        deps._cached_exact_query_count.return_value = 37

        result = sections_total_count_metadata(
            deps,
            query=MagicMock(),
            page=3,
            page_size=10,
            item_count=10,
            has_next=True,
            has_filters=True,
            count_mode="auto",
            count_cache_key="filtered-search",
        )

        self.assertEqual(result, (37, False, "query_count"))
        deps._cached_exact_query_count.assert_called_once()


class SectionTextSchemaTests(unittest.TestCase):
    def test_text_search_rejects_unbounded_deep_pages(self) -> None:
        with self.assertRaisesRegex(ValidationError, "pages up to 100"):
            _ = SectionsArgsSchema().load(
                {"text_query": "material adverse effect", "page": 101}
            )

    def test_non_text_section_listing_keeps_existing_page_behavior(self) -> None:
        loaded = cast(dict[str, object], SectionsArgsSchema().load({"page": 101}))
        self.assertEqual(loaded["page"], 101)


if __name__ == "__main__":
    unittest.main()
