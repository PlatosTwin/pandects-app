from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

os.environ.setdefault("SKIP_MAIN_DB_REFLECTION", "1")

from flask import Flask
from marshmallow import ValidationError
from sqlalchemy import Column, MetaData, Table, Text, create_engine, event, select
from sqlalchemy.dialects import mysql
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from backend.core.config import bounded_statement, configure_auth_bind, configure_main_db
from backend.models.main_db import LatestSectionsSearch, SectionTextSearch
from backend.search_counts import build_search_count_cache_key
from backend.services.sections_service import (
    SECTION_TEXT_SEARCH_UNAVAILABLE_MESSAGE,
    TEXT_QUERY_TOO_EXPENSIVE_MESSAGE,
    apply_section_text_candidate_search,
    apply_section_text_search,
    build_section_text_match_expression,
    is_statement_timeout_error,
    run_sections,
    sections_total_count_metadata,
)
from backend.schemas.sections import SectionsArgsPayload, SectionsArgsSchema
from backend.text_search import (
    MAX_ANY_TEXT_SEARCH_TERMS,
    MAX_TEXT_PREFIX_TERMS,
    MAX_TEXT_QUERY_CHARS,
    MAX_TEXT_SEARCH_TERMS,
    TEXT_SEARCH_MAX_STATEMENT_SECONDS,
    compile_boolean_text_query,
    compile_phrase_text_pattern,
    text_query_focus_terms,
)


def _chained_query() -> MagicMock:
    query = MagicMock(name="query")
    for method in ("execution_options", "join", "filter", "order_by", "offset", "limit", "with_entities"):
        getattr(query, method).return_value = query
    query.all.return_value = []
    return query


def _service_deps(query: MagicMock, *, text_search_available: bool = True) -> MagicMock:
    deps = MagicMock(name="deps")
    deps.db.session.query.return_value = query
    deps.LatestSectionsSearch = LatestSectionsSearch
    deps.SectionTextSearch = SectionTextSearch if text_search_available else None
    deps._dedupe_preserve_order = lambda values: values
    deps._to_int = lambda value, default=0: int(cast(int, value))
    deps._row_mapping_as_dict = lambda row: dict(cast(dict[str, object], row))
    deps._pagination_metadata.return_value = {}
    deps._estimated_latest_sections_search_table_rows.return_value = 0
    return deps


def _parsed_args(**overrides: object) -> SectionsArgsPayload:
    loaded = SectionsArgsSchema().load(overrides)
    return cast(SectionsArgsPayload, cast(object, loaded))


_CTX = SimpleNamespace(is_authenticated=True, tier="test")


def _statement_timeout(statement: str = "SELECT 1") -> OperationalError:
    return OperationalError(
        statement,
        {},
        Exception(1969, "Query execution was interrupted (max_statement_time exceeded)"),
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


class SectionTextSearchAvailabilityTests(unittest.TestCase):
    def test_text_query_is_rejected_when_index_table_is_unavailable(self) -> None:
        query = _chained_query()
        deps = _service_deps(query, text_search_available=False)

        with self.assertRaises(ValidationError) as caught:
            _ = run_sections(
                deps,
                ctx=_CTX,
                parsed_args=_parsed_args(text_query="material adverse effect"),
            )

        self.assertEqual(
            caught.exception.messages,
            {"text_query": [SECTION_TEXT_SEARCH_UNAVAILABLE_MESSAGE]},
        )
        query.join.assert_not_called()
        query.all.assert_not_called()

    def test_non_text_search_ignores_unavailable_index_table(self) -> None:
        query = _chained_query()
        deps = _service_deps(query, text_search_available=False)

        response = run_sections(deps, ctx=_CTX, parsed_args=_parsed_args(year=[2020]))

        self.assertEqual(response["results"], [])
        query.join.assert_not_called()
        query.all.assert_called_once()

    def test_candidate_search_rejects_unavailable_index_before_joining(self) -> None:
        query = MagicMock()
        with self.assertRaisesRegex(ValidationError, "not available"):
            _ = apply_section_text_candidate_search(
                query,
                latest=MagicMock(),
                section_text_search=None,
                text_query="material adverse effect",
                match_mode="phrase",
            )
        query.join.assert_not_called()


class StatementTimeoutTests(unittest.TestCase):
    def test_text_search_queries_carry_the_statement_timeout_option(self) -> None:
        session = Session(create_engine("sqlite://"))
        query = apply_section_text_candidate_search(
            session.query(LatestSectionsSearch.section_uuid),
            latest=LatestSectionsSearch,
            section_text_search=SectionTextSearch,
            text_query="material adverse effect",
            match_mode="all_terms",
        )

        self.assertEqual(
            query.get_execution_options()["max_statement_time"],
            TEXT_SEARCH_MAX_STATEMENT_SECONDS,
        )
        compiled_sql = str(query.statement.compile(dialect=mysql.dialect()))
        self.assertIn("INNER JOIN __main_schema__.section_text_search", compiled_sql)
        self.assertIn("AGAINST (%s IN BOOLEAN MODE)", compiled_sql)

    def test_bounded_statement_wraps_mysql_statements_that_opt_in(self) -> None:
        mysql_connection = SimpleNamespace(dialect=SimpleNamespace(name="mysql"))
        context = SimpleNamespace(execution_options={"max_statement_time": 20})

        statement, parameters = bounded_statement(
            mysql_connection, None, "SELECT 1", ("p",), context, False
        )

        self.assertEqual(statement, "SET STATEMENT max_statement_time=20 FOR SELECT 1")
        self.assertEqual(parameters, ("p",))

    def test_bounded_statement_leaves_other_statements_alone(self) -> None:
        mysql_connection = SimpleNamespace(dialect=SimpleNamespace(name="mysql"))
        sqlite_connection = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
        bounded = SimpleNamespace(execution_options={"max_statement_time": 20})
        unbounded = SimpleNamespace(execution_options={})

        for connection, context in (
            (mysql_connection, unbounded),
            (sqlite_connection, bounded),
            (mysql_connection, None),
        ):
            with self.subTest(dialect=connection.dialect.name, context=context):
                statement, _ = bounded_statement(
                    connection, None, "SELECT 1", (), context, False
                )
                self.assertEqual(statement, "SELECT 1")

    def test_sqlite_execution_passes_through_the_listener_unchanged(self) -> None:
        engine = create_engine("sqlite://")
        event.listen(engine, "before_cursor_execute", bounded_statement, retval=True)
        executed: list[str] = []

        @event.listens_for(engine, "after_cursor_execute")
        def _record(conn: object, cursor: object, statement: str, *args: object) -> None:
            executed.append(statement)

        with engine.connect() as connection:
            value = connection.execute(
                select(1).execution_options(max_statement_time=20)
            ).scalar()

        self.assertEqual(value, 1)
        self.assertTrue(all(not statement.startswith("SET STATEMENT") for statement in executed))

    def test_main_db_engine_gets_a_session_statement_ceiling_on_mysql_only(self) -> None:
        for uri, expected in (
            ("mysql+pymysql://user:pass@db.internal:3306/pdx", "SET SESSION max_statement_time=55"),
            ("mariadb+pymysql://user:pass@db.internal:3306/pdx", "SET SESSION max_statement_time=55"),
            ("sqlite:///:memory:", None),
        ):
            with self.subTest(uri=uri):
                app = Flask("statement-timeout-test")
                app.config["SQLALCHEMY_DATABASE_URI"] = uri
                app.config["MAIN_DB_SCHEMA"] = ""
                configure_auth_bind(app, auth_database_uri="sqlite:///:memory:")
                configure_main_db(app)
                connect_args = cast(
                    dict[str, object],
                    app.config["SQLALCHEMY_ENGINE_OPTIONS"].get("connect_args", {}),
                )
                self.assertEqual(connect_args.get("init_command"), expected)

    def test_is_statement_timeout_error_matches_mariadb_error_1969(self) -> None:
        self.assertTrue(is_statement_timeout_error(_statement_timeout()))
        self.assertFalse(
            is_statement_timeout_error(
                OperationalError("SELECT 1", {}, Exception(2013, "Lost connection"))
            )
        )
        self.assertFalse(is_statement_timeout_error(OperationalError("SELECT 1", {}, None)))

    def test_text_search_timeout_becomes_a_client_facing_error(self) -> None:
        query = _chained_query()
        query.all.side_effect = _statement_timeout()
        deps = _service_deps(query)

        with self.assertRaises(ValidationError) as caught:
            _ = run_sections(
                deps,
                ctx=_CTX,
                parsed_args=_parsed_args(text_query="the*", text_match_mode="phrase"),
            )

        self.assertEqual(
            caught.exception.messages,
            {"text_query": [TEXT_QUERY_TOO_EXPENSIVE_MESSAGE]},
        )
        deps.db.session.rollback.assert_called_once()

    def test_other_operational_errors_and_non_text_timeouts_propagate(self) -> None:
        for parsed_args, error in (
            (_parsed_args(text_query="material"), OperationalError("SELECT 1", {}, Exception(2013, "gone"))),
            (_parsed_args(year=[2020]), _statement_timeout()),
        ):
            with self.subTest(text_query=parsed_args["text_query"]):
                query = _chained_query()
                query.all.side_effect = error
                deps = _service_deps(query)
                with self.assertRaises(OperationalError):
                    _ = run_sections(deps, ctx=_CTX, parsed_args=parsed_args)
                deps.db.session.rollback.assert_not_called()


class SectionExactCountCacheTests(unittest.TestCase):
    def test_exact_count_mode_memoizes_section_and_agreement_counts(self) -> None:
        query = _chained_query()
        deps = _service_deps(query)
        deps._cached_exact_query_count.side_effect = [7, 3]
        parsed_args = _parsed_args(
            text_query="material adverse effect",
            count_mode="exact",
            page=2,
            page_size=5,
        )

        response = run_sections(deps, ctx=_CTX, parsed_args=parsed_args)

        cache_key = build_search_count_cache_key("sections", parsed_args)
        self.assertEqual(
            [call.kwargs["cache_key"] for call in deps._cached_exact_query_count.call_args_list],
            [cache_key, f"{cache_key}:agreements"],
        )
        self.assertEqual(response["total_agreement_count"], 3)
        self.assertEqual(response["count_metadata"]["method"], "query_count")
        self.assertEqual(response["count_metadata"]["mode"], "exact")
        query.one.assert_not_called()

    def test_exact_count_metadata_uses_the_shared_cache_key(self) -> None:
        deps = MagicMock()
        deps._cached_exact_query_count.return_value = 41

        result = sections_total_count_metadata(
            deps,
            query=MagicMock(),
            page=4,
            page_size=10,
            item_count=10,
            has_next=True,
            has_filters=True,
            count_mode="exact",
            count_cache_key="filtered-search",
        )

        self.assertEqual(result, (41, False, "query_count"))
        self.assertEqual(
            deps._cached_exact_query_count.call_args.kwargs["cache_key"],
            "filtered-search",
        )


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
