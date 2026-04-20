# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportAny=false
import unittest
from dataclasses import dataclass
from typing import cast

from etl.utils.db_utils import insert_new_tags, upsert_agreements


class _FakeScalarResult:
    def __init__(self, values: list[str]) -> None:
        self._values = values

    def scalars(self) -> "_FakeScalarResult":
        return self

    def all(self) -> list[str]:
        return self._values


class _FakeConnection:
    def __init__(self, existing_uuids: list[str] | None = None) -> None:
        self.existing_uuids = set(existing_uuids or [])
        self.executions: list[tuple[str, object]] = []

    def execute(self, statement: object, params: object | None = None) -> _FakeScalarResult:
        sql = str(statement)
        self.executions.append((sql, params))
        if "SELECT agreement_uuid" in sql:
            query_params = cast(dict[str, object], params or {})
            uuids = cast(list[str], query_params.get("uuids", []))
            return _FakeScalarResult([uuid for uuid in uuids if uuid in self.existing_uuids])
        return _FakeScalarResult([])


@dataclass
class _AgreementRow:
    agreement_uuid: str
    url: str
    filing_date: object | None = None
    target: str | None = None
    acquirer: str | None = None
    announce_date: object | None = None
    prob_filing: float | None = None
    filing_company_name: str | None = None
    filing_company_cik: str | None = None
    form_type: str | None = None
    exhibit_type: str | None = None
    secondary_filing_url: str | None = None
    auto_status_verified: bool = False


@dataclass
class _TagRow:
    page_uuid: str
    tagged_text: str
    low_count: int
    spans: list[dict[str, object]]
    tokens: list[dict[str, object]]


class UpsertAgreementsTests(unittest.TestCase):
    def test_insert_qualifying_row_sets_valid_status(self) -> None:
        conn = _FakeConnection()
        filing = _AgreementRow(
            agreement_uuid="agreement-1",
            url="https://example.com/a.htm",
            auto_status_verified=True,
        )

        upsert_agreements(cast(object, [filing]), "pdx", cast(object, conn))  # type: ignore[arg-type]

        insert_sql, insert_params = conn.executions[1]
        insert_rows = cast(list[dict[str, object]], insert_params)
        self.assertIn("status", insert_sql)
        self.assertEqual(insert_rows[0]["status"], "verified")
        self.assertTrue(cast(bool, insert_rows[0]["auto_status_verified"]))

    def test_insert_non_qualifying_row_leaves_status_null(self) -> None:
        conn = _FakeConnection()
        filing = _AgreementRow(
            agreement_uuid="agreement-1",
            url="https://example.com/a.htm",
            auto_status_verified=False,
        )

        upsert_agreements(cast(object, [filing]), "pdx", cast(object, conn))  # type: ignore[arg-type]

        _, insert_params = conn.executions[1]
        insert_rows = cast(list[dict[str, object]], insert_params)
        self.assertIsNone(insert_rows[0]["status"])
        self.assertFalse(cast(bool, insert_rows[0]["auto_status_verified"]))

    def test_update_sql_only_fills_null_status_for_qualifying_rows(self) -> None:
        conn = _FakeConnection(existing_uuids=["agreement-1"])
        filing = _AgreementRow(
            agreement_uuid="agreement-1",
            url="https://example.com/a.htm",
            auto_status_verified=True,
        )

        upsert_agreements(cast(object, [filing]), "pdx", cast(object, conn))  # type: ignore[arg-type]

        update_sql, update_params = conn.executions[1]
        update_rows = cast(list[dict[str, object]], update_params)
        self.assertIn("WHEN status IS NULL AND :auto_status_verified THEN 'verified'", update_sql)
        self.assertIn("OR (status IS NULL AND :auto_status_verified)", update_sql)
        self.assertTrue(cast(bool, update_rows[0]["auto_status_verified"]))

    def test_update_non_qualifying_row_does_not_trigger_status_fill(self) -> None:
        conn = _FakeConnection(existing_uuids=["agreement-1"])
        filing = _AgreementRow(
            agreement_uuid="agreement-1",
            url="https://example.com/a.htm",
            auto_status_verified=False,
        )

        upsert_agreements(cast(object, [filing]), "pdx", cast(object, conn))  # type: ignore[arg-type]

        update_sql, update_params = conn.executions[1]
        update_rows = cast(list[dict[str, object]], update_params)
        self.assertFalse(cast(bool, update_rows[0]["auto_status_verified"]))
        self.assertIn("WHEN status IS NULL AND :auto_status_verified THEN 'verified'", update_sql)


class InsertNewTagsTests(unittest.TestCase):
    def test_insert_new_tags_skips_preflight_select(self) -> None:
        conn = _FakeConnection()
        tag = _TagRow(
            page_uuid="page-1",
            tagged_text="<section>Intro</section>",
            low_count=0,
            spans=[],
            tokens=[],
        )

        insert_new_tags(cast(object, [tag]), "pdx", cast(object, conn))  # type: ignore[arg-type]

        self.assertEqual(len(conn.executions), 1)
        insert_sql, insert_params = conn.executions[0]
        insert_rows = cast(list[dict[str, object]], insert_params)
        self.assertIn("INSERT INTO pdx.tagged_outputs", insert_sql)
        self.assertIn("ON DUPLICATE KEY UPDATE", insert_sql)
        self.assertNotIn("SELECT page_uuid", insert_sql)
        self.assertEqual(insert_rows[0]["page_uuid"], "page-1")
        self.assertEqual(insert_rows[0]["spans"], "[]")
        self.assertEqual(insert_rows[0]["tokens"], "[]")


if __name__ == "__main__":
    _ = unittest.main()
