from __future__ import annotations

import json
import os
from datetime import date, datetime
from functools import lru_cache
from typing import Any, ClassVar, cast

from sqlalchemy import (
    CHAR,
    TEXT,
    Column,
    Integer,
    MetaData,
    Table,
    and_,
    create_engine,
    func,
    or_,
    cast as sql_cast,
)
from sqlalchemy.dialects import mysql as mysql_dialect
from sqlalchemy.orm import Mapped
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.types import NullType

from backend.extensions import db


_MAIN_SCHEMA_TOKEN = "__main_schema__"
_REQUIRED_MAIN_DB_ENV_VARS = (
    "MARIADB_USER",
    "MARIADB_PASSWORD",
    "MARIADB_HOST",
    "MARIADB_DATABASE",
)


def main_db_schema_from_env() -> str:
    raw = os.environ.get("MAIN_DB_SCHEMA")
    if raw is None:
        return "pdx"
    value = raw.strip()
    return value or "pdx"


def main_db_uri_from_env() -> str:
    raw = os.environ.get("MAIN_DATABASE_URI", "").strip()
    if raw:
        return raw
    missing = [
        env_key
        for env_key in _REQUIRED_MAIN_DB_ENV_VARS
        if not os.environ.get(env_key, "").strip()
    ]
    if missing:
        required = ", ".join(_REQUIRED_MAIN_DB_ENV_VARS)
        missing_list = ", ".join(missing)
        raise RuntimeError(
            f"Main DB reflection requires MAIN_DATABASE_URI or all MariaDB env vars ({required}). Missing: {missing_list}."
        )
    db_user = os.environ["MARIADB_USER"].strip()
    db_pass = os.environ["MARIADB_PASSWORD"].strip()
    db_host = os.environ["MARIADB_HOST"].strip()
    db_name = os.environ["MARIADB_DATABASE"].strip()
    return f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:3306/{db_name}"


def schema_translate_map(schema: str | None) -> dict[str, str | None]:
    value = schema.strip() if isinstance(schema, str) else ""
    return {_MAIN_SCHEMA_TOKEN: value or None}


_SKIP_MAIN_DB_REFLECTION = os.environ.get("SKIP_MAIN_DB_REFLECTION", "").strip() == "1"
_ENABLE_MAIN_DB_REFLECTION = (
    os.environ.get("ENABLE_MAIN_DB_REFLECTION", "1").strip() != "0"
)
SKIP_MAIN_DB_REFLECTION = _SKIP_MAIN_DB_REFLECTION
ENABLE_MAIN_DB_REFLECTION = _ENABLE_MAIN_DB_REFLECTION
MAIN_SCHEMA_TOKEN = _MAIN_SCHEMA_TOKEN
metadata = MetaData()


class _MySQLVector(NullType):
    """Opaque placeholder for MySQL/MariaDB VECTOR columns during reflection."""

    cache_ok = True

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()
        self._vector_args = args
        self._vector_kwargs = kwargs


# SQLAlchemy's MySQL reflection parser doesn't recognize VECTOR yet.
# Register a tolerant placeholder so reflected tables still load.
_mysql_base_ischema_names = cast(dict[str, object], mysql_dialect.base.ischema_names)
_mysql_dialect_ischema_names = cast(dict[str, object], mysql_dialect.dialect.ischema_names)
_ = _mysql_base_ischema_names.setdefault("vector", _MySQLVector)
_ = _mysql_dialect_ischema_names.setdefault("vector", _MySQLVector)

if _ENABLE_MAIN_DB_REFLECTION and not _SKIP_MAIN_DB_REFLECTION:
    engine = create_engine(
        main_db_uri_from_env(),
        execution_options={
            "schema_translate_map": schema_translate_map(main_db_schema_from_env())
        },
    )

    agreements_table = Table(
        "agreements",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    xml_table = Table(
        "xml",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    taxonomy_l1_table = Table(
        "taxonomy_l1",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    taxonomy_l2_table = Table(
        "taxonomy_l2",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    taxonomy_l3_table = Table(
        "taxonomy_l3",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    naics_sectors_table = Table(
        "naics_sectors",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    naics_sub_sectors_table = Table(
        "naics_sub_sectors",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    sections_table = Table(
        "sections",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    latest_sections_search_table = Table(
        "latest_sections_search",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    latest_sections_search_standard_ids_table = Table(
        "latest_sections_search_standard_ids",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    clauses_table = Table(
        "clauses",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    tax_clause_taxonomy_l1_table = Table(
        "tax_clause_taxonomy_l1",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    tax_clause_taxonomy_l2_table = Table(
        "tax_clause_taxonomy_l2",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    tax_clause_taxonomy_l3_table = Table(
        "tax_clause_taxonomy_l3",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    tax_clause_assignments_table = Table(
        "tax_clause_assignments",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    counsel_table = Table(
        "counsel",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    agreement_counsel_table = Table(
        "agreement_counsel",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    dump_versions_table = Table(
        "dump_versions",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
else:
    # Test mode: avoid connecting to the main DB at import time.
    engine = None
    agreements_table = Table(
        "agreements",
        metadata,
        Column("agreement_uuid", CHAR(36), primary_key=True),
        Column("filing_date", TEXT, nullable=True),
        Column("prob_filing", TEXT, nullable=True),
        Column("filing_company_name", TEXT, nullable=True),
        Column("filing_company_cik", TEXT, nullable=True),
        Column("form_type", TEXT, nullable=True),
        Column("exhibit_type", TEXT, nullable=True),
        Column("target", TEXT, nullable=True),
        Column("acquirer", TEXT, nullable=True),
        Column("target_counsel", TEXT, nullable=True),
        Column("acquirer_counsel", TEXT, nullable=True),
        Column("transaction_price_total", TEXT, nullable=True),
        Column("transaction_price_stock", TEXT, nullable=True),
        Column("transaction_price_cash", TEXT, nullable=True),
        Column("transaction_price_assets", TEXT, nullable=True),
        Column("transaction_consideration", TEXT, nullable=True),
        Column("target_type", TEXT, nullable=True),
        Column("acquirer_type", TEXT, nullable=True),
        Column("target_industry", TEXT, nullable=True),
        Column("acquirer_industry", TEXT, nullable=True),
        Column("announce_date", TEXT, nullable=True),
        Column("close_date", TEXT, nullable=True),
        Column("deal_status", TEXT, nullable=True),
        Column("attitude", TEXT, nullable=True),
        Column("deal_type", TEXT, nullable=True),
        Column("purpose", TEXT, nullable=True),
        Column("target_pe", Integer, nullable=True),
        Column("acquirer_pe", Integer, nullable=True),
        Column("verified", Integer, nullable=True),
        Column("gated", Integer, nullable=True),
        Column("metadata", Integer, nullable=True),
        Column("transaction_size", Integer, nullable=True),
        Column("transaction_type", TEXT, nullable=True),
        Column("consideration_type", TEXT, nullable=True),
        Column("url", TEXT, nullable=True),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    xml_table = Table(
        "xml",
        metadata,
        Column("agreement_uuid", CHAR(36), primary_key=True),
        Column("xml", TEXT, nullable=True),
        Column("version", Integer, primary_key=True, nullable=True),
        Column("status", TEXT, nullable=True),
        Column("latest", Integer, nullable=False, default=0),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    taxonomy_l1_table = Table(
        "taxonomy_l1",
        metadata,
        Column("standard_id", TEXT, primary_key=True),
        Column("label", TEXT, nullable=True),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    taxonomy_l2_table = Table(
        "taxonomy_l2",
        metadata,
        Column("standard_id", TEXT, primary_key=True),
        Column("label", TEXT, nullable=True),
        Column("parent_id", TEXT, nullable=True),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    taxonomy_l3_table = Table(
        "taxonomy_l3",
        metadata,
        Column("standard_id", TEXT, primary_key=True),
        Column("label", TEXT, nullable=True),
        Column("parent_id", TEXT, nullable=True),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    naics_sectors_table = Table(
        "naics_sectors",
        metadata,
        Column("super_sector", TEXT, nullable=True),
        Column("sector_group", TEXT, nullable=True),
        Column("sector_desc", TEXT, nullable=True),
        Column("sector_code", Integer, primary_key=True),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    naics_sub_sectors_table = Table(
        "naics_sub_sectors",
        metadata,
        Column("sub_sector_desc", TEXT, nullable=True),
        Column("sub_sector_code", Integer, primary_key=True),
        Column("sector_code", Integer, nullable=False),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    sections_table = Table(
        "sections",
        metadata,
        Column("agreement_uuid", CHAR(36), nullable=False),
        Column("section_uuid", CHAR(36), primary_key=True),
        Column("article_title", TEXT, nullable=True),
        Column("article_title_normed", TEXT, nullable=True),
        Column("article_order", Integer, nullable=True),
        Column("section_title", TEXT, nullable=True),
        Column("section_title_normed", TEXT, nullable=True),
        Column("section_order", Integer, nullable=True),
        Column("xml_content", TEXT, nullable=False),
        Column("section_standard_id", TEXT, nullable=True),
        Column("section_standard_id_gold_label", TEXT, nullable=True),
        Column("xml_version", Integer, nullable=True),
        Column("gold_label_model", TEXT, nullable=True),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    latest_sections_search_table = Table(
        "latest_sections_search",
        metadata,
        Column("section_uuid", CHAR(36), primary_key=True),
        Column("agreement_uuid", CHAR(36), nullable=False),
        Column("filing_date", TEXT, nullable=True),
        Column("prob_filing", TEXT, nullable=True),
        Column("filing_company_name", TEXT, nullable=True),
        Column("filing_company_cik", TEXT, nullable=True),
        Column("form_type", TEXT, nullable=True),
        Column("exhibit_type", TEXT, nullable=True),
        Column("target", TEXT, nullable=True),
        Column("acquirer", TEXT, nullable=True),
        Column("transaction_price_total", TEXT, nullable=True),
        Column("transaction_price_stock", TEXT, nullable=True),
        Column("transaction_price_cash", TEXT, nullable=True),
        Column("transaction_price_assets", TEXT, nullable=True),
        Column("transaction_consideration", TEXT, nullable=True),
        Column("target_type", TEXT, nullable=True),
        Column("acquirer_type", TEXT, nullable=True),
        Column("target_counsel", TEXT, nullable=True),
        Column("acquirer_counsel", TEXT, nullable=True),
        Column("target_industry", TEXT, nullable=True),
        Column("acquirer_industry", TEXT, nullable=True),
        Column("announce_date", TEXT, nullable=True),
        Column("close_date", TEXT, nullable=True),
        Column("deal_status", TEXT, nullable=True),
        Column("attitude", TEXT, nullable=True),
        Column("deal_type", TEXT, nullable=True),
        Column("purpose", TEXT, nullable=True),
        Column("target_pe", Integer, nullable=True),
        Column("acquirer_pe", Integer, nullable=True),
        Column("verified", Integer, nullable=True),
        Column("url", TEXT, nullable=True),
        Column("section_standard_ids", TEXT, nullable=True),
        Column("article_title", TEXT, nullable=True),
        Column("section_title", TEXT, nullable=True),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    latest_sections_search_standard_ids_table = Table(
        "latest_sections_search_standard_ids",
        metadata,
        Column("standard_id", TEXT, primary_key=True),
        Column("section_uuid", CHAR(36), primary_key=True),
        Column("agreement_uuid", CHAR(36), nullable=False),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    clauses_table = Table(
        "clauses",
        metadata,
        Column("clause_uuid", CHAR(36), primary_key=True),
        Column("agreement_uuid", CHAR(36), nullable=False),
        Column("section_uuid", CHAR(36), nullable=False),
        Column("xml_version", Integer, nullable=True),
        Column("module", TEXT, nullable=False),
        Column("clause_order", Integer, nullable=False),
        Column("anchor_label", TEXT, nullable=True),
        Column("start_char", Integer, nullable=False),
        Column("end_char", Integer, nullable=False),
        Column("clause_text", TEXT, nullable=False),
        Column("source_method", TEXT, nullable=False),
        Column("context_type", TEXT, nullable=False),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    tax_clause_taxonomy_l1_table = Table(
        "tax_clause_taxonomy_l1",
        metadata,
        Column("standard_id", TEXT, primary_key=True),
        Column("label", TEXT, nullable=True),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    tax_clause_taxonomy_l2_table = Table(
        "tax_clause_taxonomy_l2",
        metadata,
        Column("standard_id", TEXT, primary_key=True),
        Column("label", TEXT, nullable=True),
        Column("parent_id", TEXT, nullable=True),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    tax_clause_taxonomy_l3_table = Table(
        "tax_clause_taxonomy_l3",
        metadata,
        Column("standard_id", TEXT, primary_key=True),
        Column("label", TEXT, nullable=True),
        Column("parent_id", TEXT, nullable=True),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    tax_clause_assignments_table = Table(
        "tax_clause_assignments",
        metadata,
        Column("clause_uuid", CHAR(36), primary_key=True),
        Column("standard_id", TEXT, primary_key=True),
        Column("is_gold_label", Integer, nullable=False, default=0),
        Column("model_name", TEXT, nullable=True),
        Column("assigned_at", TEXT, nullable=True),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    counsel_table = Table(
        "counsel",
        metadata,
        Column("counsel_id", Integer, primary_key=True, autoincrement=True),
        Column("canonical_name", TEXT, nullable=False),
        Column("canonical_name_normalized", TEXT, nullable=False),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    agreement_counsel_table = Table(
        "agreement_counsel",
        metadata,
        Column("agreement_uuid", CHAR(36), primary_key=True),
        Column("side", TEXT, primary_key=True),
        Column("position", Integer, primary_key=True),
        Column("raw_name", TEXT, nullable=False),
        Column("counsel_id", Integer, nullable=False),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    dump_versions_table = Table(
        "dump_versions",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("sha256", CHAR(64), nullable=False),
        Column("dump_ts", TEXT, nullable=False),
        Column("size_bytes", Integer, nullable=True),
        Column("download_url", TEXT, nullable=True),
        Column("recorded_at", TEXT, nullable=False),
        schema=_MAIN_SCHEMA_TOKEN,
    )


class Sections(db.Model):
    __table__ = sections_table
    agreement_uuid: ClassVar[Mapped[str]]
    section_uuid: ClassVar[Mapped[str]]
    article_title: ClassVar[Mapped[str | None]]
    article_title_normed: ClassVar[Mapped[str | None]]
    article_order: ClassVar[Mapped[int | None]]
    section_title: ClassVar[Mapped[str | None]]
    section_title_normed: ClassVar[Mapped[str | None]]
    section_order: ClassVar[Mapped[int | None]]
    xml_content: ClassVar[Mapped[str]]
    section_standard_id: ClassVar[Mapped[str | None]]
    section_standard_id_gold_label: ClassVar[Mapped[str | None]]
    xml_version: ClassVar[Mapped[int | None]]
    gold_label_model: ClassVar[Mapped[str | None]]


class Agreements(db.Model):
    __table__ = agreements_table
    agreement_uuid: ClassVar[Mapped[str]]
    filing_date: ClassVar[Mapped[str | None]]
    prob_filing: ClassVar[Mapped[str | None]]
    filing_company_name: ClassVar[Mapped[str | None]]
    filing_company_cik: ClassVar[Mapped[str | None]]
    form_type: ClassVar[Mapped[str | None]]
    exhibit_type: ClassVar[Mapped[str | None]]
    target: ClassVar[Mapped[str | None]]
    acquirer: ClassVar[Mapped[str | None]]
    target_counsel: ClassVar[Mapped[str | None]]
    acquirer_counsel: ClassVar[Mapped[str | None]]
    transaction_price_total: ClassVar[Mapped[str | None]]
    transaction_price_stock: ClassVar[Mapped[str | None]]
    transaction_price_cash: ClassVar[Mapped[str | None]]
    transaction_price_assets: ClassVar[Mapped[str | None]]
    transaction_consideration: ClassVar[Mapped[str | None]]
    target_type: ClassVar[Mapped[str | None]]
    acquirer_type: ClassVar[Mapped[str | None]]
    target_counsel: ClassVar[Mapped[str | None]]
    acquirer_counsel: ClassVar[Mapped[str | None]]
    target_industry: ClassVar[Mapped[str | None]]
    acquirer_industry: ClassVar[Mapped[str | None]]
    announce_date: ClassVar[Mapped[str | None]]
    close_date: ClassVar[Mapped[str | None]]
    deal_status: ClassVar[Mapped[str | None]]
    attitude: ClassVar[Mapped[str | None]]
    deal_type: ClassVar[Mapped[str | None]]
    purpose: ClassVar[Mapped[str | None]]
    target_pe: ClassVar[Mapped[int | None]]
    acquirer_pe: ClassVar[Mapped[int | None]]
    verified: ClassVar[Mapped[int | None]]
    gated: ClassVar[Mapped[int | None]]
    transaction_size: ClassVar[Mapped[int | None]]
    transaction_type: ClassVar[Mapped[str | None]]
    consideration_type: ClassVar[Mapped[str | None]]
    url: ClassVar[Mapped[str | None]]


class LatestSectionsSearch(db.Model):
    __table__ = latest_sections_search_table
    section_uuid: ClassVar[Mapped[str]]
    agreement_uuid: ClassVar[Mapped[str]]
    filing_date: ClassVar[Mapped[str | None]]
    prob_filing: ClassVar[Mapped[str | None]]
    filing_company_name: ClassVar[Mapped[str | None]]
    filing_company_cik: ClassVar[Mapped[str | None]]
    form_type: ClassVar[Mapped[str | None]]
    exhibit_type: ClassVar[Mapped[str | None]]
    target: ClassVar[Mapped[str | None]]
    acquirer: ClassVar[Mapped[str | None]]
    transaction_price_total: ClassVar[Mapped[str | None]]
    transaction_price_stock: ClassVar[Mapped[str | None]]
    transaction_price_cash: ClassVar[Mapped[str | None]]
    transaction_price_assets: ClassVar[Mapped[str | None]]
    transaction_consideration: ClassVar[Mapped[str | None]]
    target_type: ClassVar[Mapped[str | None]]
    acquirer_type: ClassVar[Mapped[str | None]]
    target_industry: ClassVar[Mapped[str | None]]
    acquirer_industry: ClassVar[Mapped[str | None]]
    announce_date: ClassVar[Mapped[str | None]]
    close_date: ClassVar[Mapped[str | None]]
    deal_status: ClassVar[Mapped[str | None]]
    attitude: ClassVar[Mapped[str | None]]
    deal_type: ClassVar[Mapped[str | None]]
    purpose: ClassVar[Mapped[str | None]]
    target_pe: ClassVar[Mapped[int | None]]
    acquirer_pe: ClassVar[Mapped[int | None]]
    verified: ClassVar[Mapped[int | None]]
    url: ClassVar[Mapped[str | None]]
    section_standard_ids: ClassVar[Mapped[str | None]]
    article_title: ClassVar[Mapped[str | None]]
    section_title: ClassVar[Mapped[str | None]]


class LatestSectionsSearchStandardId(db.Model):
    __table__ = latest_sections_search_standard_ids_table
    standard_id: ClassVar[Mapped[str]]
    section_uuid: ClassVar[Mapped[str]]
    agreement_uuid: ClassVar[Mapped[str]]


class Clauses(db.Model):
    __table__ = clauses_table
    clause_uuid: ClassVar[Mapped[str]]
    agreement_uuid: ClassVar[Mapped[str]]
    section_uuid: ClassVar[Mapped[str]]
    xml_version: ClassVar[Mapped[int | None]]
    module: ClassVar[Mapped[str]]
    clause_order: ClassVar[Mapped[int]]
    anchor_label: ClassVar[Mapped[str | None]]
    start_char: ClassVar[Mapped[int]]
    end_char: ClassVar[Mapped[int]]
    clause_text: ClassVar[Mapped[str]]
    source_method: ClassVar[Mapped[str]]
    context_type: ClassVar[Mapped[str]]


class TaxClauseAssignment(db.Model):
    __table__ = tax_clause_assignments_table
    clause_uuid: ClassVar[Mapped[str]]
    standard_id: ClassVar[Mapped[str]]
    is_gold_label: ClassVar[Mapped[int]]
    model_name: ClassVar[Mapped[str | None]]
    assigned_at: ClassVar[Mapped[str | None]]


class Counsel(db.Model):
    __table__ = counsel_table
    counsel_id: ClassVar[Mapped[int]]
    canonical_name: ClassVar[Mapped[str]]
    canonical_name_normalized: ClassVar[Mapped[str]]


class AgreementCounsel(db.Model):
    __table__ = agreement_counsel_table
    agreement_uuid: ClassVar[Mapped[str]]
    side: ClassVar[Mapped[str]]
    position: ClassVar[Mapped[int]]
    raw_name: ClassVar[Mapped[str]]
    counsel_id: ClassVar[Mapped[int]]


class DumpVersion(db.Model):
    __table__ = dump_versions_table
    id: ClassVar[Mapped[int]]
    sha256: ClassVar[Mapped[str]]
    dump_ts: ClassVar[Mapped[str]]
    size_bytes: ClassVar[Mapped[int | None]]
    download_url: ClassVar[Mapped[str | None]]
    recorded_at: ClassVar[Mapped[str]]


class XML(db.Model):
    __table__ = xml_table
    agreement_uuid: ClassVar[Mapped[str]]
    xml: ClassVar[Mapped[str | None]]
    version: ClassVar[Mapped[int | None]]
    status: ClassVar[Mapped[str | None]]
    latest: ClassVar[Mapped[int]]


class TaxonomyL1(db.Model):
    __table__ = taxonomy_l1_table
    standard_id: ClassVar[Mapped[str]]
    label: ClassVar[Mapped[str | None]]


class TaxonomyL2(db.Model):
    __table__ = taxonomy_l2_table
    standard_id: ClassVar[Mapped[str]]
    label: ClassVar[Mapped[str | None]]
    parent_id: ClassVar[Mapped[str | None]]


class TaxonomyL3(db.Model):
    __table__ = taxonomy_l3_table
    standard_id: ClassVar[Mapped[str]]
    label: ClassVar[Mapped[str | None]]
    parent_id: ClassVar[Mapped[str | None]]


class TaxClauseTaxonomyL1(db.Model):
    __table__ = tax_clause_taxonomy_l1_table
    standard_id: ClassVar[Mapped[str]]
    label: ClassVar[Mapped[str | None]]


class TaxClauseTaxonomyL2(db.Model):
    __table__ = tax_clause_taxonomy_l2_table
    standard_id: ClassVar[Mapped[str]]
    label: ClassVar[Mapped[str | None]]
    parent_id: ClassVar[Mapped[str | None]]


class TaxClauseTaxonomyL3(db.Model):
    __table__ = tax_clause_taxonomy_l3_table
    standard_id: ClassVar[Mapped[str]]
    label: ClassVar[Mapped[str | None]]
    parent_id: ClassVar[Mapped[str | None]]


class NaicsSector(db.Model):
    __table__ = naics_sectors_table
    super_sector: ClassVar[Mapped[str | None]]
    sector_group: ClassVar[Mapped[str | None]]
    sector_desc: ClassVar[Mapped[str | None]]
    sector_code: ClassVar[Mapped[int]]


class NaicsSubSector(db.Model):
    __table__ = naics_sub_sectors_table
    sub_sector_desc: ClassVar[Mapped[str | None]]
    sub_sector_code: ClassVar[Mapped[int]]
    sector_code: ClassVar[Mapped[int]]


def coalesced_section_standard_ids():
    gold_label_col = Sections.__table__.c.get("section_standard_id_gold_label")
    if gold_label_col is None:
        return Sections.section_standard_id
    return func.coalesce(gold_label_col, Sections.section_standard_id)


def agreement_year_expr():
    return sql_cast(func.substr(Agreements.filing_date, 1, 4), Integer)


def xml_latest_ok_filter():
    return and_(XML.latest == 1, or_(XML.status.is_(None), XML.status == "verified"))


def agreement_latest_xml_join_condition():
    return and_(Agreements.agreement_uuid == XML.agreement_uuid, xml_latest_ok_filter())


def section_latest_xml_join_condition():
    return and_(
        Sections.agreement_uuid == XML.agreement_uuid,
        Sections.xml_version == XML.version,
        xml_latest_ok_filter(),
    )


def parse_section_standard_ids(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        raw_items = cast(list[object], raw)
        if not all(isinstance(item, str) for item in raw_items):
            raise ValueError("section_standard_id must contain string values.")
        return cast(list[str], raw_items)
    if isinstance(raw, str):
        parsed_obj = cast(object, json.loads(raw))
        if not isinstance(parsed_obj, list) or not all(
            isinstance(item, str) for item in cast(list[object], parsed_obj)
        ):
            raise ValueError("section_standard_id must be a JSON list of strings.")
        return cast(list[str], parsed_obj)
    raise TypeError(f"Unsupported section_standard_id type: {type(raw)!r}")


def year_from_filing_date_value(raw: object) -> int | None:
    if isinstance(raw, datetime):
        return raw.year
    if isinstance(raw, date):
        return raw.year
    if isinstance(raw, str):
        raw_value = raw.strip()
        if len(raw_value) >= 4 and raw_value[:4].isdigit():
            return int(raw_value[:4])
    return None


def expand_taxonomy_standard_ids(standard_ids: list[str]) -> list[str]:
    if not standard_ids:
        return []

    standard_ids_set = {value for value in standard_ids if value}
    if not standard_ids_set:
        return []

    l1_rows = cast(
        list[tuple[object]],
        db.session.query(TaxonomyL1.standard_id)
        .filter(TaxonomyL1.standard_id.in_(standard_ids_set))
        .all(),
    )
    l1_ids = {standard_id for (standard_id,) in l1_rows if isinstance(standard_id, str)}
    l2_rows = cast(
        list[tuple[object]],
        db.session.query(TaxonomyL2.standard_id)
        .filter(TaxonomyL2.standard_id.in_(standard_ids_set))
        .all(),
    )
    l2_ids = {standard_id for (standard_id,) in l2_rows if isinstance(standard_id, str)}
    l3_rows = cast(
        list[tuple[object]],
        db.session.query(TaxonomyL3.standard_id)
        .filter(TaxonomyL3.standard_id.in_(standard_ids_set))
        .all(),
    )
    l3_ids = {standard_id for (standard_id,) in l3_rows if isinstance(standard_id, str)}

    expanded_l2_ids: set[str] = set()
    expanded_l3_ids: set[str] = set()
    if l1_ids:
        expanded_l2_rows = cast(
            list[tuple[object]],
            db.session.query(TaxonomyL2.standard_id)
            .filter(TaxonomyL2.parent_id.in_(l1_ids))
            .all(),
        )
        expanded_l2_ids.update(
            standard_id for (standard_id,) in expanded_l2_rows if isinstance(standard_id, str)
        )
        expanded_l3_rows_from_l1 = cast(
            list[tuple[object]],
            db.session.query(TaxonomyL3.standard_id)
            .join(TaxonomyL2, TaxonomyL3.parent_id == TaxonomyL2.standard_id)
            .filter(TaxonomyL2.parent_id.in_(l1_ids))
            .all(),
        )
        expanded_l3_ids.update(
            standard_id
            for (standard_id,) in expanded_l3_rows_from_l1
            if isinstance(standard_id, str)
        )
    if l2_ids:
        expanded_l3_rows_from_l2 = cast(
            list[tuple[object]],
            db.session.query(TaxonomyL3.standard_id)
            .filter(TaxonomyL3.parent_id.in_(l2_ids))
            .all(),
        )
        expanded_l3_ids.update(
            standard_id
            for (standard_id,) in expanded_l3_rows_from_l2
            if isinstance(standard_id, str)
        )

    return list(
        standard_ids_set | l1_ids | l2_ids | l3_ids | expanded_l2_ids | expanded_l3_ids
    )


@lru_cache(maxsize=512)
def expand_taxonomy_standard_ids_cached(standard_ids_key: tuple[str, ...]) -> tuple[str, ...]:
    if not standard_ids_key:
        return ()
    return tuple(expand_taxonomy_standard_ids(list(standard_ids_key)))


def standard_id_filter_expr(expanded_standard_ids: list[str]) -> ColumnElement[bool]:
    """Match sections via the normalized latest-search standard-id mapping."""
    return cast(
        ColumnElement[bool],
        db.session.query(LatestSectionsSearchStandardId.section_uuid)
        .filter(
            LatestSectionsSearchStandardId.section_uuid == LatestSectionsSearch.section_uuid,
            LatestSectionsSearchStandardId.standard_id.in_(expanded_standard_ids),
        )
        .exists(),
    )


def expand_tax_clause_taxonomy_standard_ids(standard_ids: list[str]) -> list[str]:
    if not standard_ids:
        return []

    standard_ids_set = {value for value in standard_ids if value}
    if not standard_ids_set:
        return []

    l1_rows = cast(
        list[tuple[object]],
        db.session.query(TaxClauseTaxonomyL1.standard_id)
        .filter(TaxClauseTaxonomyL1.standard_id.in_(standard_ids_set))
        .all(),
    )
    l1_ids = {standard_id for (standard_id,) in l1_rows if isinstance(standard_id, str)}
    l2_rows = cast(
        list[tuple[object]],
        db.session.query(TaxClauseTaxonomyL2.standard_id)
        .filter(TaxClauseTaxonomyL2.standard_id.in_(standard_ids_set))
        .all(),
    )
    l2_ids = {standard_id for (standard_id,) in l2_rows if isinstance(standard_id, str)}
    l3_rows = cast(
        list[tuple[object]],
        db.session.query(TaxClauseTaxonomyL3.standard_id)
        .filter(TaxClauseTaxonomyL3.standard_id.in_(standard_ids_set))
        .all(),
    )
    l3_ids = {standard_id for (standard_id,) in l3_rows if isinstance(standard_id, str)}

    expanded_l2_ids: set[str] = set()
    expanded_l3_ids: set[str] = set()
    if l1_ids:
        expanded_l2_rows = cast(
            list[tuple[object]],
            db.session.query(TaxClauseTaxonomyL2.standard_id)
            .filter(TaxClauseTaxonomyL2.parent_id.in_(l1_ids))
            .all(),
        )
        expanded_l2_ids.update(
            standard_id for (standard_id,) in expanded_l2_rows if isinstance(standard_id, str)
        )
        expanded_l3_from_l1 = cast(
            list[tuple[object]],
            db.session.query(TaxClauseTaxonomyL3.standard_id)
            .join(TaxClauseTaxonomyL2, TaxClauseTaxonomyL3.parent_id == TaxClauseTaxonomyL2.standard_id)
            .filter(TaxClauseTaxonomyL2.parent_id.in_(l1_ids))
            .all(),
        )
        expanded_l3_ids.update(
            standard_id for (standard_id,) in expanded_l3_from_l1 if isinstance(standard_id, str)
        )
    if l2_ids:
        expanded_l3_from_l2 = cast(
            list[tuple[object]],
            db.session.query(TaxClauseTaxonomyL3.standard_id)
            .filter(TaxClauseTaxonomyL3.parent_id.in_(l2_ids))
            .all(),
        )
        expanded_l3_ids.update(
            standard_id for (standard_id,) in expanded_l3_from_l2 if isinstance(standard_id, str)
        )

    return list(
        standard_ids_set | l1_ids | l2_ids | l3_ids | expanded_l2_ids | expanded_l3_ids
    )


@lru_cache(maxsize=512)
def expand_tax_clause_taxonomy_standard_ids_cached(
    standard_ids_key: tuple[str, ...],
) -> tuple[str, ...]:
    if not standard_ids_key:
        return ()
    return tuple(expand_tax_clause_taxonomy_standard_ids(list(standard_ids_key)))


def tax_clause_standard_id_filter_expr(
    expanded_standard_ids: list[str],
) -> ColumnElement[bool]:
    """Match tax clauses that have any of the given taxonomy standard_ids assigned."""
    return cast(
        ColumnElement[bool],
        db.session.query(TaxClauseAssignment.clause_uuid)
        .filter(
            TaxClauseAssignment.clause_uuid == Clauses.clause_uuid,
            TaxClauseAssignment.standard_id.in_(expanded_standard_ids),
        )
        .exists(),
    )


def standard_id_agreement_filter_expr(
    agreement_uuid_column: Any,
    expanded_standard_ids: list[str],
) -> ColumnElement[bool]:
    """Match agreements that contain at least one section with any of the given taxonomy ids."""
    return cast(
        ColumnElement[bool],
        db.session.query(LatestSectionsSearchStandardId.agreement_uuid)
        .filter(
            LatestSectionsSearchStandardId.agreement_uuid == agreement_uuid_column,
            LatestSectionsSearchStandardId.standard_id.in_(expanded_standard_ids),
        )
        .exists(),
    )


__all__ = [
    "MAIN_SCHEMA_TOKEN",
    "ENABLE_MAIN_DB_REFLECTION",
    "SKIP_MAIN_DB_REFLECTION",
    "_MAIN_SCHEMA_TOKEN",
    "_ENABLE_MAIN_DB_REFLECTION",
    "_SKIP_MAIN_DB_REFLECTION",
    "Agreements",
    "Clauses",
    "DumpVersion",
    "LatestSectionsSearch",
    "LatestSectionsSearchStandardId",
    "NaicsSector",
    "NaicsSubSector",
    "Sections",
    "TaxClauseAssignment",
    "TaxClauseTaxonomyL1",
    "TaxClauseTaxonomyL2",
    "TaxClauseTaxonomyL3",
    "TaxonomyL1",
    "TaxonomyL2",
    "TaxonomyL3",
    "XML",
    "agreement_latest_xml_join_condition",
    "agreement_year_expr",
    "coalesced_section_standard_ids",
    "expand_tax_clause_taxonomy_standard_ids",
    "expand_tax_clause_taxonomy_standard_ids_cached",
    "expand_taxonomy_standard_ids",
    "expand_taxonomy_standard_ids_cached",
    "tax_clause_standard_id_filter_expr",
    "main_db_schema_from_env",
    "main_db_uri_from_env",
    "parse_section_standard_ids",
    "schema_translate_map",
    "section_latest_xml_join_condition",
    "standard_id_filter_expr",
    "standard_id_agreement_filter_expr",
    "year_from_filing_date_value",
    "xml_latest_ok_filter",
]
