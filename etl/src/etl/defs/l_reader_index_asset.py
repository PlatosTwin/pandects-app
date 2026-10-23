# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
"""Smart reader index (docs/plans/smart-reader.md): outline, defined terms and
cross-references for each latest verified XML, stored in `reader_index`.

Manual asset, outside every job for now. A row is recomputed when the agreement's
latest verified XML version or ENRICHER_VERSION changes. No LLM calls.
"""

import xml.etree.ElementTree as ET
from typing import Any, List, Mapping, Sequence

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection

from etl.defs.f_xml_asset import xml_verify_asset
from etl.defs.f_xml_repair_cycle_asset import post_repair_verify_xml_asset
from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.reader_index import ENRICHER_VERSION, reader_index_row
from etl.utils.run_config import runs_single_batch
from etl.utils.schema_guards import assert_tables_exist

READER_INDEX_TABLE = "reader_index"

_ROW_COLUMNS = (
    "agreement_uuid",
    "xml_version",
    "enricher_version",
    "text_hash",
    "payload_gz",
    "n_terms",
    "n_term_uses",
    "n_xrefs",
    "n_xrefs_resolved",
    "n_xrefs_unresolved",
    "body_section_level_rate",
)


def reader_index_queue_sql(schema: str) -> str:
    """Latest verified XMLs with no row for their version at the current enricher version."""
    return f"""
    SELECT x.agreement_uuid
    FROM {schema}.xml AS x
    LEFT JOIN {schema}.{READER_INDEX_TABLE} AS r
      ON r.agreement_uuid = x.agreement_uuid
     AND r.xml_version = x.version
     AND r.enricher_version = :enricher_version
    WHERE x.latest = 1
      AND x.status = 'verified'
      AND r.agreement_uuid IS NULL
      AND x.agreement_uuid > :last_uuid
    ORDER BY x.agreement_uuid ASC
    LIMIT :lim
    """


def upsert_reader_index(conn: Connection, schema: str, rows: Sequence[Mapping[str, Any]]) -> None:
    cols = ", ".join(_ROW_COLUMNS)
    params = ", ".join(f":{c}" for c in _ROW_COLUMNS)
    updates = ", ".join(f"{c} = VALUES({c})" for c in _ROW_COLUMNS if c not in ("agreement_uuid", "xml_version"))
    _ = conn.execute(
        text(
            f"""
            INSERT INTO {schema}.{READER_INDEX_TABLE} ({cols}) VALUES ({params})
            ON DUPLICATE KEY UPDATE {updates}, created_date = UTC_TIMESTAMP(3)
            """
        ),
        [dict(r) for r in rows],
    )


def _run_reader_index(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    *,
    log_prefix: str,
) -> List[str]:
    batch_size = pipeline_config.xml_agreement_batch_size
    single_batch_run = runs_single_batch(context, pipeline_config)
    engine = db.get_engine()
    schema = db.database

    with engine.begin() as conn:
        assert_tables_exist(conn, schema=schema, table_names=["xml", READER_INDEX_TABLE])

    last_uuid = ""
    processed: List[str] = []
    failed: List[str] = []
    while True:
        with engine.begin() as conn:
            agreement_uuids = [
                str(u)
                for u in conn.execute(
                    text(reader_index_queue_sql(schema)),
                    {"enricher_version": ENRICHER_VERSION, "last_uuid": last_uuid, "lim": batch_size},
                )
                .scalars()
                .all()
            ]
            if not agreement_uuids:
                break
            rows = (
                conn.execute(
                    text(
                        f"""
                        SELECT x.agreement_uuid, x.version AS xml_version, x.xml
                        FROM {schema}.xml AS x
                        WHERE x.agreement_uuid IN :auuids
                          AND x.latest = 1
                          AND x.status = 'verified'
                        ORDER BY x.agreement_uuid
                        """
                    ).bindparams(bindparam("auuids", expanding=True)),
                    {"auuids": tuple(agreement_uuids)},
                )
                .mappings()
                .fetchall()
            )
            staged: List[dict[str, Any]] = []
            for r in rows:
                agreement_uuid = str(r["agreement_uuid"])
                try:
                    staged.append(
                        reader_index_row(str(r["xml"]), agreement_uuid=agreement_uuid, xml_version=int(r["xml_version"]))
                    )
                except ET.ParseError as exc:
                    # verified XML should always parse; leave the row missing and keep going
                    failed.append(agreement_uuid)
                    context.log.warning("%s: %s did not parse: %s", log_prefix, agreement_uuid, exc)
            if staged:
                upsert_reader_index(conn, schema, staged)
                processed.extend(row["agreement_uuid"] for row in staged)
            context.log.info(
                "%s: indexed %s agreements (%s total, %s failed)", log_prefix, len(staged), len(processed), len(failed)
            )
        last_uuid = agreement_uuids[-1]
        if single_batch_run:
            break
    return sorted(processed)


@dg.asset(deps=[xml_verify_asset, post_repair_verify_xml_asset], name="12_reader_index")
def reader_index_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> List[str]:
    """Manual run over every latest verified XML whose reader index is missing or stale."""
    return _run_reader_index(context, db, pipeline_config, log_prefix="reader_index_asset")
