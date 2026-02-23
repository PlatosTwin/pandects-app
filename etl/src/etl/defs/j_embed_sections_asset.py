# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
import importlib
import json
import os
from typing import Any, List

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection

from etl.defs.g_sections_asset import sections_asset
from etl.defs.resources import DBResource, EmbedTarget, PipelineConfig
from etl.utils.post_asset_refresh import run_post_asset_refresh

_VOYAGE_MODEL = "voyage-4-large"
_VOYAGE_EMBED_BATCH_SIZE = 128


def _voyage_client() -> Any:
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("VOYAGE_API_KEY is required for 9_embed_sections.")
    voyageai = importlib.import_module("voyageai")
    return voyageai.Client(api_key=api_key)


def _assert_sections_embedding_columns(conn: Connection, schema: str) -> None:
    required_columns = [
        "section_uuid",
        "agreement_uuid",
        "xml_content",
        "section_standard_id",
        "section_standard_id_gold_label",
        "embedding",
    ]
    query = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = 'sections'
          AND column_name IN :column_names
        """
    ).bindparams(bindparam("column_names", expanding=True))

    existing_columns = set(
        conn.execute(
            query,
            {"schema": schema, "column_names": required_columns},
        ).scalars().all()
    )
    missing = [col for col in required_columns if col not in existing_columns]
    if missing:
        missing_csv = ", ".join(missing)
        raise RuntimeError(
            f"9_embed_sections requires columns on {schema}.sections: {missing_csv}."
        )


def _select_agreement_pool(
    conn: Connection,
    sections_table: str,
    agreement_batch_size: int,
) -> List[str]:
    return [
        str(v)
        for v in conn.execute(
            text(
                f"""
                SELECT DISTINCT agreement_uuid
                FROM {sections_table}
                WHERE embedding IS NULL
                ORDER BY agreement_uuid
                LIMIT :lim
                """
            ),
            {"lim": agreement_batch_size},
        ).scalars().all()
    ]


def _select_sections_for_agreements(
    conn: Connection,
    sections_table: str,
    agreement_uuids: List[str],
) -> List[dict[str, Any]]:
    if not agreement_uuids:
        return []
    return [
        dict(r)
        for r in conn.execute(
            text(
                f"""
                SELECT section_uuid, agreement_uuid, xml_content
                FROM {sections_table}
                WHERE agreement_uuid IN :agreement_uuids
                ORDER BY agreement_uuid, section_uuid
                """
            ).bindparams(bindparam("agreement_uuids", expanding=True)),
            {"agreement_uuids": tuple(agreement_uuids)},
        ).mappings().fetchall()
    ]


def _select_focus_sections(
    conn: Connection,
    sections_table: str,
    focus_section: str,
    focus_batch_size: int,
) -> List[dict[str, Any]]:
    return [
        dict(r)
        for r in conn.execute(
            text(
                f"""
                SELECT section_uuid, agreement_uuid, xml_content
                FROM {sections_table}
                WHERE COALESCE(section_standard_id_gold_label, section_standard_id) = :focus_section
                  AND embedding IS NULL
                ORDER BY agreement_uuid, section_uuid
                LIMIT :lim
                """
            ),
            {"focus_section": focus_section, "lim": focus_batch_size},
        ).mappings().fetchall()
    ]


def _embed_documents(client: Any, documents: List[str]) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for i in range(0, len(documents), _VOYAGE_EMBED_BATCH_SIZE):
        batch = documents[i : i + _VOYAGE_EMBED_BATCH_SIZE]
        response = client.embed(
            batch,
            model=_VOYAGE_MODEL,
            input_type="document",
        )
        embeddings.extend(response.embeddings)
    return embeddings


@dg.asset(deps=[sections_asset], name="9_embed_sections")
def embed_sections_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> int:
    agreement_batch_size = pipeline_config.embed_agreement_batch_size
    focus_batch_size = pipeline_config.embed_focus_section_batch_size
    focus_section = pipeline_config.embed_focus_section.strip()

    if agreement_batch_size <= 0:
        raise ValueError("embed_agreement_batch_size must be > 0.")
    if focus_batch_size <= 0:
        raise ValueError("embed_focus_section_batch_size must be > 0.")
    if pipeline_config.embed_target == EmbedTarget.SECTION and not focus_section:
        raise ValueError("embed_focus_section is required when embed_target='section'.")

    engine = db.get_engine()
    schema = db.database
    sections_table = f"{schema}.sections"

    with engine.begin() as conn:
        _assert_sections_embedding_columns(conn, schema)
        if pipeline_config.embed_target == EmbedTarget.AGREEMENT:
            agreement_uuids = _select_agreement_pool(conn, sections_table, agreement_batch_size)
            section_rows = _select_sections_for_agreements(conn, sections_table, agreement_uuids)
            context.log.info(
                "9_embed_sections: selected %s agreements and %s sections.",
                len(agreement_uuids),
                len(section_rows),
            )
        else:
            section_rows = _select_focus_sections(
                conn, sections_table, focus_section, focus_batch_size
            )
            context.log.info(
                "9_embed_sections: selected %s sections where section_standard_id='%s'.",
                len(section_rows),
                focus_section,
            )

    if not section_rows:
        run_post_asset_refresh(context, db, pipeline_config)
        return 0

    documents = [str(r["xml_content"]) for r in section_rows]
    client = _voyage_client()
    embeddings = _embed_documents(client, documents)
    if len(embeddings) != len(section_rows):
        raise RuntimeError("VoyageAI returned an embedding count that does not match input size.")

    update_payload = [
        {
            "section_uuid": str(row["section_uuid"]),
            "embedding": json.dumps(embedding, separators=(",", ":")),
        }
        for row, embedding in zip(section_rows, embeddings)
    ]

    update_sql = text(
        f"""
        UPDATE {sections_table}
        SET embedding = :embedding
        WHERE section_uuid = :section_uuid
          AND NOT (embedding <=> :embedding)
        """
    )

    updated_rows = 0
    with engine.begin() as conn:
        for i in range(0, len(update_payload), 250):
            chunk = update_payload[i : i + 250]
            result = conn.execute(update_sql, chunk)
            updated_rows += int(result.rowcount or 0)

    context.log.info(
        "9_embed_sections: embedded %s sections, updated=%s.",
        len(section_rows),
        updated_rows,
    )
    run_post_asset_refresh(context, db, pipeline_config)
    return len(section_rows)
