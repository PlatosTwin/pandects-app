#!/usr/bin/env python3
"""Generate public bulk-dump schema docs from the live MariaDB schema.

Reads the dump table allowlist (bulk/public_tables.txt) and hand-written
descriptions (bulk/schema_docs/table_docs.yml), introspects the live database
for types/keys/indexes, and writes two generated, checked-in artifacts:

  - bulk/schema_docs/pandects.dbml         -> published to dbdocs.io by CI
  - docs/docs/guides/bulk-data-schema.md   -> deployed with the docs site

Fails if any dump table or column lacks a description, or if a description
refers to a table/column that no longer exists — so schema changes cannot
reach the public dump undocumented (bulk/push_to_r2.sh runs this script).

Deliberately excludes volatile data (row counts) so the outputs only change
when the schema or the prose changes.
"""

from __future__ import annotations

import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

import pymysql
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST_PATH = REPO_ROOT / "bulk" / "public_tables.txt"
TABLE_DOCS_PATH = REPO_ROOT / "bulk" / "schema_docs" / "table_docs.yml"
DBML_OUT_PATH = REPO_ROOT / "bulk" / "schema_docs" / "pandects.dbml"
MARKDOWN_OUT_PATH = REPO_ROOT / "docs" / "docs" / "guides" / "bulk-data-schema.md"
BACKEND_ENV_PATH = REPO_ROOT / "backend" / ".env"

DBDOCS_URL = "https://dbdocs.io/nmbogdan/Pandects"
DUMP_URL = "https://bulk.pandects.org/dumps/latest.sql.gz"
CHECKSUM_URL = "https://bulk.pandects.org/dumps/latest.sql.gz.sha256"
MANIFEST_URL = "https://bulk.pandects.org/dumps/latest.json"


def load_allowlist() -> list[str]:
    tables: list[str] = []
    for line in ALLOWLIST_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tables.append(line)
    if not tables:
        raise SystemExit(f"No tables found in {ALLOWLIST_PATH}")
    return tables


def load_db_credentials() -> dict[str, str]:
    keys = ("MARIADB_HOST", "MARIADB_PORT", "MARIADB_USER", "MARIADB_PASSWORD", "MARIADB_DATABASE")
    creds = {k: os.environ[k] for k in keys if os.environ.get(k)}
    missing = [k for k in keys if k != "MARIADB_PORT" and k not in creds]
    if missing:
        if not BACKEND_ENV_PATH.is_file():
            raise SystemExit(f"Missing {missing} in environment and no {BACKEND_ENV_PATH}")
        for line in BACKEND_ENV_PATH.read_text().splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, _, value = line.partition("=")
            if key in keys and key not in creds:
                creds[key] = value.strip()
        missing = [k for k in keys if k != "MARIADB_PORT" and k not in creds]
        if missing:
            raise SystemExit(f"Missing DB credentials: {missing}")
    return creds


class Column:
    def __init__(self, row: dict[str, object]) -> None:
        self.name = str(row["COLUMN_NAME"])
        self.column_type = str(row["COLUMN_TYPE"])
        self.nullable = row["IS_NULLABLE"] == "YES"
        self.default = row["COLUMN_DEFAULT"]
        self.extra = str(row["EXTRA"] or "")
        self.description = ""


class Index:
    def __init__(self, name: str, unique: bool, columns: list[str]) -> None:
        self.name = name
        self.unique = unique
        self.columns = columns


class Table:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = ""
        self.columns: "OrderedDict[str, Column]" = OrderedDict()
        self.pk_columns: list[str] = []
        self.indexes: list[Index] = []


def introspect(conn: pymysql.connections.Connection, tables: list[str]) -> dict[str, Table]:
    placeholders = ", ".join(["%s"] * len(tables))
    schema: dict[str, Table] = {name: Table(name) for name in tables}

    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(
            f"""
            SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_DEFAULT, EXTRA
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME IN ({placeholders})
            ORDER BY TABLE_NAME, ORDINAL_POSITION
            """,
            tables,
        )
        for row in cur.fetchall():
            table = schema[str(row["TABLE_NAME"])]
            table.columns[str(row["COLUMN_NAME"])] = Column(row)

        cur.execute(
            f"""
            SELECT TABLE_NAME, INDEX_NAME, NON_UNIQUE, SEQ_IN_INDEX, COLUMN_NAME
            FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME IN ({placeholders})
            ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX
            """,
            tables,
        )
        grouped: "OrderedDict[tuple[str, str], dict[str, object]]" = OrderedDict()
        for row in cur.fetchall():
            key = (str(row["TABLE_NAME"]), str(row["INDEX_NAME"]))
            entry = grouped.setdefault(key, {"unique": row["NON_UNIQUE"] == 0, "columns": []})
            entry["columns"].append(str(row["COLUMN_NAME"]))  # type: ignore[union-attr]
        for (table_name, index_name), entry in grouped.items():
            table = schema[table_name]
            columns = list(entry["columns"])  # type: ignore[arg-type]
            if index_name == "PRIMARY":
                table.pk_columns = columns
            else:
                table.indexes.append(Index(index_name, bool(entry["unique"]), columns))
        for table in schema.values():
            table.indexes.sort(key=lambda index: index.name)

    empty = [name for name, table in schema.items() if not table.columns]
    if empty:
        raise SystemExit(f"Tables in allowlist but not in database: {empty}")
    return schema


def fetch_enforced_fk_refs(conn: pymysql.connections.Connection, tables: list[str]) -> list[str]:
    placeholders = ", ".join(["%s"] * len(tables))
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(
            f"""
            SELECT TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
            FROM information_schema.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = DATABASE()
              AND REFERENCED_TABLE_NAME IS NOT NULL
              AND TABLE_NAME IN ({placeholders})
              AND REFERENCED_TABLE_NAME IN ({placeholders})
            ORDER BY TABLE_NAME, COLUMN_NAME
            """,
            tables + tables,
        )
        return [
            f"{row['TABLE_NAME']}.{row['COLUMN_NAME']} > "
            f"{row['REFERENCED_TABLE_NAME']}.{row['REFERENCED_COLUMN_NAME']}"
            for row in cur.fetchall()
        ]


def apply_docs(schema: dict[str, Table], docs: dict[str, object], allowlist: list[str]) -> list[dict[str, object]]:
    """Merge prose onto the schema; validate exact table/column coverage."""
    errors: list[str] = []
    documented_tables = docs["tables"]
    assert isinstance(documented_tables, dict)

    for name in allowlist:
        if name not in documented_tables:
            errors.append(f"table `{name}` is in the dump but has no entry in table_docs.yml")
    for name in documented_tables:
        if name not in schema:
            errors.append(f"table_docs.yml documents `{name}`, which is not in the dump allowlist")

    for name, table in schema.items():
        entry = documented_tables.get(name)
        if not isinstance(entry, dict):
            continue
        table.description = str(entry.get("description", "")).strip()
        if not table.description:
            errors.append(f"table `{name}` has no description")
        column_docs = entry.get("columns") or {}
        for column_name, column in table.columns.items():
            description = column_docs.get(column_name)
            if not description or not str(description).strip():
                errors.append(f"column `{name}.{column_name}` has no description")
            else:
                column.description = str(description).strip()
        for column_name in column_docs:
            if column_name not in table.columns:
                errors.append(f"table_docs.yml documents `{name}.{column_name}`, which does not exist")

    groups = docs.get("groups")
    assert isinstance(groups, list)
    grouped_tables = [name for group in groups for name in group["tables"]]
    if sorted(grouped_tables) != sorted(allowlist):
        extra = set(grouped_tables) - set(allowlist)
        missing = set(allowlist) - set(grouped_tables)
        duplicated = {name for name in grouped_tables if grouped_tables.count(name) > 1}
        errors.append(
            f"groups must cover each dump table exactly once "
            f"(missing: {sorted(missing)}, extra: {sorted(extra)}, duplicated: {sorted(duplicated)})"
        )

    if errors:
        print("Schema docs are out of sync with the database:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(1)
    return groups


ENUM_RE = re.compile(r"^enum\((.*)\)$")


def parse_enum_values(column_type: str) -> list[str] | None:
    match = ENUM_RE.match(column_type)
    if not match:
        return None
    return [value[1:-1].replace("''", "'") for value in re.findall(r"'(?:[^']|'')*'", match.group(1))]


def collect_enums(
    schema: dict[str, Table], ordered_tables: list[str]
) -> dict[tuple[str, tuple[str, ...]], str]:
    """Map (column name, values) to DBML enum names; same-named columns with the
    same values share one enum across tables."""
    enums: dict[tuple[str, tuple[str, ...]], str] = {}
    used_names: set[str] = set()
    for table_name in ordered_tables:
        for column in schema[table_name].columns.values():
            values = parse_enum_values(column.column_type)
            if values is None:
                continue
            key = (column.name, tuple(values))
            if key in enums:
                continue
            name = column.name if column.name not in used_names else f"{table_name}_{column.name}"
            enums[key] = name
            used_names.add(name)
    return enums


def dbml_quote(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def dbml_type(column: Column, enums: dict[tuple[str, tuple[str, ...]], str]) -> str:
    values = parse_enum_values(column.column_type)
    if values is not None:
        return enums[(column.name, tuple(values))]
    if re.fullmatch(r"[A-Za-z0-9_]+(\([0-9, ]*\))?", column.column_type):
        return column.column_type.replace(", ", ",")
    return '"' + column.column_type + '"'


def dbml_default(column: Column) -> str | None:
    default = column.default
    if default is None or str(default).upper() == "NULL":
        return None
    text = str(default)
    if text.endswith(")"):  # expression, e.g. utc_timestamp()
        return f"`{text}`"
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return text
    return dbml_quote(text.strip("'"))


def render_dbml(
    schema: dict[str, Table],
    groups: list[dict[str, object]],
    refs: list[str],
    enums: dict[tuple[str, tuple[str, ...]], str],
) -> str:
    lines: list[str] = [
        "// GENERATED FILE — do not edit by hand.",
        "// Built by bulk/schema_docs/generate_schema_docs.py from the live schema",
        "// plus bulk/schema_docs/table_docs.yml. Regenerated by bulk/push_to_r2.sh.",
        "",
        "Project pandects {",
        "  database_type: 'MariaDB'",
        "  Note: '''",
        "    Schema of the Pandects public bulk dump — every table in",
        f"    {DUMP_URL}.",
        f"    Full column reference: https://pandects.org/docs (Bulk Data Schema guide).",
        "  '''",
        "}",
        "",
    ]

    for (_, values), name in enums.items():
        lines.append(f"Enum {name} {{")
        for value in values:
            lines.append(f'  "{value}"')
        lines.append("}")
        lines.append("")

    ordered_tables = [name for group in groups for name in group["tables"]]
    for table_name in ordered_tables:
        table = schema[table_name]
        lines.append(f"Table {table.name} {{")
        for column in table.columns.values():
            attrs: list[str] = []
            if [column.name] == table.pk_columns:
                attrs.append("pk")
            if not column.nullable:
                attrs.append("not null")
            if "auto_increment" in column.extra:
                attrs.append("increment")
            default = dbml_default(column)
            if default is not None:
                attrs.append(f"default: {default}")
            attrs.append(f"note: {dbml_quote(column.description)}")
            lines.append(f"  {column.name} {dbml_type(column, enums)} [{', '.join(attrs)}]")

        index_lines: list[str] = []
        if len(table.pk_columns) > 1:
            index_lines.append(f"    ({', '.join(table.pk_columns)}) [pk]")
        for index in table.indexes:
            columns = f"({', '.join(index.columns)})" if len(index.columns) > 1 else index.columns[0]
            flags = ["unique"] if index.unique else []
            flags.append(f"name: {dbml_quote(index.name)}")
            index_lines.append(f"    {columns} [{', '.join(flags)}]")
        if index_lines:
            lines.append("")
            lines.append("  indexes {")
            lines.extend(index_lines)
            lines.append("  }")

        lines.append("")
        lines.append(f"  Note: {dbml_quote(table.description)}")
        lines.append("}")
        lines.append("")

    for group in groups:
        identifier = re.sub(r"_+", "_", re.sub(r"[^a-z0-9]", "_", str(group["name"]).lower())).strip("_")
        lines.append(f"TableGroup {identifier} {{")
        for table_name in group["tables"]:
            lines.append(f"  {table_name}")
        lines.append("}")
        lines.append("")

    for ref in refs:
        lines.append(f"Ref: {ref}")
    lines.append("")
    return "\n".join(lines)


def render_markdown(
    schema: dict[str, Table],
    groups: list[dict[str, object]],
    refs: list[str],
) -> str:
    refs_by_table: dict[str, list[str]] = {}
    for ref in refs:
        left, op, right = ref.split(" ")
        left_table = left.split(".")[0]
        right_table = right.split(".")[0]
        if op == ">":
            refs_by_table.setdefault(left_table, []).append(f"`{left}` → `{right}` (many-to-one)")
            refs_by_table.setdefault(right_table, []).append(f"`{left}` → `{right}` (one-to-many)")
        else:
            refs_by_table.setdefault(left_table, []).append(f"`{left}` ↔ `{right}` (one-to-one)")
            refs_by_table.setdefault(right_table, []).append(f"`{left}` ↔ `{right}` (one-to-one)")

    lines: list[str] = [
        "---",
        "id: bulk-data-schema",
        "title: Bulk Data Schema",
        "description: Table-by-table reference for the Pandects public database dump.",
        "sidebar_position: 4",
        "---",
        "",
        "{/* GENERATED FILE — do not edit by hand.",
        "    Built by bulk/schema_docs/generate_schema_docs.py from the live schema",
        "    plus bulk/schema_docs/table_docs.yml. Regenerated by bulk/push_to_r2.sh. */}",
        "",
        "# Bulk Data Schema",
        "",
        "Pandects publishes its full public dataset as a MariaDB dump on Cloudflare R2.",
        "This page documents every table in that dump.",
        "",
        "## Getting the dump",
        "",
        f"- **Dump**: [{DUMP_URL}]({DUMP_URL})",
        f"- **Checksum**: [{CHECKSUM_URL}]({CHECKSUM_URL})",
        f"- **Manifest** (size, SHA-256, timestamp): [{MANIFEST_URL}]({MANIFEST_URL})",
        "",
        "Restore into a local MariaDB database:",
        "",
        "```bash",
        f"curl -LO {DUMP_URL}",
        "gunzip latest.sql.gz",
        'mysql -e "CREATE DATABASE pandects"',
        "mysql pandects < latest.sql",
        "```",
        "",
        "The dump targets MariaDB 11+. `sections.embedding` uses the MariaDB",
        "`VECTOR` type; on servers without vector support, load the dump with that",
        "column's DDL adjusted or skip the `sections` table.",
        "",
        f"An interactive ER diagram of this schema is published at [dbdocs.io]({DBDOCS_URL}).",
        "",
        "## Conventions",
        "",
        "- UUIDs are stored as `char(36)` strings.",
        "- Boolean flags are `tinyint(1)` with values 0/1.",
        "- Timestamps are UTC.",
        "- Money columns are USD.",
        "",
    ]

    for group in groups:
        lines.append(f"## {group['name']}")
        lines.append("")
        description = str(group.get("description", "")).strip()
        if description:
            lines.append(description)
            lines.append("")
        for table_name in group["tables"]:
            table = schema[table_name]
            lines.append(f"### `{table.name}`")
            lines.append("")
            lines.append(table.description)
            lines.append("")
            if table.pk_columns:
                pk = ", ".join(f"`{column}`" for column in table.pk_columns)
                lines.append(f"**Primary key**: {pk}")
                lines.append("")
            table_refs = refs_by_table.get(table_name)
            if table_refs:
                lines.append("**Relationships**:")
                lines.extend(f"- {ref}" for ref in table_refs)
                lines.append("")
            lines.append("| Column | Type | Nullable | Description |")
            lines.append("| --- | --- | --- | --- |")
            for column in table.columns.values():
                nullable = "yes" if column.nullable else "no"
                description = column.description.replace("|", "\\|")
                lines.append(f"| `{column.name}` | `{column.column_type}` | {nullable} | {description} |")
            lines.append("")
            if table.indexes:
                index_parts = []
                for index in table.indexes:
                    columns = ", ".join(f"`{column}`" for column in index.columns)
                    unique = "unique, " if index.unique else ""
                    index_parts.append(f"`{index.name}` ({unique}{columns})")
                lines.append(f"**Indexes**: {'; '.join(index_parts)}")
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    allowlist = load_allowlist()
    docs = yaml.safe_load(TABLE_DOCS_PATH.read_text())
    creds = load_db_credentials()

    conn = pymysql.connect(
        host=creds["MARIADB_HOST"],
        port=int(creds.get("MARIADB_PORT", "3306")),
        user=creds["MARIADB_USER"],
        password=creds["MARIADB_PASSWORD"],
        database=creds["MARIADB_DATABASE"],
    )
    try:
        schema = introspect(conn, allowlist)
        enforced_refs = fetch_enforced_fk_refs(conn, allowlist)
    finally:
        conn.close()

    groups = apply_docs(schema, docs, allowlist)

    declared_refs = [str(ref) for ref in docs.get("relationships", [])]
    refs = list(dict.fromkeys(enforced_refs + declared_refs))
    known_endpoints = {
        f"{table.name}.{column}" for table in schema.values() for column in table.columns
    }
    for ref in refs:
        parts = ref.split(" ")
        if len(parts) != 3 or parts[1] not in {">", "<", "-"}:
            raise SystemExit(f"Malformed relationship: {ref!r}")
        for endpoint in (parts[0], parts[2]):
            if endpoint not in known_endpoints:
                raise SystemExit(f"Relationship references unknown column: {endpoint} (in {ref!r})")

    ordered_tables = [name for group in groups for name in group["tables"]]
    enums = collect_enums(schema, ordered_tables)

    DBML_OUT_PATH.write_text(render_dbml(schema, groups, refs, enums))
    MARKDOWN_OUT_PATH.write_text(render_markdown(schema, groups, refs))
    print(f"Wrote {DBML_OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {MARKDOWN_OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
