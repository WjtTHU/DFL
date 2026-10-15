import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError

from wip_analysis.data_loader.sql_template import _load_sql_conf

_CONFIG_REF_RE = re.compile(r"\b(?P<section>tables|columns)\.(?P<key>[A-Za-z_][A-Za-z0-9_]*)")
_CREATE_OBJECT_RE = re.compile(
    r"\bCREATE\s+(?P<kind>TABLE|MATERIALIZED\s+VIEW)\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>\"[^\"]+\"|[^\s(]+)",
    re.IGNORECASE,
)
_COLUMN_CONSTRAINTS = {
    "CHECK",
    "CONSTRAINT",
    "EXCLUDE",
    "FOREIGN",
    "LIKE",
    "PRIMARY",
    "UNIQUE",
}
_COLUMN_TYPE_STOP_WORDS = {
    "CHECK",
    "COLLATE",
    "CONSTRAINT",
    "DEFAULT",
    "GENERATED",
    "IDENTITY",
    "NOT",
    "NULL",
    "PRIMARY",
    "REFERENCES",
    "UNIQUE",
}

_PG_COLUMNS_QUERY = """
SELECT
    c.relkind,
    a.attname,
    pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type
FROM pg_catalog.pg_class AS c
JOIN pg_catalog.pg_attribute AS a
  ON a.attrelid = c.oid
WHERE c.oid = to_regclass(%s)
  AND a.attnum > 0
  AND NOT a.attisdropped
ORDER BY a.attnum
"""
_RELKIND_TO_KIND = {
    "m": "MATERIALIZED VIEW",
    "p": "TABLE",
    "r": "TABLE",
    "v": "VIEW",
}


class SqlSchemaConfigError(RuntimeError):
    """Raised when SQL config mappings do not match schema declarations."""


@dataclass(frozen=True)
class _SchemaColumn:
    data_type: str


@dataclass(frozen=True)
class _SchemaObject:
    kind: str
    columns: Mapping[str, _SchemaColumn]


@dataclass(frozen=True)
class _TemplateRefs:
    tables: frozenset[str]
    columns: frozenset[str]


def validate_sql_schema_config(
    *,
    config_path: str | Path = "config/sql_base.yaml",
    schema_dir: str | Path = "sql/schemas",
) -> None:
    """Validate configured table and column mappings against schema declarations.

    The check is intentionally static: it verifies the checked-in SQL config and
    schema templates before the API opens database connections.
    """
    expected = _build_expected_schema(config_path=config_path, schema_dir=schema_dir)
    errors = _validate_expected_schema_declarations(expected)
    if errors:
        raise SqlSchemaConfigError(
            _format_errors(expected.config_path, expected.schema_dir, errors)
        )


def validate_sql_schema_config_against_database(
    conn_pool: Any,
    *,
    config_path: str | Path = "config/sql_base.yaml",
    schema_dir: str | Path = "sql/schemas",
) -> None:
    """Validate actual database tables against the rendered sql/schemas declarations."""
    expected = _build_expected_schema(config_path=config_path, schema_dir=schema_dir)
    actual_objects = _fetch_actual_schema_objects(
        conn_pool,
        table_keys=expected.used_table_keys,
        tables=expected.tables,
    )

    errors = _compare_schema_objects(
        expected_objects=expected.schema_objects,
        actual_objects=actual_objects,
        table_keys=expected.used_table_keys,
        tables=expected.tables,
    )
    if errors:
        raise SqlSchemaConfigError(
            _format_errors(expected.config_path, expected.schema_dir, errors)
        )


@dataclass(frozen=True)
class _ExpectedSchema:
    config_path: Path
    schema_dir: Path
    tables: Mapping[str, object]
    columns: Mapping[str, object]
    schema_objects: Mapping[str, _SchemaObject]
    used_table_keys: frozenset[str]
    used_column_keys: frozenset[str]


def _build_expected_schema(
    *,
    config_path: str | Path,
    schema_dir: str | Path,
) -> _ExpectedSchema:
    config_path = Path(config_path)
    schema_dir = Path(schema_dir)

    conf = _load_sql_conf(str(config_path))
    db = conf.get("database", {})
    tables = _require_mapping(db.get("tables"), "database.tables")
    columns = _require_mapping(db.get("columns"), "database.columns")
    templates_dir = Path(db.get("templates", {}).get("base_dir", "sql/templates"))

    schema_files = sorted(schema_dir.glob("*.sql.j2"))
    if not schema_files:
        raise SqlSchemaConfigError(f"No SQL schema templates found in {schema_dir}")

    schema_refs = _collect_template_refs(schema_files)
    query_template_files = sorted(templates_dir.glob("*.sql.j2"))
    query_refs = (
        _collect_template_refs(query_template_files)
        if query_template_files
        else _TemplateRefs(
            frozenset(),
            frozenset(),
        )
    )

    errors: list[str] = []
    errors.extend(_missing_ref_errors("sql/schemas", schema_refs, tables, columns, config_path))
    errors.extend(_missing_ref_errors(str(templates_dir), query_refs, tables, columns, config_path))

    if errors:
        raise SqlSchemaConfigError(_format_errors(config_path, schema_dir, errors))

    rendered_schemas = _render_schema_templates(schema_files, tables=tables, columns=columns)
    schema_objects = _extract_schema_objects(rendered_schemas, tables)

    used_table_keys = schema_refs.tables | query_refs.tables
    used_column_keys = schema_refs.columns | query_refs.columns

    return _ExpectedSchema(
        config_path=config_path,
        schema_dir=schema_dir,
        tables=tables,
        columns=columns,
        schema_objects=schema_objects,
        used_table_keys=used_table_keys,
        used_column_keys=used_column_keys,
    )


def _validate_expected_schema_declarations(expected: _ExpectedSchema) -> list[str]:
    errors: list[str] = []

    missing_table_declarations = sorted(expected.used_table_keys - set(expected.schema_objects))
    if missing_table_declarations:
        errors.append(
            "Tables referenced by SQL config/templates but not created in sql/schemas: "
            + ", ".join(
                f"tables.{key} -> {expected.tables[key]!r}" for key in missing_table_declarations
            )
        )

    declared_columns = frozenset(
        column
        for schema_object in expected.schema_objects.values()
        for column in schema_object.columns
    )
    missing_column_declarations = sorted(
        key
        for key in expected.used_column_keys
        if key in expected.columns
        and _normalize_identifier(str(expected.columns[key])) not in declared_columns
    )
    if missing_column_declarations:
        errors.append(
            "Columns referenced by SQL config/templates but not declared in sql/schemas: "
            + ", ".join(
                f"columns.{key} -> {expected.columns[key]!r}" for key in missing_column_declarations
            )
        )

    return errors


def _require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SqlSchemaConfigError(f"SQL config must contain a mapping at {name}")
    return dict(value)


def _collect_template_refs(template_paths: list[Path]) -> _TemplateRefs:
    table_refs: set[str] = set()
    column_refs: set[str] = set()

    for template_path in template_paths:
        text = template_path.read_text()
        for match in _CONFIG_REF_RE.finditer(text):
            target = table_refs if match.group("section") == "tables" else column_refs
            target.add(match.group("key"))

    return _TemplateRefs(frozenset(table_refs), frozenset(column_refs))


def _missing_ref_errors(
    label: str,
    refs: _TemplateRefs,
    tables: Mapping[str, object],
    columns: Mapping[str, object],
    config_path: Path,
) -> list[str]:
    errors: list[str] = []

    missing_tables = sorted(refs.tables - set(tables))
    if missing_tables:
        errors.append(
            f"{label} references table keys missing from {config_path}: "
            + ", ".join(f"tables.{key}" for key in missing_tables)
        )

    missing_columns = sorted(refs.columns - set(columns))
    if missing_columns:
        errors.append(
            f"{label} references column keys missing from {config_path}: "
            + ", ".join(f"columns.{key}" for key in missing_columns)
        )

    return errors


def _render_schema_templates(
    schema_files: list[Path],
    *,
    tables: Mapping[str, object],
    columns: Mapping[str, object],
) -> dict[Path, str]:
    env = Environment(
        loader=FileSystemLoader(str(schema_files[0].parent)),
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
        keep_trailing_newline=False,
        undefined=StrictUndefined,
    )

    rendered: dict[Path, str] = {}
    for schema_file in schema_files:
        try:
            rendered[schema_file] = env.get_template(schema_file.name).render(
                tables=tables,
                columns=columns,
            )
        except TemplateError as exc:
            raise SqlSchemaConfigError(
                f"Failed to render SQL schema template {schema_file}: {exc}"
            ) from exc

    return rendered


def _fetch_actual_schema_objects(
    conn_pool: Any,
    *,
    table_keys: Iterable[str],
    tables: Mapping[str, object],
) -> dict[str, _SchemaObject]:
    schema_objects: dict[str, _SchemaObject] = {}

    with conn_pool.connection() as conn:
        with conn.cursor() as cursor:
            for table_key in sorted(table_keys):
                cursor.execute(_PG_COLUMNS_QUERY, (str(tables[table_key]),))
                rows = cursor.fetchall()
                if not rows:
                    continue

                relkind = str(rows[0][0])
                columns = {
                    _normalize_identifier(str(row[1])): _SchemaColumn(
                        data_type=_canonicalize_data_type(str(row[2]))
                    )
                    for row in rows
                }
                schema_objects[table_key] = _SchemaObject(
                    kind=_RELKIND_TO_KIND.get(relkind, relkind),
                    columns=columns,
                )

    return schema_objects


def _compare_schema_objects(
    *,
    expected_objects: Mapping[str, _SchemaObject],
    actual_objects: Mapping[str, _SchemaObject],
    table_keys: Iterable[str],
    tables: Mapping[str, object],
) -> list[str]:
    errors: list[str] = []

    for table_key in sorted(table_keys):
        if table_key not in expected_objects:
            continue

        expected = expected_objects[table_key]
        actual = actual_objects.get(table_key)
        table_label = f"tables.{table_key} -> {tables[table_key]!r}"
        if actual is None:
            errors.append(f"Database object not found for {table_label}")
            continue

        if expected.kind != actual.kind:
            errors.append(
                f"Database object kind mismatch for {table_label}: "
                f"expected {expected.kind}, actual {actual.kind}"
            )

        expected_columns = set(expected.columns)
        actual_columns = set(actual.columns)

        missing_columns = sorted(expected_columns - actual_columns)
        if missing_columns:
            errors.append(
                f"Database object {table_label} is missing columns declared in sql/schemas: "
                + ", ".join(missing_columns)
            )

        extra_columns = sorted(actual_columns - expected_columns)
        if extra_columns:
            errors.append(
                f"Database object {table_label} has columns not declared in sql/schemas: "
                + ", ".join(extra_columns)
            )

        for column in sorted(expected_columns & actual_columns):
            expected_type = expected.columns[column].data_type
            actual_type = actual.columns[column].data_type
            if expected_type != actual_type:
                errors.append(
                    f"Database column type mismatch for {table_label}.{column}: "
                    f"expected {expected_type}, actual {actual_type}"
                )

    return errors


def _extract_schema_objects(
    rendered_schemas: Mapping[Path, str],
    tables: Mapping[str, object],
) -> dict[str, _SchemaObject]:
    table_key_by_name = {
        _normalize_identifier(str(table_name)): key for key, table_name in tables.items()
    }
    schema_objects: dict[str, _SchemaObject] = {}

    for sql_text in rendered_schemas.values():
        sql_text = _strip_line_comments(sql_text)
        for match in _CREATE_OBJECT_RE.finditer(sql_text):
            object_name = _normalize_identifier(match.group("name"))
            table_key = table_key_by_name.get(object_name)
            if table_key is None:
                continue

            kind = re.sub(r"\s+", " ", match.group("kind").upper())
            columns: dict[str, _SchemaColumn] = {}
            if kind == "TABLE":
                columns = _extract_create_table_columns(sql_text, match.end())
            schema_objects[table_key] = _SchemaObject(kind=kind, columns=columns)

    # clean_snapshot.sql.j2 declares snapshot_table as SELECT h.* from raw_table.
    # Treat it as exposing the same configured columns as raw_table for static checks.
    snapshot = schema_objects.get("snapshot_table")
    raw = schema_objects.get("raw_table")
    if snapshot is not None and not snapshot.columns and raw is not None:
        schema_objects["snapshot_table"] = _SchemaObject(
            kind=snapshot.kind,
            columns=raw.columns,
        )

    return schema_objects


def _extract_create_table_columns(sql_text: str, search_start: int) -> dict[str, _SchemaColumn]:
    open_paren = sql_text.find("(", search_start)
    if open_paren < 0:
        return {}

    close_paren = _find_matching_paren(sql_text, open_paren)
    if close_paren < 0:
        return {}

    columns: dict[str, _SchemaColumn] = {}
    for item in _split_top_level_commas(sql_text[open_paren + 1 : close_paren]):
        item = item.strip()
        first_token = _first_identifier(item)
        if first_token and first_token.upper() not in _COLUMN_CONSTRAINTS:
            data_type = _extract_column_type(item, first_token)
            if data_type:
                columns[_normalize_identifier(first_token)] = _SchemaColumn(
                    data_type=_canonicalize_data_type(data_type)
                )
    return columns


def _extract_column_type(column_def: str, column_name: str) -> str:
    remaining = column_def[len(column_name) :].strip()
    type_tokens: list[str] = []
    for token in _split_top_level_whitespace(remaining):
        if token.upper() in _COLUMN_TYPE_STOP_WORDS:
            break
        type_tokens.append(token)
    return " ".join(type_tokens)


def _split_top_level_whitespace(text: str) -> list[str]:
    tokens: list[str] = []
    start: int | None = None
    depth = 0

    for idx, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1

        if char.isspace() and depth == 0:
            if start is not None:
                tokens.append(text[start:idx])
                start = None
        elif start is None:
            start = idx

    if start is not None:
        tokens.append(text[start:])
    return tokens


def _canonicalize_data_type(data_type: str) -> str:
    normalized = data_type.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.replace(" []", "[]")
    normalized = normalized.replace("[ ]", "[]")

    aliases = {
        "bool": "boolean",
        "int": "integer",
        "int4": "integer",
        "int8": "bigint",
        "timestamp": "timestamp without time zone",
        "varchar": "character varying",
    }
    if normalized in aliases:
        return aliases[normalized]
    if normalized.startswith("varchar("):
        return "character varying" + normalized[len("varchar") :]
    return normalized


def _strip_line_comments(sql_text: str) -> str:
    return "\n".join(line.split("--", 1)[0] for line in sql_text.splitlines())


def _find_matching_paren(text: str, open_paren: int) -> int:
    depth = 0
    for idx in range(open_paren, len(text)):
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _split_top_level_commas(text: str) -> list[str]:
    items: list[str] = []
    start = 0
    depth = 0
    for idx, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            items.append(text[start:idx])
            start = idx + 1
    items.append(text[start:])
    return items


def _first_identifier(text: str) -> str | None:
    match = re.match(r'(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_$]*)', text)
    return match.group(0) if match else None


def _normalize_identifier(identifier: str) -> str:
    identifier = identifier.strip()
    if "." in identifier and not identifier.startswith('"'):
        identifier = identifier.rsplit(".", 1)[-1]
    if len(identifier) >= 2 and identifier[0] == '"' and identifier[-1] == '"':
        return identifier[1:-1]
    return identifier.lower()


def _format_errors(config_path: Path, schema_dir: Path, errors: list[str]) -> str:
    details = "\n".join(f"- {error}" for error in errors)
    return f"SQL schema config validation failed for {config_path} against {schema_dir}:\n{details}"
