"""Run every `CREATE OR REPLACE TABLE` in code/SQL/hypertension_sql_queries.sql
against BigQuery so the PRIME data layer can be rebuilt without the UI.

Ad-hoc `SELECT APPROX_QUANTILES(...)` lookups in the SQL file are skipped: their
results are already baked into the hardcoded BP bounds downstream.
"""

import os
import re
import time
from argparse import ArgumentParser

from bq_config import BILLING_PROJECT, bq_client

CREATE_TABLE_RE = re.compile(
    r"^\s*CREATE\s+OR\s+REPLACE\s+TABLE\s+`([^`]+)`",
    re.IGNORECASE,
)


def default_sql_path():
    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.dirname(current_directory)
    overall_folder = os.path.dirname(parent_directory)
    return os.path.join(overall_folder, "code", "SQL", "hypertension_sql_queries.sql")


def strip_line_comments(sql_text):
    cleaned = []
    for line in sql_text.splitlines():
        idx = line.find("--")
        cleaned.append(line if idx == -1 else line[:idx])
    return "\n".join(cleaned)


def parse_create_statements(sql_text):
    """Return list of (fully_qualified_name, short_name, statement) tuples in file order."""
    stripped = strip_line_comments(sql_text)
    statements = []
    for raw in stripped.split(";"):
        stmt = raw.strip()
        if not stmt:
            continue
        match = CREATE_TABLE_RE.match(stmt)
        if not match:
            continue
        fqn = match.group(1)
        short = fqn.rsplit(".", 1)[-1]
        statements.append((fqn, short, stmt + ";"))
    return statements


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sql",
        default=default_sql_path(),
        help="Path to hypertension_sql_queries.sql (default: repo copy).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and list target tables without running anything.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="TABLE",
        help="Rebuild only the given table (short name, e.g. hypertension_meds). Repeatable.",
    )
    args = parser.parse_args()

    with open(args.sql) as f:
        sql_text = f.read()

    statements = parse_create_statements(sql_text)
    if not statements:
        raise SystemExit(f"No CREATE OR REPLACE TABLE statements found in {args.sql}")

    if args.only:
        wanted = set(args.only)
        filtered = [s for s in statements if s[1] in wanted]
        missing = wanted - {s[1] for s in filtered}
        if missing:
            raise SystemExit(
                f"--only requested tables not found in SQL file: {sorted(missing)}"
            )
        statements = filtered

    total = len(statements)

    if args.dry_run:
        print(f"{total} CREATE OR REPLACE TABLE statements parsed from {args.sql}:")
        for i, (fqn, short, _) in enumerate(statements, 1):
            print(f"  [{i:>2}/{total}] {short}  ({fqn})")
        return

    client = bq_client(BILLING_PROJECT)

    for i, (fqn, short, stmt) in enumerate(statements, 1):
        print(f"[{i}/{total}] building {short} ({fqn}) ...", flush=True)
        start = time.time()
        try:
            client.query(stmt).result()
        except Exception as exc:
            raise RuntimeError(
                f"Statement {i}/{total} failed while building {fqn}: {exc}"
            ) from exc
        print(f"[{i}/{total}] {short} done in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
