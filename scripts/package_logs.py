#!/usr/bin/env python3
"""Package a source archive with backend API logs matched to known routes."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROUTE_METHODS = {"get", "post", "put", "delete", "patch", "options", "head"}
REQUEST_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)"
    r".*wip_api\.main:log_request_duration:\d+ - "
    r"(?P<method>[A-Z]+) (?P<path>\S+) (?P<status>\d{3}) - took "
    r"(?P<elapsed>[0-9.]+) ms$"
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


@dataclasses.dataclass(frozen=True)
class TagInfo:
    name: str
    created_at: datetime


@dataclasses.dataclass(frozen=True)
class Route:
    method: str
    path: str
    source: str
    line: int
    regex: re.Pattern[str]
    specificity: tuple[int, int]


@dataclasses.dataclass(frozen=True)
class RequestRecord:
    timestamp: datetime
    method: str
    path: str
    status: int
    elapsed_ms: float
    log_file: Path
    line: int
    route: Route


def run_git(args: list[str], repo: Path) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 7z package containing a git source archive, backend API logs, "
            "and a route-to-log index."
        )
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/backend_api"))
    parser.add_argument("--routes-dir", type=Path, default=Path("src/wip_api/routes"))
    parser.add_argument("--main-file", type=Path, default=Path("src/wip_api/main.py"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=None,
        help="Use an existing source archive instead of generating one with git archive.",
    )
    parser.add_argument(
        "--target-ref",
        default=None,
        help="Git ref to archive. Defaults to the latest recent tag, otherwise HEAD.",
    )
    parser.add_argument("--from-tag", default=None, help="Override the log window start tag.")
    parser.add_argument("--to-tag", default=None, help="Override the log window end tag.")
    parser.add_argument(
        "--tag-lookback-hours",
        type=float,
        default=24.0,
        help="Use the latest tag as the target if it was created within this many hours.",
    )
    parser.add_argument(
        "--threads",
        type=positive_int,
        default=None,
        help="Number of CPU threads passed to 7z as -mmtN.",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep the temporary staging directory for inspection.",
    )
    return parser.parse_args()


def resolve_path(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def local_now() -> datetime:
    return datetime.now().astimezone()


def get_tags(repo: Path) -> list[TagInfo]:
    output = run_git(
        [
            "for-each-ref",
            "--sort=-creatordate",
            "--format=%(refname:short)%00%(creatordate:iso8601-strict)",
            "refs/tags",
        ],
        repo,
    )
    tags: list[TagInfo] = []
    for line in output.splitlines():
        if not line:
            continue
        name, created = line.split("\0", 1)
        tags.append(TagInfo(name=name, created_at=datetime.fromisoformat(created)))
    return tags


def get_tag(tags: list[TagInfo], name: str) -> TagInfo:
    for tag in tags:
        if tag.name == name:
            return tag
    raise SystemExit(f"Tag not found: {name}")


def short_head(repo: Path) -> str:
    return run_git(["rev-parse", "--short", "HEAD"], repo)


def determine_window(
    repo: Path,
    tags: list[TagInfo],
    target_ref: str | None,
    from_tag: str | None,
    to_tag: str | None,
    tag_lookback_hours: float,
) -> tuple[str, str, TagInfo | None, datetime | None, datetime]:
    now = local_now()
    latest_recent = None
    if tags:
        tag_age = now - tags[0].created_at.astimezone(now.tzinfo)
        if timedelta(0) <= tag_age <= timedelta(hours=tag_lookback_hours):
            latest_recent = tags[0]

    if to_tag:
        to_info = get_tag(tags, to_tag)
        selected_ref = target_ref or to_info.name
        selected_label = sanitize(selected_ref)
        end = to_info.created_at
    elif target_ref:
        selected_ref = target_ref
        selected_label = sanitize(
            target_ref if target_ref != "HEAD" else f"HEAD-{short_head(repo)}"
        )
        end = now if target_ref == "HEAD" else now
    elif latest_recent:
        selected_ref = latest_recent.name
        selected_label = sanitize(latest_recent.name)
        end = latest_recent.created_at
    else:
        selected_ref = "HEAD"
        selected_label = f"HEAD-{short_head(repo)}-{now.strftime('%Y%m%d_%H%M%S')}"
        end = now

    if from_tag:
        start_info = get_tag(tags, from_tag)
        start = start_info.created_at
    elif to_tag:
        to_index = next((idx for idx, tag in enumerate(tags) if tag.name == to_tag), None)
        start_info = (
            tags[to_index + 1] if to_index is not None and to_index + 1 < len(tags) else None
        )
        start = start_info.created_at if start_info else None
    elif latest_recent:
        start_info = tags[1] if len(tags) > 1 else None
        start = start_info.created_at if start_info else None
    else:
        start_info = tags[0] if tags else None
        start = start_info.created_at if start_info else None

    return selected_ref, selected_label, start_info, start, end


def imported_route_modules(main_file: Path) -> set[str]:
    tree = ast.parse(main_file.read_text(), filename=str(main_file))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "wip_api.routes":
            modules.update(alias.name for alias in node.names)
    return modules


def included_prefixes(main_file: Path) -> dict[str, str]:
    tree = ast.parse(main_file.read_text(), filename=str(main_file))
    prefixes: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "include_router"
            and isinstance(func.value, ast.Name)
            and func.value.id == "app"
            and node.args
        ):
            continue

        router_arg = node.args[0]
        if not (
            isinstance(router_arg, ast.Attribute)
            and router_arg.attr == "router"
            and isinstance(router_arg.value, ast.Name)
        ):
            continue

        prefix = ""
        for keyword in node.keywords:
            if keyword.arg == "prefix" and isinstance(keyword.value, ast.Constant):
                prefix = str(keyword.value.value)
                break
        prefixes[router_arg.value.id] = prefix
    return prefixes


def route_regex(path: str) -> tuple[re.Pattern[str], tuple[int, int]]:
    pieces: list[str] = []
    literal_chars = 0
    param_count = 0
    cursor = 0
    for match in re.finditer(r"\{([^}:]+)(?::([^}]+))?\}", path):
        literal = path[cursor : match.start()]
        pieces.append(re.escape(literal))
        literal_chars += len(literal)
        param_count += 1
        converter = match.group(2)
        pieces.append(".+" if converter == "path" else "[^/]+")
        cursor = match.end()
    tail = path[cursor:]
    pieces.append(re.escape(tail))
    literal_chars += len(tail)
    return re.compile("^" + "".join(pieces) + "$"), (literal_chars, -param_count)


def discover_routes(repo: Path, routes_dir: Path, main_file: Path) -> list[Route]:
    prefixes = included_prefixes(main_file)
    modules = imported_route_modules(main_file) or {path.stem for path in routes_dir.glob("*.py")}
    routes: list[Route] = []

    for module in sorted(modules):
        route_file = routes_dir / f"{module}.py"
        if not route_file.exists():
            continue
        tree = ast.parse(route_file.read_text(), filename=str(route_file))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                func = decorator.func
                if not (
                    isinstance(func, ast.Attribute)
                    and func.attr in ROUTE_METHODS
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "router"
                    and decorator.args
                    and isinstance(decorator.args[0], ast.Constant)
                ):
                    continue
                method = func.attr.upper()
                path = prefixes.get(module, "") + str(decorator.args[0].value)
                regex, specificity = route_regex(path)
                source = str(route_file.relative_to(repo))
                routes.append(
                    Route(
                        method=method,
                        path=path,
                        source=source,
                        line=decorator.lineno,
                        regex=regex,
                        specificity=specificity,
                    )
                )

    return sorted(routes, key=lambda route: (route.method, route.specificity), reverse=True)


def parse_request_timestamp(value: str, tzinfo) -> datetime:
    parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f")
    return parsed.replace(tzinfo=tzinfo)


def match_route(routes: list[Route], method: str, path: str) -> Route | None:
    candidates = [route for route in routes if route.method == method and route.regex.match(path)]
    if not candidates:
        return None
    return max(candidates, key=lambda route: route.specificity)


def find_records(
    routes: list[Route],
    logs_dir: Path,
    start: datetime | None,
    end: datetime,
) -> list[RequestRecord]:
    tzinfo = end.tzinfo or local_now().tzinfo
    records: list[RequestRecord] = []
    for log_file in sorted(logs_dir.glob("*.log")):
        with log_file.open(errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                match = REQUEST_RE.match(line.rstrip("\n"))
                if not match:
                    continue
                timestamp = parse_request_timestamp(match.group("ts"), tzinfo)
                if start and timestamp <= start:
                    continue
                if timestamp > end:
                    continue
                method = match.group("method")
                path = match.group("path")
                route = match_route(routes, method, path)
                if route is None:
                    continue
                records.append(
                    RequestRecord(
                        timestamp=timestamp,
                        method=method,
                        path=path,
                        status=int(match.group("status")),
                        elapsed_ms=float(match.group("elapsed")),
                        log_file=log_file,
                        line=line_number,
                        route=route,
                    )
                )
    return records


def render_index(
    repo: Path,
    source_archive_name: str,
    selected_ref: str,
    start_info: TagInfo | None,
    start: datetime | None,
    end: datetime,
    routes: list[Route],
    records: list[RequestRecord],
) -> str:
    by_route: dict[tuple[str, str], list[RequestRecord]] = defaultdict(list)
    for record in records:
        by_route[(record.route.method, record.route.path)].append(record)

    matched_files = sorted({record.log_file.relative_to(repo) for record in records})
    generated = local_now().strftime("%Y-%m-%d %H:%M:%S %z")
    lines = [
        "# API Route Log Index",
        "",
        f"- Generated: {generated}",
        f"- Source archive: `{source_archive_name}`",
        f"- Code ref: `{selected_ref}`",
        f"- Log window start: `{start.isoformat() if start else 'beginning'}`"
        + (f" (`{start_info.name}`)" if start_info else ""),
        f"- Log window end: `{end.isoformat()}`",
        f"- Matched request records: {len(records)}",
        f"- Matched log files: {len(matched_files)}",
        "",
        "## Matched Log Files",
        "",
    ]
    if matched_files:
        lines.extend(f"- `{path}`" for path in matched_files)
    else:
        lines.append("- None")

    lines.extend(["", "## Routes With Matched Requests", ""])
    if by_route:
        for route in routes:
            key = (route.method, route.path)
            route_records = by_route.get(key)
            if not route_records:
                continue
            lines.extend(
                [
                    f"### {route.method} `{route.path}`",
                    "",
                    f"Source: `{route.source}:{route.line}`",
                    "",
                    "| time | status | elapsed_ms | request_path | log |",
                    "| --- | ---: | ---: | --- | --- |",
                ]
            )
            for record in sorted(route_records, key=lambda item: item.timestamp):
                log_path = record.log_file.relative_to(repo)
                lines.append(
                    f"| {record.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} "
                    f"| {record.status} | {record.elapsed_ms:.1f} | "
                    f"`{record.path}` | `{log_path}:{record.line}` |"
                )
            lines.append("")
    else:
        lines.append("None")
        lines.append("")

    unmatched = [route for route in routes if (route.method, route.path) not in by_route]
    lines.extend(["## Routes Without Matched Requests", ""])
    if unmatched:
        lines.extend(
            f"- {route.method} `{route.path}` ({route.source}:{route.line})" for route in unmatched
        )
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def copy_package_inputs(
    repo: Path,
    staging: Path,
    source_archive: Path,
    records: list[RequestRecord],
) -> list[str]:
    archive_dest = staging / source_archive.name
    if source_archive.resolve() != archive_dest.resolve():
        shutil.copy2(source_archive, archive_dest)
    relative_inputs = [source_archive.name, "api_route_log_index.md"]

    for log_file in sorted({record.log_file for record in records}):
        rel = log_file.relative_to(repo)
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log_file, dest)
        relative_inputs.append(str(rel))
    return relative_inputs


def create_source_archive(
    repo: Path,
    staging: Path,
    selected_ref: str,
    selected_label: str,
    existing_archive: Path | None,
) -> Path:
    if existing_archive:
        archive = existing_archive if existing_archive.is_absolute() else repo / existing_archive
        if not archive.exists():
            raise SystemExit(f"Source archive not found: {archive}")
        return archive

    archive = staging / f"{selected_label}.tar.gz"
    subprocess.run(["git", "archive", "-o", str(archive), selected_ref], cwd=repo, check=True)
    return archive


def default_output_name(selected_label: str) -> str:
    return f"{selected_label}_logs.7z"


def create_7z(staging: Path, output: Path, inputs: list[str], threads: int | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    cmd = [
        "7z",
        "a",
        str(output),
        "-mx=9",
        "-m0=lzma2",
        "-md=1536m",
        "-mfb=273",
        "-ms=on",
        "-mqs=on",
        *([f"-mmt{threads}"] if threads is not None else []),
        *inputs,
    ]
    subprocess.run(cmd, cwd=staging, check=True)


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    logs_dir = resolve_path(repo, args.logs_dir)
    routes_dir = resolve_path(repo, args.routes_dir)
    main_file = resolve_path(repo, args.main_file)

    tags = get_tags(repo)
    selected_ref, selected_label, start_info, start, end = determine_window(
        repo=repo,
        tags=tags,
        target_ref=args.target_ref,
        from_tag=args.from_tag,
        to_tag=args.to_tag,
        tag_lookback_hours=args.tag_lookback_hours,
    )
    output = args.output
    if output is None:
        output = repo / default_output_name(selected_label)
    else:
        output = output if output.is_absolute() else repo / output

    staging_root = Path(tempfile.mkdtemp(prefix="api-route-log-package-"))
    try:
        routes = discover_routes(repo, routes_dir, main_file)
        records = find_records(routes, logs_dir, start, end)
        source_archive = create_source_archive(
            repo=repo,
            staging=staging_root,
            selected_ref=selected_ref,
            selected_label=selected_label,
            existing_archive=args.source_archive,
        )
        index = render_index(
            repo=repo,
            source_archive_name=source_archive.name,
            selected_ref=selected_ref,
            start_info=start_info,
            start=start,
            end=end,
            routes=routes,
            records=records,
        )
        (staging_root / "api_route_log_index.md").write_text(index)
        inputs = copy_package_inputs(repo, staging_root, source_archive, records)
        create_7z(staging_root, output, inputs, args.threads)

        print(f"created: {output}")
        print(f"source_archive: {source_archive.name}")
        print(f"code_ref: {selected_ref}")
        print(f"log_window_start: {start.isoformat() if start else 'beginning'}")
        print(f"log_window_end: {end.isoformat()}")
        print(f"matched_records: {len(records)}")
        print(f"matched_log_files: {len({record.log_file for record in records})}")
        if args.keep_staging:
            print(f"staging_kept: {staging_root}")
        return 0
    finally:
        if not args.keep_staging:
            shutil.rmtree(staging_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
