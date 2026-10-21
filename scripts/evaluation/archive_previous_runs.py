"""Archive inventoried historical artifacts after the replacement report is complete.

Does not delete anything. Produces a narrowly scoped cleanup plan for verified
duplicate APP artifacts; primary historical checkpoints/reports remain readable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new-run", required=True)
    p.add_argument("--inventory", required=True)
    p.add_argument("--source-backup", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    resolve = lambda name: (ROOT / name).resolve()
    run, inventory, backup, output = map(resolve, (
        args.new_run, args.inventory, args.source_backup, args.output))
    for path in (run, inventory, backup, output):
        if not path.is_relative_to(ROOT) or path == ROOT:
            raise ValueError("Archive paths must stay within the project.")
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    if read(run / "run_manifest.json")["status"] != "report_completed":
        raise ValueError("New experiment report has not completed.")
    verification = read(run / "evaluation_summary.json")["verification"]
    if not all(verification.values()):
        raise ValueError("New report verification is incomplete.")
    records = read(inventory)["files"]
    for rec in records:
        path = resolve(rec["path"])
        if not path.is_relative_to(ROOT / "outputs") or not path.is_file():
            raise ValueError(f"Invalid historical artifact: {path}")
        if sha(path) != rec["sha256"]:
            raise ValueError(f"Historical artifact changed: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.zip")
    if output.exists() or partial.exists():
        raise FileExistsError("Archive already exists; inspect it rather than overwrite.")
    with zipfile.ZipFile(partial, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for rec in records:
            archive.write(resolve(rec["path"]), rec["path"])
        archive.write(backup, "before_changes_source.zip")
        archive.writestr("archive_manifest.json", json.dumps(read(inventory), ensure_ascii=False, indent=2))
        archive.writestr("RESTORE.txt",
            "Extract under a separate project directory or restore selected original relative paths.\n"
            "before_changes_source.zip contains the pre-upgrade source/configuration snapshot.\n"
            "Raw enterprise CSVs and the Python environment are not deleted or replaced by this archive.\n"
            "Main historical checkpoints and reports remain available; only listed duplicate copies may be cleaned.\n")
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC verification failed.")
        for rec in records:
            with archive.open(rec["path"]) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != rec["sha256"]:
                    raise ValueError(f"ZIP hash verification failed: {rec['path']}")
        with archive.open("before_changes_source.zip") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != sha(backup):
                raise ValueError("Source backup verification failed.")
    partial.replace(output)
    duplicate_dir = ROOT / "outputs/app/dk_3x3_v1_20261019_reproduction"
    original_dir = ROOT / "outputs/app/dk_3x3_v1_20261019"
    cleanup = []
    if duplicate_dir.is_dir():
        duplicates = [f for f in duplicate_dir.rglob("*") if f.is_file()]
        original_rel = {f.relative_to(original_dir) for f in original_dir.rglob("*") if f.is_file()}
        if ({f.relative_to(duplicate_dir) for f in duplicates} == original_rel
                and all(sha(f) == sha(original_dir / f.relative_to(duplicate_dir)) for f in duplicates)):
            cleanup.append(str(duplicate_dir.resolve()))
    duplicate_zip = ROOT / "outputs/app/dk_3x3_v1_20261019.zip"
    if duplicate_zip.is_file() and any(resolve(r["path"]) == duplicate_zip for r in records):
        cleanup.append(str(duplicate_zip.resolve()))
    result = {"archive": str(output), "archive_sha256": sha(output),
              "historical_files_verified": len(records), "source_backup_verified": True,
              "new_run": str(run), "cleanup_candidates": cleanup,
              "cleanup_executed": False,
              "policy": "Only verified duplicate APP copies; retain historical models and reports."}
    output.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
