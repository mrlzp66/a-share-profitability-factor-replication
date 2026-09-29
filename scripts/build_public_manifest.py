"""Regenerate the public-release file manifest.

The manifest is intentionally built from the publishable repository tree only.
It excludes local raw-data/cache folders, binary backtest objects, credentials,
Jupyter checkpoint folders, and the manifest file itself.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "PUBLIC_RELEASE_MANIFEST.json"
EXCLUDED_DIRS = {".git", "data", "cache", "release", "__pycache__", ".ipynb_checkpoints"}
EXCLUDED_SUFFIXES = {".pkl", ".pickle"}
EXCLUDED_NAME_PARTS = {"token", "secret", "credential"}


def should_include(path: Path) -> bool:
    relative = path.relative_to(REPO)
    if path == MANIFEST:
        return False
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    lower_name = path.name.lower()
    return not any(part in lower_name for part in EXCLUDED_NAME_PARTS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = []
    for path in sorted(REPO.rglob("*")):
        if path.is_file() and should_include(path):
            files.append(
                {
                    "path": path.relative_to(REPO).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    manifest = {
        "package_name": "a-share-profitability-factor-replication",
        "release_baseline": "v0.2-source-weight-replication",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_status": "candidate_not_confirmed",
        "file_count_excluding_manifest": len(files),
        "total_size_bytes_excluding_manifest": sum(item["size_bytes"] for item in files),
        "excluded_categories": [
            "data",
            "cache",
            "pkl_and_pickle_files",
            "credentials_by_filename",
            "notebook_outputs",
            "ipynb_checkpoints",
            "local_release_staging",
        ],
        "files": files,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST}")
    print(f"files={len(files)}, total_bytes={manifest['total_size_bytes_excluding_manifest']}")


if __name__ == "__main__":
    main()
