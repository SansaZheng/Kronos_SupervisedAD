"""Restore missing publisher files using pinned URLs and SHA-256 checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data/supervised_financial"


def download(root=DEFAULT_ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / "sources.json").read_text())
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source path escapes dataset root")
        if path.exists():
            content = path.read_bytes()
            if len(content) == item["bytes"] and hashlib.sha256(content).hexdigest() == item["sha256"]:
                print(f"Verified: {item['path']}")
                continue
            raise ValueError(f"Existing file differs; inspect it before replacing: {path}")
        if item.get("download", True) is False:
            raise ValueError(f"Restore this frozen metadata snapshot from Git: {path}")
        request = urllib.request.Request(item["url"], headers={"User-Agent": "KronosAD-research-data/1.0"})
        with urllib.request.urlopen(request, timeout=180) as response:
            content = response.read()
        if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError(f"Download differs from pinned release: {item['path']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        partial.write_bytes(content)
        partial.replace(path)
        print(f"Downloaded: {item['path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    download(parser.parse_args().root)
