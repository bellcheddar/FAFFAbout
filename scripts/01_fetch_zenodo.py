#!/usr/bin/env python
"""01_fetch_zenodo.py: download, verify and extract the PSI TargetTrack archive.

Zenodo record 821654 (DOI 10.5281/zenodo.821654), "Protein Structure Initiative:
TargetTrack 2000-2017, All Data Files". One tarball, ~795 MiB, licence CC-BY-SA-4.0.

Steps
  1. Read the record metadata from the Zenodo API (file name, size, md5, licence).
  2. Download the tarball to data/raw/ (resumable; skipped if present and md5 matches).
  3. Verify the md5 against the record.
  4. Extract into data/raw/TargetTrack/ (skipped if already extracted).
  5. Print an inventory of the Documentation/ folder and locate tt.xml.gz.
  6. Write data/raw/zenodo_record.json so the licence and checksum are on disk.

Usage
  .venv/bin/python scripts/01_fetch_zenodo.py            # full run
  .venv/bin/python scripts/01_fetch_zenodo.py --no-extract
  .venv/bin/python scripts/01_fetch_zenodo.py --force     # re-download even if present
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

RECORD_ID = 821654
DOI = "10.5281/zenodo.821654"
API = f"https://zenodo.org/api/records/{RECORD_ID}"

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXTRACT_DIR = RAW / "TargetTrack"
RECORD_JSON = RAW / "zenodo_record.json"

CHUNK = 1 << 20  # 1 MiB


def fetch_record() -> dict:
    r = requests.get(API, timeout=60)
    r.raise_for_status()
    rec = r.json()
    files = rec.get("files") or []
    if len(files) != 1:
        sys.exit(f"expected exactly one file on record {RECORD_ID}, found {len(files)}")
    f = files[0]
    algo, _, digest = f["checksum"].partition(":")
    if algo != "md5":
        sys.exit(f"unexpected checksum algorithm {algo!r}")
    return {
        "record_id": RECORD_ID,
        "doi": rec.get("doi", DOI),
        "title": rec["metadata"]["title"],
        "licence": rec["metadata"].get("license", {}).get("id"),
        "access_right": rec["metadata"].get("access_right"),
        "publication_date": rec["metadata"].get("publication_date"),
        "file_name": f["key"],
        "file_size": f["size"],
        "md5": digest,
        "url": f["links"]["self"],
    }


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh, tqdm(
        total=path.stat().st_size, unit="B", unit_scale=True, desc="md5", leave=False
    ) as bar:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
            bar.update(len(block))
    return h.hexdigest()


def download(url: str, dest: Path, size: int) -> None:
    """Resumable streaming download."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, stream=True, headers=headers, timeout=120) as r:
        if have and r.status_code != 206:
            # server ignored the range: start over
            have = 0
            tmp.unlink(missing_ok=True)
        r.raise_for_status()
        mode = "ab" if have else "wb"
        with tmp.open(mode) as fh, tqdm(
            total=size, initial=have, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for block in r.iter_content(CHUNK):
                fh.write(block)
                bar.update(len(block))
    tmp.rename(dest)


def safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract, refusing any member that would escape dest."""
    dest = dest.resolve()
    for m in tar.getmembers():
        target = (dest / m.name).resolve()
        if dest not in target.parents and target != dest:
            sys.exit(f"refusing to extract {m.name!r}: escapes {dest}")
    tar.extractall(dest, filter="data")


def inventory(extract_dir: Path) -> None:
    docs = [p for p in extract_dir.rglob("*") if p.is_dir() and p.name.lower() == "documentation"]
    if not docs:
        print("!! no Documentation/ folder found under", extract_dir)
    for d in docs:
        print(f"\nDocumentation inventory ({d.relative_to(extract_dir)}):")
        for p in sorted(d.rglob("*")):
            if p.is_file():
                print(f"  {p.stat().st_size:>12,}  {p.relative_to(d)}")
    xml = list(extract_dir.rglob("tt.xml.gz")) + list(extract_dir.rglob("tt.xml"))
    print("\nTargetTrack XML:")
    for p in xml:
        print(f"  {p.stat().st_size:>12,}  {p.relative_to(extract_dir)}")
    if not xml:
        print("  !! tt.xml.gz not found")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download even if the tarball verifies")
    ap.add_argument("--no-extract", action="store_true", help="download and verify only")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    rec = fetch_record()
    RECORD_JSON.write_text(json.dumps(rec, indent=2))
    print(f"record   : {rec['title']}")
    print(f"doi      : {rec['doi']}")
    print(f"licence  : {rec['licence']}  (record this in README.md before any redistribution)")
    print(f"file     : {rec['file_name']}  {rec['file_size']:,} bytes  md5 {rec['md5']}")

    tarball = RAW / rec["file_name"]
    ok = False
    if tarball.exists() and not args.force:
        if tarball.stat().st_size == rec["file_size"]:
            ok = md5_of(tarball) == rec["md5"]
            print("existing tarball", "verifies" if ok else "FAILS md5, re-downloading")
        else:
            print(f"existing tarball is {tarball.stat().st_size:,} bytes, expected {rec['file_size']:,}: re-downloading")
    if not ok:
        tarball.unlink(missing_ok=True)
        download(rec["url"], tarball, rec["file_size"])
        digest = md5_of(tarball)
        if digest != rec["md5"]:
            sys.exit(f"md5 mismatch after download: {digest} != {rec['md5']}")
        print("download verifies")

    if args.no_extract:
        return

    marker = EXTRACT_DIR / ".extracted"
    if marker.exists():
        print(f"already extracted to {EXTRACT_DIR}")
    else:
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"extracting to {EXTRACT_DIR} ...")
        with tarfile.open(tarball, "r:gz") as tar:
            safe_extract(tar, EXTRACT_DIR)
        marker.write_text(rec["md5"] + "\n")
    inventory(EXTRACT_DIR)


if __name__ == "__main__":
    main()
