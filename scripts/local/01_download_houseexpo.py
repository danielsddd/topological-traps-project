#!/usr/bin/env python3
"""
Phase 1 — Download HouseExpo dataset.

Downloads JSON floor plan files directly from the public HouseExpo GitHub repo.
No account or API key required.

Usage:
    python scripts/01_download_houseexpo.py
    python scripts/01_download_houseexpo.py --num-maps 100   # small test batch
    python scripts/01_download_houseexpo.py --num-maps 800   # full dataset
"""

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
import tarfile  
import requests
from tqdm import tqdm


# HouseExpo public GitHub repo — no auth needed
HOUSEEXPO_ZIP_URL = (
    "https://github.com/TeaganLi/HouseExpo/archive/refs/heads/master.zip"
)

# Where JSON files live inside the extracted zip
ZIP_JSON_SUBDIR = "HouseExpo-master/HouseExpo/json"

# Where we put them
OUTPUT_DIR = Path("data/raw_maps")
DOWNLOAD_CACHE = Path("data/.download_cache")


def parse_args():
    p = argparse.ArgumentParser(description="Download HouseExpo dataset")
    p.add_argument(
        "--num-maps",
        type=int,
        default=800,
        help="How many JSON files to keep (default: 800). Use 50 for a quick test.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where to place JSON files (default: data/raw_maps)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    return p.parse_args()


def download_zip(url: str, dest: Path) -> Path:
    """Download a file with a progress bar. Returns path to downloaded file."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "houseexpo.zip"

    if zip_path.exists():
        print(f"  ZIP already cached at {zip_path} — skipping download.")
        return zip_path

    print(f"Downloading HouseExpo from GitHub...")
    print(f"  URL: {url}")
    print(f"  This is ~30–60 MB, takes 1–2 minutes on a normal connection.")

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        desc="  Downloading",
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"  Downloaded to: {zip_path}")
    return zip_path

def extract_json_files(zip_path: Path, output_dir: Path, num_maps: int) -> list[Path]:
    """
    Extract JSON files from json.tar.gz which is inside the main zip.
    Returns list of extracted file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_cache = DOWNLOAD_CACHE / "json.tar.gz"

    # Step 1: pull json.tar.gz out of the zip
    if not tar_cache.exists():
        print("\nExtracting json.tar.gz from zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            tar_entry = "HouseExpo-master/HouseExpo/json.tar.gz"
            if tar_entry not in zf.namelist():
                print(f"ERROR: '{tar_entry}' not found in zip.")
                print(f"  Available: {zf.namelist()[:20]}")
                sys.exit(1)
            with zf.open(tar_entry) as src, open(tar_cache, "wb") as dst:
                shutil.copyfileobj(src, dst)
        print(f"  Saved to: {tar_cache}")
    else:
        print(f"\nUsing cached tar: {tar_cache}")

    # Step 2: extract JSON files from the tar.gz
    print(f"Extracting up to {num_maps} JSON files from tar.gz...")
    extracted = []
    with tarfile.open(tar_cache, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".json")]
        print(f"  Found {len(members)} JSON files in tar.")
        members = members[:num_maps]

        for member in tqdm(members, desc="  Extracting"):
            filename = Path(member.name).name
            dest_file = output_dir / filename
            if dest_file.exists():
                extracted.append(dest_file)
                continue
            member.name = filename  # flatten — no subdirs
            tf.extract(member, path=output_dir)
            extracted.append(dest_file)

    return extracted

def validate_json_files(json_files: list[Path]) -> tuple[int, int]:
    """
    Quick sanity check: load each JSON and verify it has wall data.
    Returns (valid_count, invalid_count).
    """
    print(f"\nValidating {len(json_files)} JSON files...")
    valid = 0
    invalid = 0
    invalid_files = []

    for f in tqdm(json_files, desc="  Validating"):
        try:
            with open(f) as fp:
                data = json.load(fp)

            # HouseExpo stores walls as 'verts' + 'edges' or as 'walls'
            has_data = (
                ("verts" in data and "edges" in data)
                or "walls" in data
                or len(data) > 0  # any non-empty JSON
            )
            if has_data:
                valid += 1
            else:
                invalid += 1
                invalid_files.append(f.name)

        except (json.JSONDecodeError, IOError):
            invalid += 1
            invalid_files.append(f.name)

    if invalid_files:
        print(f"  WARNING: {invalid} invalid files: {invalid_files[:5]}{'...' if len(invalid_files) > 5 else ''}")

    return valid, invalid


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HOUSEEXPO DOWNLOAD")
    print("=" * 60)
    print(f"  Target maps : {args.num_maps}")
    print(f"  Output dir  : {output_dir.resolve()}")

    # Check if we already have enough files
    existing = list(output_dir.glob("*.json"))
    if len(existing) >= args.num_maps and not args.force:
        print(f"\n✓ Already have {len(existing)} JSON files in {output_dir}")
        print(f"  Use --force to re-download.")
        return

    # Download
    zip_path = download_zip(HOUSEEXPO_ZIP_URL, DOWNLOAD_CACHE)

    # Extract
    json_files = extract_json_files(zip_path, output_dir, args.num_maps)

    # Validate
    valid, invalid = validate_json_files(json_files)

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Files extracted : {len(json_files)}")
    print(f"  Valid           : {valid}")
    print(f"  Invalid         : {invalid}")
    print(f"  Location        : {output_dir.resolve()}")

    if valid < args.num_maps * 0.9:
        print(f"\nWARNING: Only {valid}/{args.num_maps} files are valid.")
        print("  The dataset may be structured differently than expected.")
        print("  Check data/raw_maps/ and inspect a few JSON files manually.")
        sys.exit(1)

    print(f"\n✓ Ready for Phase 2: python scripts/02_preprocess_maps.py")


if __name__ == "__main__":
    main()