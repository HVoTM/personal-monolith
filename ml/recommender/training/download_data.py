"""Download and extract the MovieLens ml-latest-small dataset.

Usage (from apps/recommender):
    python training/download_data.py

Extracts to data/raw/ml-latest-small/. Re-running is a no-op unless --force is passed.
"""

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

try:
    # Verify TLS with the OS certificate store. On Windows, Python's default store only has
    # roots already installed locally and fails on servers whose root Windows fetches on demand.
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

DATASET = "ml-latest-small"
URL = f"https://files.grouplens.org/datasets/movielens/{DATASET}.zip"
MD5_URL = URL + ".md5"

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
EXPECTED_FILES = ["ratings.csv", "movies.csv", "tags.csv", "links.csv"]


def md5sum(path: Path) -> str:
    """
    32-character hex fingerprint of its bytes
    """
    # Start empty hasher
    h = hashlib.md5()
    # Binary mode opening
    with path.open("rb") as f:
        # Keep calling f.read in chunk for fixed-size memory, 1MiB at a time
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_expected_md5() -> str | None:
    try:
        with urllib.request.urlopen(MD5_URL, timeout=30) as resp:
            # File looks like "<hash>  ml-latest-small.zip" or "MD5 (...) = <hash>".
            text = resp.read().decode().strip()
    except OSError as e:
        print(f"warning: could not fetch checksum ({e}); skipping verification")
        return None
    for token in text.replace("=", " ").split():
        if len(token) == 32 and all(c in "0123456789abcdef" for c in token.lower()):
            return token.lower()
    print("warning: checksum file had unexpected format; skipping verification")
    return None


def safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract, refusing any member that would land outside dest."""

    # Resolve dest to an absolute path
    dest = dest.resolve()
    ## Build where the file would land
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest):
                raise RuntimeError(f"unsafe path in archive: {member.filename}")
        # entry passes -> extract all
        zf.extractall(dest)

"""
Download MovieLens file
"""
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    out_dir = RAW_DIR / DATASET
    if out_dir.exists() and not args.force:
        print(f"{out_dir} already exists; pass --force to re-download")
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / f"{DATASET}.zip"

    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, zip_path)

    expected = fetch_expected_md5()
    if expected is not None:
        actual = md5sum(zip_path)
        if actual != expected:
            zip_path.unlink()
            print(f"checksum mismatch: expected {expected}, got {actual}")
            return 1
        print("checksum ok")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    safe_extract(zip_path, RAW_DIR)
    zip_path.unlink()

    missing = [f for f in EXPECTED_FILES if not (out_dir / f).exists()]
    if missing:
        print(f"extracted, but missing expected files: {missing}")
        return 1

    print(f"done: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
