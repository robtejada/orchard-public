#!/usr/bin/env python
"""
Download and install EOS data from Zenodo for ORCHARD.

This script downloads the equation-of-state data files (~30 GB compressed)
from Zenodo and extracts them into the eos/ submodule directory. It is the
recommended alternative to Git LFS for obtaining the EOS data.

Usage:
    python setup_eos.py              # download if data is missing
    python setup_eos.py --force      # re-download even if data exists
    python setup_eos.py --record-id 99999999  # override Zenodo record ID

Only the Python standard library is required (no pip install needed).
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EOS_DIR = os.path.join(SCRIPT_DIR, "eos")
ZENODO_CONFIG = os.path.join(SCRIPT_DIR, "eos_zenodo.json")

# Sentinel: the v1.0 CD21+AQUA forward table (~981 MB when real).
# LFS pointer stubs are ~130 bytes.
SENTINEL_FILE = os.path.join(EOS_DIR, "cd", "cd_aqua_pt.npz")
SENTINEL_MIN_BYTES = 1_000_000  # 1 MB


def eos_data_present():
    """Return True if EOS data files appear to be real (not LFS stubs)."""
    if not os.path.isfile(SENTINEL_FILE):
        return False
    return os.path.getsize(SENTINEL_FILE) > SENTINEL_MIN_BYTES


def eos_python_present():
    """Return True if the EOS Python module is already populated.

    Used to distinguish a submodule-clone install (where git provides the
    authoritative Python files and Zenodo only fills in data tables) from
    a Zenodo-tarball-only install (where eos/ ships empty and Zenodo must
    provide both data and Python code).
    """
    return os.path.isfile(os.path.join(EOS_DIR, "eos_class.py"))


def load_zenodo_config(record_id_override=None):
    """Load Zenodo record info from eos_zenodo.json."""
    if not os.path.isfile(ZENODO_CONFIG):
        print(f"Error: {ZENODO_CONFIG} not found.", file=sys.stderr)
        print("Make sure you are running this from the ORCHARD root directory.",
              file=sys.stderr)
        sys.exit(1)

    with open(ZENODO_CONFIG) as f:
        config = json.load(f)

    if record_id_override:
        config["zenodo_record_id"] = record_id_override

    return config


def resolve_zenodo_record(record_id):
    """Resolve a Zenodo record id to the LATEST published version's id.

    eos_zenodo.json ships the permanent concept-record id, which always
    tracks the newest published EOS release. Zenodo's landing pages redirect
    concept ids to the latest version, but the /files/ download paths do
    not, so we resolve through the API first. If the lookup fails (offline,
    API change), fall back to the configured id unchanged.
    """
    import json as _json
    import urllib.request

    api_url = f"https://zenodo.org/api/records/{record_id}"
    try:
        with urllib.request.urlopen(api_url, timeout=30) as resp:
            resolved = str(_json.load(resp).get("id", record_id))
        if resolved != str(record_id):
            print(f"Zenodo record {record_id} resolves to latest version "
                  f"{resolved}.")
        return resolved
    except Exception as exc:  # pragma: no cover - network-dependent
        print(f"Warning: could not resolve Zenodo record {record_id} "
              f"({exc}); using it directly.", file=sys.stderr)
        return record_id


def download_with_progress(url, dest_path, size_gb=None):
    """Download a file with a console progress bar."""
    size_hint = f" (~{size_gb} GB)" if size_gb else ""
    print(f"Downloading EOS data{size_hint}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")
    print()

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            downloaded_gb = downloaded / 1e9
            total_gb = total_size / 1e9
            bar_len = 40
            filled = int(bar_len * pct / 100)
            bar = "=" * filled + "-" * (bar_len - filled)
            sys.stdout.write(
                f"\r  [{bar}] {pct:5.1f}%  ({downloaded_gb:.2f} / {total_gb:.2f} GB)"
            )
        else:
            downloaded_gb = downloaded / 1e9
            sys.stdout.write(f"\r  Downloaded {downloaded_gb:.2f} GB...")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
        print()  # newline after progress bar
    except urllib.error.HTTPError as e:
        print(f"\n\nHTTP Error {e.code}: {e.reason}", file=sys.stderr)
        if e.code == 404:
            print("The Zenodo record was not found. The record ID may be outdated.",
                  file=sys.stderr)
            print("Check eos_zenodo.json or try --record-id with the correct ID.",
                  file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"\n\nNetwork error: {e.reason}", file=sys.stderr)
        print("Check your internet connection and try again.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.", file=sys.stderr)
        if os.path.isfile(dest_path):
            os.remove(dest_path)
            print("Partial download removed.", file=sys.stderr)
        sys.exit(1)


def check_disk_space(path, required_gb=80):
    """Warn if available disk space is below required_gb."""
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / 1e9
        if free_gb < required_gb:
            print(f"Warning: Only {free_gb:.1f} GB free on disk.", file=sys.stderr)
            print(f"  The EOS data requires ~{required_gb} GB when fully extracted.",
                  file=sys.stderr)
            resp = input("  Continue anyway? [y/N]: ").strip().lower()
            if resp != "y":
                print("Aborted.", file=sys.stderr)
                sys.exit(1)
    except OSError:
        pass  # disk_usage may not work on all platforms


def extract_data_files(zip_path, target_eos_dir, include_python=False):
    """Extract files from the Zenodo zip into eos/.

    By default, Python module files are skipped because the git submodule
    provides the authoritative versions. When include_python=True, all files
    are extracted — used for distribution-only installs (e.g., the ORCHARD
    Zenodo release tarball, where eos/ ships empty without a populated submodule).
    """
    if include_python:
        print("Extracting EOS data files (including Python modules)...")
    else:
        print("Extracting EOS data files...")

    skip_extensions = set() if include_python else {".py", ".pyc"}

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the top-level directory in the zip (e.g., "eos/" or "eos-eos_orchard/")
        top_dirs = set()
        for name in zf.namelist():
            parts = name.split("/")
            if len(parts) > 1:
                top_dirs.add(parts[0])

        # Determine the prefix to strip
        if len(top_dirs) == 1:
            zip_prefix = top_dirs.pop() + "/"
        else:
            zip_prefix = ""

        members = zf.namelist()
        total = len(members)
        extracted = 0
        skipped_py = 0

        for i, member in enumerate(members):
            # Skip directories themselves (we create them as needed)
            if member.endswith("/"):
                continue

            # Strip the zip's top-level prefix to get the relative path within eos/
            if zip_prefix and member.startswith(zip_prefix):
                rel_path = member[len(zip_prefix):]
            else:
                rel_path = member

            if not rel_path:
                continue

            # Skip Python files — git has the authoritative versions
            _, ext = os.path.splitext(rel_path)
            if ext.lower() in skip_extensions:
                skipped_py += 1
                continue

            dest = os.path.join(target_eos_dir, rel_path)
            dest_dir = os.path.dirname(dest)
            os.makedirs(dest_dir, exist_ok=True)

            # Extract the file
            with zf.open(member) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1

            # Progress indicator every 50 files
            if extracted % 50 == 0:
                sys.stdout.write(f"\r  Extracted {extracted} files...")
                sys.stdout.flush()

        print(f"\r  Extracted {extracted} data files (skipped {skipped_py} Python files).")


def main():
    parser = argparse.ArgumentParser(
        description="Download EOS data from Zenodo for ORCHARD."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if EOS data already exists."
    )
    parser.add_argument(
        "--record-id", type=str, default=None,
        help="Override the Zenodo record ID from eos_zenodo.json."
    )
    args = parser.parse_args()

    # Check if data already present
    if not args.force and eos_data_present():
        print("EOS data files already present. Nothing to do.")
        print(f"  (sentinel: {SENTINEL_FILE})")
        print("  Use --force to re-download.")
        return

    if not eos_data_present():
        if os.path.isfile(SENTINEL_FILE):
            print("EOS data files appear to be Git LFS pointer stubs.")
        else:
            print("EOS data files not found.")
        print("Will download from Zenodo.\n")

    # Load config
    config = load_zenodo_config(record_id_override=args.record_id)
    record_id = config["zenodo_record_id"]
    filename = config["filename"]
    size_gb = config.get("size_gb")

    record_id = resolve_zenodo_record(record_id)
    url = f"https://zenodo.org/records/{record_id}/files/{filename}?download=1"

    # Check disk space
    check_disk_space(SCRIPT_DIR, required_gb=80)

    # Download to a temp file in the same filesystem (for atomic-ish operations)
    zip_path = os.path.join(SCRIPT_DIR, f".eos_download_{filename}")
    try:
        download_with_progress(url, zip_path, size_gb=size_gb)

        # Verify it is a valid zip
        if not zipfile.is_zipfile(zip_path):
            print("Error: Downloaded file is not a valid zip archive.", file=sys.stderr)
            print("The download may have been corrupted. Try again.", file=sys.stderr)
            sys.exit(1)

        # If eos/ already has Python module code, it came from a git submodule
        # clone and is the authoritative source — only extract data files.
        # Otherwise (Zenodo-tarball install), extract everything including .py.
        include_python = not eos_python_present()
        extract_data_files(zip_path, EOS_DIR, include_python=include_python)

    finally:
        # Clean up the zip
        if os.path.isfile(zip_path):
            print("Cleaning up downloaded archive...")
            os.remove(zip_path)

    # Verify
    if eos_data_present():
        print("\nEOS data installed successfully!")
        print(f"  Location: {EOS_DIR}")
        print(f"  Zenodo DOI: {config.get('zenodo_doi', 'N/A')}")
    else:
        print("\nWarning: EOS data may not have been fully extracted.",
              file=sys.stderr)
        print(f"  Expected sentinel: {SENTINEL_FILE}", file=sys.stderr)
        print("  You may need to re-run this script or download manually.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
