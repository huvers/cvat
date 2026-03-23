#!/usr/bin/env python3
"""
One-time S3 unzip script for surgical procedure datasets.

Extracts zip files stored on S3 and writes individual files back to S3,
preserving the LeRobot directory structure.

Usage (on an EC2 instance in the same region as your bucket):
    pip install boto3
    python3 s3-unzip.py --bucket YOUR_BUCKET --zip cholecystectomy.zip
    python3 s3-unzip.py --bucket YOUR_BUCKET --all

The script streams the zip from S3, extracts each file in memory,
and uploads it back to S3 under the procedure name prefix:
    cholecystectomy.zip → cholecystectomy/videos/chunk-000/.../episode_000000.mp4

For large zips (>500GB), this uses streaming to avoid loading the
entire zip into memory.
"""

import argparse
import io
import logging
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ALL_ZIPS = [
    "cholecystectomy.zip",
    "hysterectomy.zip",
    "inguinal_hernia.zip",
    "prostatectomy.zip",
]

# S3 multipart upload threshold
UPLOAD_THREADS = 8


def get_s3_client():
    return boto3.client("s3")


def extract_zip_from_s3(bucket: str, zip_key: str, target_prefix: str, dry_run: bool = False):
    """
    Download a zip from S3, extract all files, and upload them back.

    For zips up to ~50GB, this loads into memory.
    For larger zips, it uses a temporary file on disk.
    """
    s3 = get_s3_client()

    # Get zip file size
    head = s3.head_object(Bucket=bucket, Key=zip_key)
    zip_size_gb = head["ContentLength"] / (1024 ** 3)
    logger.info("Processing %s (%.1f GB)", zip_key, zip_size_gb)

    # Check if already extracted (look for at least one file under the prefix)
    existing = s3.list_objects_v2(Bucket=bucket, Prefix=f"{target_prefix}/", MaxKeys=5)
    if existing.get("KeyCount", 0) > 0:
        logger.info("  Target prefix %s/ already has %d files. Skipping (use --force to override).",
                     target_prefix, existing["KeyCount"])
        return 0

    if dry_run:
        logger.info("  [DRY RUN] Would extract to %s/", target_prefix)
        return 0

    # Download zip to temp file — use home dir to avoid tmpfs (RAM) limits
    tmp_dir = os.environ.get("UNZIP_TMPDIR", os.path.expanduser("~/tmp"))
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, os.path.basename(zip_key))
    logger.info("  Downloading %s to %s...", zip_key, tmp_path)
    start = time.time()
    s3.download_file(bucket, zip_key, tmp_path)
    elapsed = time.time() - start
    logger.info("  Downloaded in %.0fs (%.1f GB/s)", elapsed, zip_size_gb / max(elapsed, 1))

    # Extract and upload
    uploaded = 0
    skipped = 0
    errors = 0

    with zipfile.ZipFile(tmp_path, "r") as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        logger.info("  %d files in archive", len(members))

        def upload_member(name):
            nonlocal uploaded, skipped, errors
            # Build target key: procedure_name/rest/of/path
            target_key = f"{target_prefix}/{name}"

            try:
                data = zf.read(name)
                s3.put_object(Bucket=bucket, Key=target_key, Body=data)
                return True
            except Exception as e:
                logger.error("  Failed to upload %s: %s", name, e)
                return False

        # Upload with thread pool for parallelism
        with ThreadPoolExecutor(max_workers=UPLOAD_THREADS) as executor:
            futures = {executor.submit(upload_member, name): name for name in members}
            for i, future in enumerate(as_completed(futures)):
                name = futures[future]
                try:
                    if future.result():
                        uploaded += 1
                    else:
                        errors += 1
                except Exception:
                    errors += 1

                if (i + 1) % 100 == 0:
                    logger.info("  Progress: %d/%d files uploaded", i + 1, len(members))

    # Clean up temp file
    os.unlink(tmp_path)

    logger.info("  Done: %d uploaded, %d errors", uploaded, errors)
    return uploaded


def main():
    parser = argparse.ArgumentParser(description="Extract S3 zip files for CVAT surgery platform")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--zip", help="Specific zip file to extract (e.g., cholecystectomy.zip)")
    parser.add_argument("--all", action="store_true", help="Extract all procedure zips")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without extracting")
    parser.add_argument("--force", action="store_true", help="Extract even if target prefix already has files")
    parser.add_argument("--prefix", default="", help="S3 prefix where zips are located (default: root)")
    args = parser.parse_args()

    if not args.zip and not args.all:
        parser.error("Specify --zip FILE or --all")

    zips_to_process = ALL_ZIPS if args.all else [args.zip]

    total_files = 0
    for zip_name in zips_to_process:
        zip_key = f"{args.prefix}{zip_name}" if args.prefix else zip_name
        # Target prefix = zip name without .zip extension
        target_prefix = zip_name.replace(".zip", "")

        try:
            count = extract_zip_from_s3(
                args.bucket, zip_key, target_prefix, dry_run=args.dry_run,
            )
            total_files += count
        except Exception:
            logger.exception("Failed to process %s", zip_name)

    logger.info("=" * 50)
    logger.info("Total: %d files extracted", total_files)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Register this S3 bucket in CVAT (Cloud Storages)")
    logger.info("  2. Go to /bulk-ingest and enter each procedure prefix:")
    for zip_name in zips_to_process:
        prefix = zip_name.replace(".zip", "")
        logger.info("     - %s", prefix)
    logger.info("  3. Click 'Start Ingestion' for each")


if __name__ == "__main__":
    main()
