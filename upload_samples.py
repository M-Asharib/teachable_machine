# -*- coding: utf-8 -*-
"""
upload_samples.py
-----------------
Extracts a random sample of images from a zip file and uploads them
directly to the Teachable Machine FastAPI backend via POST /upload-sample.

Usage:
    python upload_samples.py --zip <path_to_zip> --class_name <label> [--count 40] [--backend http://localhost:8000]

Examples:
    python upload_samples.py --zip Models/human_faces.zip --class_name Human
    python upload_samples.py --zip Models/mobile_phones.zip --class_name Mobile --count 50
"""

import argparse
import io
import random
import sys
import zipfile

import requests


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_image_entries(zip_path: str) -> list[str]:
    """Return all image file paths inside the zip."""
    with zipfile.ZipFile(zip_path, "r") as z:
        entries = [
            name for name in z.namelist()
            if not name.endswith("/")  # skip directories
            and any(name.lower().endswith(ext) for ext in VALID_EXTENSIONS)
        ]
    return entries


def upload_batch(zip_path: str, entries: list[str], class_name: str, backend_url: str) -> dict:
    """Open each entry from the zip in-memory and POST to /upload-sample."""
    files_payload = []

    with zipfile.ZipFile(zip_path, "r") as z:
        for entry in entries:
            file_bytes = z.read(entry)
            filename = entry.split("/")[-1]  # strip folder prefix
            mime = "image/jpeg" if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg") else "image/png"
            files_payload.append(("files", (filename, io.BytesIO(file_bytes), mime)))

    response = requests.post(
        f"{backend_url}/upload-sample",
        data={"class_name": class_name},
        files=files_payload,
        timeout=60,
    )
    return response


def main():
    parser = argparse.ArgumentParser(description="Upload random image samples to Teachable Machine backend.")
    parser.add_argument("--zip",        required=True,                        help="Path to the .zip file containing images")
    parser.add_argument("--class_name", required=True,                        help="Class label to upload images under (e.g. Human, Mobile)")
    parser.add_argument("--count",      type=int, default=40,                 help="Number of random images to upload (default: 40)")
    parser.add_argument("--backend",    default="http://localhost:8000",       help="FastAPI backend URL (default: http://localhost:8000)")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Teachable Machine — Bulk Sample Uploader")
    print(f"{'='*55}")
    print(f"  ZIP file   : {args.zip}")
    print(f"  Class name : {args.class_name}")
    print(f"  Count      : {args.count}")
    print(f"  Backend    : {args.backend}")
    print(f"{'='*55}\n")

    # Step 1: Check backend connection
    print("[..] Checking backend connection...")
    try:
        res = requests.get(args.backend, timeout=3)
        res.raise_for_status()
        print(f"[OK] Backend connected: {res.json().get('status')}\n")
    except Exception as e:
        print(f"[!!] Cannot reach backend at {args.backend}")
        print(f"   Make sure FastAPI is running: python -m uvicorn backend.main:app --port 8000")
        print(f"   Error: {e}")
        sys.exit(1)

    # Step 2: Read image list from zip
    print(f"[..] Scanning zip file...")
    try:
        all_images = get_image_entries(args.zip)
    except FileNotFoundError:
        print(f"[!!] Zip file not found: {args.zip}")
        sys.exit(1)
    except zipfile.BadZipFile:
        print(f"[!!] Invalid or corrupted zip file: {args.zip}")
        sys.exit(1)

    total_available = len(all_images)
    print(f"   Found {total_available} image files in the zip.\n")

    if total_available == 0:
        print("[!!] No valid image files found inside the zip.")
        sys.exit(1)

    # Step 3: Pick random sample
    count = min(args.count, total_available)
    selected = random.sample(all_images, count)
    print(f"[>>] Randomly selected {count} images for upload.\n")

    # Step 4: Upload in batches of 10 to avoid timeout
    BATCH_SIZE = 10
    total_uploaded = 0

    for batch_start in range(0, count, BATCH_SIZE):
        batch = selected[batch_start: batch_start + BATCH_SIZE]
        batch_num = (batch_start // BATCH_SIZE) + 1
        total_batches = (count + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"[UP] Uploading batch {batch_num}/{total_batches} ({len(batch)} images)...", end=" ", flush=True)
        try:
            res = upload_batch(args.zip, batch, args.class_name, args.backend)
            if res.status_code == 200:
                saved = res.json().get("saved_count", len(batch))
                total_uploaded += saved
                print(f"[OK] {saved} saved")
            else:
                detail = res.json().get("detail", "Unknown error")
                print(f"[!!] Failed -- {detail}")
        except Exception as e:
            print(f"[!!] Error -- {e}")

    # Step 5: Summary
    print(f"\n{'='*55}")
    print(f"  [OK] Upload complete!")
    print(f"  Class '{args.class_name}' now has {total_uploaded} new samples.")
    print(f"  Open http://localhost:8501 and click Train Custom Model")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
