from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io
import time

import requests
from PIL import Image


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FASHIONIQ_ROOT = PROJECT_ROOT / "data" / "raw" / "fashioniq"
BROKEN_DIR = FASHIONIQ_ROOT / "broken_links"

CATEGORIES = {
    "dress": FASHIONIQ_ROOT / "asin2url.dress.txt",
    "shirt": FASHIONIQ_ROOT / "asin2url.shirt.txt",
    "toptee": FASHIONIQ_ROOT / "asin2url.toptee.txt",
}

MAX_WORKERS = 8
TIMEOUT = 20

# Change this to None for the complete download.
# For the first test, keep it at 100.
LIMIT_PER_CATEGORY = None


# ============================================================
# Session
# ============================================================

session = requests.Session()

session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
        )
    }
)


# ============================================================
# Helpers
# ============================================================

def parse_url_file(url_file):
    """Read IMAGE_ID + URL pairs."""

    entries = []

    with url_file.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                print(
                    f"[WARNING] Invalid line {line_number}: {line}"
                )
                continue

            image_id = parts[0]
            url = parts[1]

            entries.append((image_id, url))

    return entries


def is_valid_image(data):
    """Check whether downloaded bytes are actually a valid image."""

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()

        return True

    except Exception:
        return False


def save_image(data, output_path):
    """Save image bytes after validating them."""

    if not is_valid_image(data):
        return False

    output_path.write_bytes(data)

    return True


def download_from_url(image_id, url, output_path):
    """Try downloading an image from its URL."""

    urls_to_try = [url]

    # Some URLs in the metadata are HTTP.
    # Try HTTPS as a fallback.
    if url.startswith("http://"):
        urls_to_try.append("https://" + url[len("http://"):])

    for candidate_url in urls_to_try:

        try:

            response = session.get(
                candidate_url,
                timeout=TIMEOUT,
            )

            if response.status_code != 200:
                continue

            if save_image(response.content, output_path):
                return "downloaded"

        except requests.RequestException:
            continue

    return None


def copy_broken_link_image(image_id, output_path):
    """Use repository-provided replacement image if available."""

    fallback = BROKEN_DIR / f"{image_id}.jpg"

    if not fallback.exists():
        return False

    try:
        data = fallback.read_bytes()

        if save_image(data, output_path):
            return True

    except Exception:
        pass

    return False


def process_image(category, image_id, url):
    """Download one FashionIQ image."""

    output_dir = FASHIONIQ_ROOT / category
    output_path = output_dir / f"{image_id}.jpg"

    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Already downloaded
    # --------------------------------------------------------

    if output_path.exists() and output_path.stat().st_size > 0:

        return {
            "category": category,
            "image_id": image_id,
            "status": "already_exists",
            "url": url,
        }

    # --------------------------------------------------------
    # Try original URL
    # --------------------------------------------------------

    result = download_from_url(
        image_id,
        url,
        output_path,
    )

    if result == "downloaded":

        return {
            "category": category,
            "image_id": image_id,
            "status": "downloaded",
            "url": url,
        }

    # --------------------------------------------------------
    # Try broken_links fallback
    # --------------------------------------------------------

    if copy_broken_link_image(
        image_id,
        output_path,
    ):

        return {
            "category": category,
            "image_id": image_id,
            "status": "fallback",
            "url": url,
        }

    # --------------------------------------------------------
    # Failed
    # --------------------------------------------------------

    return {
        "category": category,
        "image_id": image_id,
        "status": "failed",
        "url": url,
    }


# ============================================================
# Download category
# ============================================================

def download_category(category, url_file):

    print()
    print("=" * 70)
    print(f"Downloading category: {category}")
    print("=" * 70)

    entries = parse_url_file(url_file)

    print(f"URL entries found: {len(entries):,}")

    if LIMIT_PER_CATEGORY is not None:
        entries = entries[:LIMIT_PER_CATEGORY]

        print(
            f"TEST MODE: downloading first "
            f"{len(entries):,} images"
        )

    results = []

    start_time = time.time()

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = [
            executor.submit(
                process_image,
                category,
                image_id,
                url,
            )
            for image_id, url in entries
        ]

        for completed, future in enumerate(
            as_completed(futures),
            start=1,
        ):

            result = future.result()

            results.append(result)

            if (
                completed % 10 == 0
                or completed == len(futures)
            ):

                print(
                    f"[{category}] "
                    f"{completed:,}/{len(futures):,}"
                )

    elapsed = time.time() - start_time

    return results, elapsed


# ============================================================
# Main
# ============================================================

def main():

    all_results = []

    for category, url_file in CATEGORIES.items():

        results, elapsed = download_category(
            category,
            url_file,
        )

        all_results.extend(results)

        downloaded = sum(
            r["status"] == "downloaded"
            for r in results
        )

        fallback = sum(
            r["status"] == "fallback"
            for r in results
        )

        existing = sum(
            r["status"] == "already_exists"
            for r in results
        )

        failed = sum(
            r["status"] == "failed"
            for r in results
        )

        print()
        print(f"{category} summary:")
        print(f"  Downloaded : {downloaded}")
        print(f"  Fallback   : {fallback}")
        print(f"  Existing   : {existing}")
        print(f"  Failed     : {failed}")
        print(f"  Time       : {elapsed:.1f} seconds")

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_path = (
        FASHIONIQ_ROOT / "download_report.csv"
    )

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "image_id",
                "status",
                "url",
            ],
        )

        writer.writeheader()
        writer.writerows(all_results)

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()