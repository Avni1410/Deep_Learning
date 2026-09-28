from pathlib import Path
import json
import csv
from collections import defaultdict


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "fashioniq"

CATEGORIES = ["dress", "shirt", "toptee"]
SPLITS = ["train", "val", "test"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# Load download report
# ------------------------------------------------------------

report_path = DATA / "download_report.csv"

failed_ids = defaultdict(set)
fallback_ids = defaultdict(set)

with open(report_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        category = row["category"]
        image_id = row["image_id"]
        status = row["status"]

        if status == "failed":
            failed_ids[category].add(image_id)

        elif status == "fallback":
            fallback_ids[category].add(image_id)


print("=" * 70)
print("DOWNLOAD REPORT")
print("=" * 70)

for category in CATEGORIES:
    print(
        f"{category:8s} | "
        f"failed = {len(failed_ids[category]):5d} | "
        f"fallback = {len(fallback_ids[category]):5d}"
    )


# ------------------------------------------------------------
# Check split files
# ------------------------------------------------------------

print()
print("=" * 70)
print("FAILED IMAGES REFERENCED BY SPLITS")
print("=" * 70)

for category in CATEGORIES:

    for split in SPLITS:

        split_path = (
            DATA
            / "image_splits"
            / f"split.{category}.{split}.json"
        )

        ids = set(load_json(split_path))

        failed_in_split = ids & failed_ids[category]

        print(
            f"{category:8s} {split:5s} | "
            f"images = {len(ids):5d} | "
            f"failed = {len(failed_in_split):5d}"
        )

        if failed_in_split:
            output = (
                DATA
                / f"failed_{category}_{split}.txt"
            )

            with open(output, "w", encoding="utf-8") as f:
                for image_id in sorted(failed_in_split):
                    f.write(image_id + "\n")

            print(f"    Saved: {output}")


# ------------------------------------------------------------
# Check caption references
# ------------------------------------------------------------

print()
print("=" * 70)
print("FAILED IMAGES REFERENCED BY CAPTIONS")
print("=" * 70)

for category in CATEGORIES:

    caption_failed = set()

    for split in SPLITS:

        caption_path = (
            DATA
            / "captions"
            / f"cap.{category}.{split}.json"
        )

        captions = load_json(caption_path)

        for item in captions:

            # FashionIQ caption entries contain
            # candidate image IDs.
            for key in ["candidate", "target"]:

                if key in item:

                    value = item[key]

                    if isinstance(value, list):
                        caption_failed.update(
                            set(value) & failed_ids[category]
                        )

                    elif isinstance(value, str):
                        if value in failed_ids[category]:
                            caption_failed.add(value)

    print(
        f"{category:8s} | "
        f"failed IDs referenced by captions = "
        f"{len(caption_failed)}"
    )


print()
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)