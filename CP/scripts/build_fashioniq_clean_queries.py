from pathlib import Path
import json
import csv
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data" / "raw" / "fashioniq"
OUT = ROOT / "data" / "processed" / "fashioniq"

CATEGORIES = ["dress", "shirt", "toptee"]
SPLITS = ["train", "val", "test"]

OUT.mkdir(parents=True, exist_ok=True)


# =========================================================
# Load download status
# =========================================================

status = defaultdict(dict)

report_path = RAW / "download_report.csv"

with open(report_path, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        category = row["category"]
        image_id = row["image_id"]

        status[category][image_id] = row["status"]


# =========================================================
# Check whether an image is actually usable
# =========================================================

def image_available(category, image_id):

    image_path = (
        RAW
        / category
        / f"{image_id}.jpg"
    )

    return image_path.exists()


# =========================================================
# Process each category/split
# =========================================================

all_excluded = []
statistics = {}

for category in CATEGORIES:

    statistics[category] = {}

    for split in SPLITS:

        caption_file = (
            RAW
            / "captions"
            / f"cap.{category}.{split}.json"
        )

        with open(
            caption_file,
            "r",
            encoding="utf-8"
        ) as f:

            records = json.load(f)

        clean_queries = []
        excluded_queries = []

        for idx, item in enumerate(records):

            candidate = item.get("candidate")
            target = item.get("target")

            candidate_available = (
                candidate is not None
                and image_available(
                    category,
                    candidate
                )
            )

            # Test has no target field.
            if split == "test":

                target_available = True

            else:

                target_available = (
                    target is not None
                    and image_available(
                        category,
                        target
                    )
                )

            # -------------------------------------------------
            # Clean query
            # -------------------------------------------------

            if candidate_available and target_available:

                clean_queries.append(
                    {
                        "category": category,
                        "split": split,
                        "query_id": f"{category}_{split}_{idx:05d}",
                        "candidate": candidate,
                        "target": target or "",
                        "caption_1": (
                            item.get("captions", [""])[0]
                        ),
                        "caption_2": (
                            item.get("captions", ["", ""])[1]
                            if len(item.get("captions", [])) > 1
                            else ""
                        ),
                    }
                )

            # -------------------------------------------------
            # Excluded query
            # -------------------------------------------------

            else:

                reasons = []

                if not candidate_available:

                    reasons.append(
                        "missing_candidate"
                    )

                if split != "test" and not target_available:

                    reasons.append(
                        "missing_target"
                    )

                excluded = {
                    "category": category,
                    "split": split,
                    "query_id": f"{category}_{split}_{idx:05d}",
                    "candidate": candidate or "",
                    "target": target or "",
                    "reason": ";".join(reasons),
                    "caption_1": (
                        item.get("captions", [""])[0]
                    ),
                    "caption_2": (
                        item.get("captions", ["", ""])[1]
                        if len(item.get("captions", [])) > 1
                        else ""
                    ),
                }

                excluded_queries.append(excluded)
                all_excluded.append(excluded)

        # -----------------------------------------------------
        # Save clean split
        # -----------------------------------------------------

        clean_file = (
            OUT / f"{split}_queries_{category}.csv"
        )

        with open(
            clean_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            fieldnames = [
                "category",
                "split",
                "query_id",
                "candidate",
                "target",
                "caption_1",
                "caption_2",
            ]

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(clean_queries)

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        statistics[category][split] = {
            "total_queries": len(records),
            "usable_queries": len(clean_queries),
            "excluded_queries": len(excluded_queries),
        }

        print(
            f"{category:7s} | "
            f"{split:5s} | "
            f"total = {len(records):5d} | "
            f"usable = {len(clean_queries):5d} | "
            f"excluded = {len(excluded_queries):5d}"
        )


# =========================================================
# Save combined exclusion list
# =========================================================

excluded_file = OUT / "excluded_queries.csv"

with open(
    excluded_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "category",
        "split",
        "query_id",
        "candidate",
        "target",
        "reason",
        "caption_1",
        "caption_2",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_excluded)


# =========================================================
# Save statistics
# =========================================================

stats_file = OUT / "dataset_statistics.json"

with open(
    stats_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        statistics,
        f,
        indent=4
    )


print()
print("=" * 70)
print("FASHIONIQ CLEAN DATASET CREATED")
print("=" * 70)

print(
    f"Excluded queries: "
    f"{len(all_excluded)}"
)

print(
    f"Output directory: "
    f"{OUT}"
)

print()
print("Original raw annotations were NOT modified.")