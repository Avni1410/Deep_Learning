from pathlib import Path
import json
import csv
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "fashioniq"

CATEGORIES = ["dress", "shirt", "toptee"]
SPLITS = ["train", "val", "test"]

# ---------------------------------------------------------
# Load failed downloads
# ---------------------------------------------------------

failed = defaultdict(set)

report_path = DATA / "download_report.csv"

with open(report_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["status"] == "failed":
            failed[row["category"]].add(row["image_id"])


# ---------------------------------------------------------
# Analyze FashionIQ caption files
# ---------------------------------------------------------

for category in CATEGORIES:

    print()
    print("=" * 70)
    print(category.upper())
    print("=" * 70)

    failed_reference = set()
    failed_target = set()
    affected_queries = []

    total_queries = 0

    for split in SPLITS:

        caption_file = (
            DATA
            / "captions"
            / f"cap.{category}.{split}.json"
        )

        with open(caption_file, "r", encoding="utf-8") as f:
            captions = json.load(f)

        split_total = 0
        split_affected = 0
        split_failed_reference = 0
        split_failed_target = 0

        for item in captions:

            # Every caption record should have candidate.
            candidate = item.get("candidate")

            if candidate is None:
                continue

            total_queries += 1
            split_total += 1

            # Candidate = reference/source image
            candidate_failed = candidate in failed[category]

            # Test split does not contain target.
            target = item.get("target")

            target_failed = (
                target is not None
                and target in failed[category]
            )

            if candidate_failed:
                failed_reference.add(candidate)
                split_failed_reference += 1

            if target_failed:
                failed_target.add(target)
                split_failed_target += 1

            if candidate_failed or target_failed:

                affected_queries.append(
                    {
                        "category": category,
                        "split": split,
                        "candidate": candidate,
                        "target": target if target else "",
                        "candidate_failed": candidate_failed,
                        "target_failed": target_failed,
                        "captions": " || ".join(
                            item.get("captions", [])
                        ),
                    }
                )

                split_affected += 1

        print(
            f"{split:5s} | "
            f"queries = {split_total:5d} | "
            f"affected = {split_affected:5d} | "
            f"failed reference = {split_failed_reference:5d} | "
            f"failed target = {split_failed_target:5d}"
        )

    print()
    print(f"Total queries:              {total_queries}")
    print(f"Unique failed references:   {len(failed_reference)}")
    print(f"Unique failed targets:      {len(failed_target)}")
    print(f"Affected queries:           {len(affected_queries)}")

    # -----------------------------------------------------
    # Save affected queries
    # -----------------------------------------------------

    output_file = (
        DATA / f"affected_queries_{category}.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "split",
                "candidate",
                "target",
                "candidate_failed",
                "target_failed",
                "captions",
            ],
        )

        writer.writeheader()
        writer.writerows(affected_queries)


print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)