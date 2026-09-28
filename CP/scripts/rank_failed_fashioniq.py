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
# Count query impact for every failed image
# ---------------------------------------------------------

for category in CATEGORIES:

    print()
    print("=" * 80)
    print(category.upper())
    print("=" * 80)

    impact = defaultdict(
        lambda: {
            "train_reference": 0,
            "train_target": 0,
            "val_reference": 0,
            "val_target": 0,
            "test_reference": 0,
        }
    )

    for split in SPLITS:

        caption_file = (
            DATA
            / "captions"
            / f"cap.{category}.{split}.json"
        )

        with open(caption_file, "r", encoding="utf-8") as f:
            captions = json.load(f)

        for item in captions:

            candidate = item.get("candidate")
            target = item.get("target")

            if candidate in failed[category]:

                key = f"{split}_reference"

                impact[candidate][key] += 1

            if target in failed[category]:

                key = f"{split}_target"

                impact[target][key] += 1

    # -----------------------------------------------------
    # Sort by total query impact
    # -----------------------------------------------------

    ranked = []

    for image_id, counts in impact.items():

        total = sum(counts.values())

        ranked.append(
            (
                total,
                image_id,
                counts
            )
        )

    ranked.sort(reverse=True)

    print()
    print(
        f"{'Image ID':15s} "
        f"{'Total':>6s} "
        f"{'TR':>5s} "
        f"{'TT':>5s} "
        f"{'VR':>5s} "
        f"{'VT':>5s} "
        f"{'TeR':>5s}"
    )

    print("-" * 80)

    for total, image_id, counts in ranked[:30]:

        print(
            f"{image_id:15s} "
            f"{total:6d} "
            f"{counts['train_reference']:5d} "
            f"{counts['train_target']:5d} "
            f"{counts['val_reference']:5d} "
            f"{counts['val_target']:5d} "
            f"{counts['test_reference']:5d}"
        )

    # -----------------------------------------------------
    # Save complete ranking
    # -----------------------------------------------------

    output_file = (
        DATA / f"failed_image_impact_{category}.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "image_id",
                "total_impact",
                "train_reference",
                "train_target",
                "val_reference",
                "val_target",
                "test_reference",
            ]
        )

        for total, image_id, counts in ranked:

            writer.writerow(
                [
                    image_id,
                    total,
                    counts["train_reference"],
                    counts["train_target"],
                    counts["val_reference"],
                    counts["val_target"],
                    counts["test_reference"],
                ]
            )


print()
print("=" * 80)
print("RANKING COMPLETE")
print("=" * 80)