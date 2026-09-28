import json
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path(
    "data/raw/deepfashion2/matching_pairs.json"
)

OUTPUT_DIR = Path(
    "data/processed/deepfashion2"
)

OUTPUT_FILE = OUTPUT_DIR / "domain_alignment_pairs.csv"

TARGET_PAIRS = 2000
SEED = 42


# ============================================================
# Load
# ============================================================

print("=" * 80)
print("DeepFashion2 Domain-Alignment Subset Builder")
print("=" * 80)

print(f"\nInput: {INPUT_FILE}")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

print(f"Total metadata records: {len(records):,}")


# ============================================================
# Basic filtering
# ============================================================

valid = []

for record in records:

    # We only use positive style annotations.
    if record.get("style", 0) <= 0:
        continue

    shop = record.get("shop_image", {})
    user = record.get("user_image", {})

    shop_name = shop.get("name")
    user_name = user.get("name")

    shop_path = shop.get("path")
    user_path = user.get("path")

    if not all([
        shop_name,
        user_name,
        shop_path,
        user_path
    ]):
        continue

    valid.append({
        "pair_id": record["pair_id"],
        "style": record["style"],
        "category_name": record["category_name"],
        "category_id": record["category_id"],

        "shop_image": shop_name,
        "shop_path": shop_path,
        "shop_bbox": shop.get("bbox"),

        "user_image": user_name,
        "user_path": user_path,
        "user_bbox": user.get("bbox"),
    })


print(f"Valid records after filtering: {len(valid):,}")


# ============================================================
# Remove exact duplicate image pairs
# ============================================================

unique_pairs = {}

for record in valid:

    key = (
        record["user_image"],
        record["shop_image"]
    )

    if key not in unique_pairs:
        unique_pairs[key] = record

valid = list(unique_pairs.values())

print(
    f"After removing duplicate image pairs: "
    f"{len(valid):,}"
)


# ============================================================
# Group by category
# ============================================================

by_category = defaultdict(list)

for record in valid:
    by_category[record["category_name"]].append(record)


print("\nAvailable categories:")

for category in sorted(by_category):
    print(
        f"  {category:<30} "
        f"{len(by_category[category]):>8,}"
    )


# ============================================================
# Category-balanced sampling
# ============================================================

random.seed(SEED)

categories = sorted(by_category.keys())

# Determine a target quota per category.
base_quota = TARGET_PAIRS // len(categories)
remainder = TARGET_PAIRS % len(categories)

selected = []

for index, category in enumerate(categories):

    records_for_category = by_category[category].copy()

    random.shuffle(records_for_category)

    quota = base_quota

    if index < remainder:
        quota += 1

    quota = min(
        quota,
        len(records_for_category)
    )

    selected.extend(
        records_for_category[:quota]
    )


# ============================================================
# Shuffle final subset
# ============================================================

random.shuffle(selected)

# In case category availability caused a smaller subset.
selected = selected[:TARGET_PAIRS]


# ============================================================
# Create DataFrame
# ============================================================

df = pd.DataFrame(selected)

# Add deterministic subset index.
df.insert(
    0,
    "sample_id",
    range(1, len(df) + 1)
)


# ============================================================
# Save
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Report
# ============================================================

print("\n" + "=" * 80)
print("SUBSET CREATED")
print("=" * 80)

print(f"\nSelected pairs: {len(df):,}")
print(f"Output: {OUTPUT_FILE}")

print("\nCategory distribution:")

for category, count in (
    df["category_name"]
    .value_counts()
    .sort_index()
    .items()
):
    print(
        f"  {category:<30} {count:>6,}"
    )

print("\nStyle distribution:")

for style, count in (
    df["style"]
    .value_counts()
    .sort_index()
    .items()
):
    print(
        f"  style={style:<4} {count:>6,}"
    )

print("\nFirst 5 records:")

print(
    df[
        [
            "sample_id",
            "pair_id",
            "style",
            "category_name",
            "user_image",
            "shop_image"
        ]
    ].head().to_string(index=False)
)

print("\n" + "=" * 80)
print("Done")
print("=" * 80)