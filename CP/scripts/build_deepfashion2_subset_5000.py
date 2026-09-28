import json
import random
from pathlib import Path

import pandas as pd


INPUT = Path("data/raw/deepfashion2/matching_pairs.json")
OUTPUT = Path("data/processed/deepfashion2/domain_alignment_pairs_5000.csv")

TARGET_PAIRS = 5000
SEED = 42


def main():
    random.seed(SEED)

    print("=" * 70)
    print("Building DeepFashion2 5,000-Pair Subset")
    print("=" * 70)

    print("\nLoading matching_pairs.json...")
    with open(INPUT, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Total metadata records: {len(records):,}")

    rows = []

    for record in records:
        if record.get("style", 0) <= 0:
            continue

        user = record.get("user_image", {})
        shop = record.get("shop_image", {})

        user_name = user.get("name")
        shop_name = shop.get("name")

        if not user_name or not shop_name:
            continue

        rows.append({
            "pair_id": record.get("pair_id"),
            "style": record.get("style"),
            "category_name": record.get("category_name"),
            "category_id": record.get("category_id"),
            "user_image": user_name,
            "shop_image": shop_name,
            "user_bbox": str(user.get("bbox")),
            "shop_bbox": str(shop.get("bbox")),
            "user_path": user.get("path"),
            "shop_path": shop.get("path"),
        })

    df = pd.DataFrame(rows)

    print(f"Valid records: {len(df):,}")

    # Remove exact duplicate image pairs
    df = df.drop_duplicates(
        subset=["user_image", "shop_image"]
    ).reset_index(drop=True)

    print(f"After duplicate removal: {len(df):,}")

    # Ensure we have enough records
    if len(df) < TARGET_PAIRS:
        raise ValueError(
            f"Only {len(df)} valid pairs available; "
            f"cannot select {TARGET_PAIRS}."
        )

    # Balanced sampling across categories
    categories = sorted(df["category_name"].dropna().unique())

    print(f"\nCategories: {len(categories)}")

    base_per_category = TARGET_PAIRS // len(categories)
    remainder = TARGET_PAIRS % len(categories)

    selected_parts = []

    for index, category in enumerate(categories):
        category_df = df[df["category_name"] == category].copy()

        n = base_per_category + (1 if index < remainder else 0)

        if len(category_df) < n:
            raise ValueError(
                f"Category '{category}' only has "
                f"{len(category_df)} pairs, needs {n}."
            )

        sampled = category_df.sample(
            n=n,
            random_state=SEED + index
        )

        selected_parts.append(sampled)

    selected = pd.concat(
        selected_parts,
        ignore_index=True
    )

    # Shuffle final dataset
    selected = selected.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    # Safety check
    if len(selected) != TARGET_PAIRS:
        raise ValueError(
            f"Expected {TARGET_PAIRS} pairs, "
            f"got {len(selected)}."
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    selected.to_csv(
        OUTPUT,
        index=False
    )

    print(f"\nSelected pairs: {len(selected):,}")
    print(f"Output: {OUTPUT}")

    print("\nCategory distribution:")
    print(
        selected["category_name"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nStyle distribution:")
    print(
        selected["style"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    user_images = set(selected["user_image"])
    shop_images = set(selected["shop_image"])
    all_images = user_images | shop_images

    print("\nImage statistics:")
    print(f"Unique user images: {len(user_images):,}")
    print(f"Unique shop images: {len(shop_images):,}")
    print(f"Unique total images: {len(all_images):,}")
    print(f"User/shop overlap: {len(user_images & shop_images):,}")

    print("\n" + "=" * 70)
    print("5,000-pair subset created successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()