from pathlib import Path
import hashlib
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "fashioniq"

QUERY_FILES = [
    PROJECT_ROOT / "data" / "processed" / "fashioniq" / "train_queries_dress.csv",
    PROJECT_ROOT / "data" / "processed" / "fashioniq" / "train_queries_shirt.csv",
    PROJECT_ROOT / "data" / "processed" / "fashioniq" / "train_queries_toptee.csv",
]

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "phase6"

OUTPUT_CSV = OUTPUT_DIR / "fashioniq_training_duplicate_groups.csv"


# ============================================================
# MD5 helper
# ============================================================

def md5_file(path):
    """Return the MD5 hash of a file."""

    hasher = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


# ============================================================
# Load training image IDs
# ============================================================

def collect_training_images():

    records = []

    for csv_path in QUERY_FILES:

        print(f"Reading: {csv_path}")

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Training CSV not found: {csv_path}"
            )

        df = pd.read_csv(csv_path)

        required_columns = {
            "category",
            "candidate",
            "target",
        }

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing columns in {csv_path}: {sorted(missing)}"
            )

        for _, row in df.iterrows():

            category = str(row["category"])

            candidate = str(row["candidate"])
            target = str(row["target"])

            records.append(
                {
                    "category": category,
                    "asin": candidate,
                }
            )

            records.append(
                {
                    "category": category,
                    "asin": target,
                }
            )

    # Remove duplicate references to the same image.
    unique_df = pd.DataFrame(records).drop_duplicates(
        subset=["category", "asin"]
    )

    return unique_df


# ============================================================
# Main duplicate analysis
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("FashionIQ Training Gallery Duplicate Analysis")
    print("=" * 70)
    print()

    images_df = collect_training_images()

    print()
    print(f"Unique training images referenced: {len(images_df):,}")
    print()

    hash_records = []

    missing_count = 0

    for i, row in enumerate(
        images_df.itertuples(index=False),
        start=1,
    ):

        category = row.category
        asin = row.asin

        image_path = (
            IMAGE_ROOT
            / category
            / f"{asin}.jpg"
        )

        if not image_path.exists():

            missing_count += 1

            print(
                f"[MISSING] {category}/{asin}.jpg"
            )

            continue

        file_hash = md5_file(image_path)

        hash_records.append(
            {
                "category": category,
                "asin": asin,
                "image_path": str(image_path),
                "md5": file_hash,
            }
        )

        if i % 1000 == 0:
            print(
                f"Processed {i:,} / {len(images_df):,}"
            )

    print()
    print(f"Existing images hashed: {len(hash_records):,}")
    print(f"Missing images: {missing_count:,}")
    print()

    hash_df = pd.DataFrame(hash_records)

    if hash_df.empty:
        print("No images were successfully hashed.")
        return

    # --------------------------------------------------------
    # Find exact duplicate image groups.
    #
    # Same MD5 = byte-for-byte identical image file.
    # --------------------------------------------------------

    duplicate_counts = (
        hash_df
        .groupby(["category", "md5"])
        .size()
        .reset_index(name="group_size")
    )

    duplicate_counts = duplicate_counts[
        duplicate_counts["group_size"] > 1
    ]

    print("=" * 70)
    print("Duplicate Summary")
    print("=" * 70)

    if duplicate_counts.empty:

        print("No exact duplicate image groups found.")

        # Still save an empty report with useful columns.
        empty_df = pd.DataFrame(
            columns=[
                "category",
                "md5",
                "group_size",
                "asin",
                "image_path",
            ]
        )

        empty_df.to_csv(
            OUTPUT_CSV,
            index=False,
        )

        print()
        print(f"Report saved to: {OUTPUT_CSV}")

        return

    # --------------------------------------------------------
    # Build detailed duplicate report.
    # --------------------------------------------------------

    duplicate_df = hash_df.merge(
        duplicate_counts,
        on=["category", "md5"],
        how="inner",
    )

    duplicate_df = duplicate_df.sort_values(
        by=[
            "category",
            "group_size",
            "md5",
            "asin",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
    )

    duplicate_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    duplicate_groups = len(duplicate_counts)

    duplicate_images = int(
        duplicate_counts["group_size"].sum()
    )

    extra_duplicates = duplicate_images - duplicate_groups

    print(
        f"Exact duplicate groups: {duplicate_groups:,}"
    )

    print(
        f"Images belonging to duplicate groups: "
        f"{duplicate_images:,}"
    )

    print(
        f"Redundant duplicate images: "
        f"{extra_duplicates:,}"
    )

    print()

    print("Duplicate groups by category:")

    category_summary = (
        duplicate_counts
        .groupby("category")
        .agg(
            duplicate_groups=("group_size", "size"),
            duplicate_images=("group_size", "sum"),
        )
        .reset_index()
    )

    print(
        category_summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Detailed report saved to:\n{OUTPUT_CSV}"
    )

    # --------------------------------------------------------
    # Print first 10 duplicate groups for inspection.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("First 10 Duplicate Groups")
    print("=" * 70)

    shown = 0

    for (category, md5), group in duplicate_df.groupby(
        ["category", "md5"],
        sort=False,
    ):

        print()
        print(
            f"Category: {category}"
        )
        print(
            f"MD5:      {md5}"
        )
        print(
            f"Count:    {len(group)}"
        )

        print(
            "ASINs:    "
            + ", ".join(group["asin"].tolist())
        )

        shown += 1

        if shown >= 10:
            break

    print()
    print("=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()