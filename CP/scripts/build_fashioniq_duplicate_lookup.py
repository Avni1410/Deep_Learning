from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "experiments"
    / "phase6"
    / "fashioniq_training_duplicate_groups.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "experiments"
    / "phase6"
    / "fashioniq_duplicate_lookup.csv"
)


def main():

    print("=" * 70)
    print("Building FashionIQ Duplicate-Image Lookup")
    print("=" * 70)
    print()

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required = {
        "category",
        "md5",
        "group_size",
        "asin",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # Each row represents:
    #
    # category + ASIN + exact image hash
    #
    # Therefore this directly gives us:
    #
    # ASIN -> duplicate image group
    # --------------------------------------------------------

    lookup = df[
        [
            "category",
            "asin",
            "md5",
            "group_size",
        ]
    ].copy()

    lookup = lookup.sort_values(
        [
            "category",
            "md5",
            "asin",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Assign a compact group ID.
    # --------------------------------------------------------

    lookup["duplicate_group_id"] = (
        lookup["category"]
        + "::"
        + lookup["md5"]
    )

    # Put group ID first.
    lookup = lookup[
        [
            "duplicate_group_id",
            "category",
            "asin",
            "md5",
            "group_size",
        ]
    ]

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lookup.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        f"ASINs in duplicate groups: {len(lookup):,}"
    )

    print(
        f"Duplicate groups: "
        f"{lookup['duplicate_group_id'].nunique():,}"
    )

    print()

    print("Example lookup rows:")

    print(
        lookup.head(10).to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved to:\n{OUTPUT_CSV}"
    )

    print()
    print("=" * 70)
    print("Complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()