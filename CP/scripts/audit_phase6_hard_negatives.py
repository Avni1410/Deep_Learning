"""
Phase 6 - Hard-negative mining audit.

Audits mined FashionIQ hard negatives for:

1. Missing negatives
2. Target/reference leakage
3. Exact-image duplicate leakage
4. Multiple negatives from the same duplicate group
5. Negative category distribution
6. Positive vs. negative similarity
7. Number of queries where negative_1 > positive
8. Average similarity margin
"""

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

PHASE6_ROOT = REPO_ROOT / "experiments" / "phase6"

DUPLICATE_LOOKUP = (
    PHASE6_ROOT / "fashioniq_duplicate_lookup.csv"
)

HARD_NEGATIVE_ROOT = (
    PHASE6_ROOT / "hard_negatives"
)

CSV_FILES = {
    "dress": (
        HARD_NEGATIVE_ROOT
        / "dress_train_similarity_negatives.csv"
    ),
    "shirt": (
        HARD_NEGATIVE_ROOT
        / "shirt_train_similarity_negatives.csv"
    ),
    "toptee": (
        HARD_NEGATIVE_ROOT
        / "toptee_train_similarity_negatives.csv"
    ),
}



def load_duplicate_lookup():
    df = pd.read_csv(DUPLICATE_LOOKUP)

    asin_to_group = {}

    for row in df.itertuples(index=False):
        asin_to_group[
            (str(row.category), str(row.asin))
        ] = str(row.duplicate_group_id)

    return asin_to_group


def audit_file(
    category,
    csv_path,
    asin_to_group,
):
    print("\n" + "=" * 70)
    print(f"AUDITING: {category.upper()}")
    print("=" * 70)

    if not csv_path.exists():
        print(f"ERROR: CSV not found:")
        print(csv_path)
        return None

    df = pd.read_csv(csv_path)

    print(f"CSV: {csv_path}")
    print(f"Queries: {len(df)}")

    required = {
        "query_id",
        "category",
        "target",
        "candidate",
        "negative_1",
        "negative_2",
        "negative_3",
        "similarity_1",
        "similarity_2",
        "similarity_3",
        "positive_similarity",
        "negative_1_category",
        "negative_2_category",
        "negative_3_category",
    }

    missing = required - set(df.columns)

    if missing:
        print(f"\nERROR: Missing columns: {sorted(missing)}")
        return None

    # ---------------------------------------------------------------
    # Basic missing-value check
    # ---------------------------------------------------------------

    negative_columns = [
        "negative_1",
        "negative_2",
        "negative_3",
    ]

    missing_negatives = df[negative_columns].isna().any(axis=1)

    print(
        f"\nMissing-negative queries: "
        f"{missing_negatives.sum()}"
    )

    # ---------------------------------------------------------------
    # Target/reference leakage
    # ---------------------------------------------------------------

    leakage_rows = []

    for _, row in df.iterrows():

        forbidden = {
            str(row["target"]),
            str(row["candidate"]),
        }

        negatives = [
            str(row[col])
            for col in negative_columns
        ]

        leaked = [
            neg
            for neg in negatives
            if neg in forbidden
        ]

        if leaked:
            leakage_rows.append(
                (
                    row["query_id"],
                    leaked,
                )
            )

    print(
        f"Target/reference leakage: "
        f"{len(leakage_rows)}"
    )

    if leakage_rows:
        print("Examples:")
        for item in leakage_rows[:5]:
            print(" ", item)

    # ---------------------------------------------------------------
    # Exact duplicate-group leakage
    #
    # A negative must not belong to the same exact-image group
    # as the target or candidate.
    # ---------------------------------------------------------------

    duplicate_leakage = []

    for _, row in df.iterrows():

        query_category = str(row["category"])

        forbidden_groups = set()

        for asin_column in ["target", "candidate"]:

            asin = str(row[asin_column])

            group = asin_to_group.get(
                (query_category, asin)
            )

            if group is not None:
                forbidden_groups.add(group)

        negatives = [
            str(row[col])
            for col in negative_columns
        ]

        for neg_col, neg in zip(
            negative_columns,
            negatives,
        ):

            neg_category_column = (
                neg_col.replace(
                    "negative_",
                    "negative_",
                )
                + "_category"
            )

            # Easier explicit mapping.
            number = neg_col.split("_")[1]

            neg_category = str(
                row[f"negative_{number}_category"]
            )

            neg_group = asin_to_group.get(
                (neg_category, neg)
            )

            if (
                neg_group is not None
                and neg_group in forbidden_groups
            ):
                duplicate_leakage.append(
                    (
                        row["query_id"],
                        neg,
                        neg_category,
                        neg_group,
                    )
                )

    print(
        f"Target/candidate duplicate-group leakage: "
        f"{len(duplicate_leakage)}"
    )

    if duplicate_leakage:
        print("Examples:")
        for item in duplicate_leakage[:5]:
            print(" ", item)

    # ---------------------------------------------------------------
    # Multiple negatives from same duplicate group
    # ---------------------------------------------------------------

    repeated_groups = []

    for _, row in df.iterrows():

        groups = []

        for number in [1, 2, 3]:

            asin = str(
                row[f"negative_{number}"]
            )

            neg_category = str(
                row[
                    f"negative_{number}_category"
                ]
            )

            group = asin_to_group.get(
                (neg_category, asin)
            )

            if group is not None:
                groups.append(
                    (
                        neg_category,
                        group,
                        asin,
                    )
                )

        group_keys = [
            (category_name, group_id)
            for category_name, group_id, _ in groups
        ]

        if len(group_keys) != len(set(group_keys)):

            repeated_groups.append(
                (
                    row["query_id"],
                    groups,
                )
            )

    print(
        f"Repeated duplicate groups among negatives: "
        f"{len(repeated_groups)}"
    )

    if repeated_groups:
        print("Examples:")
        for item in repeated_groups[:5]:
            print(" ", item)

    # ---------------------------------------------------------------
    # Similarity analysis
    # ---------------------------------------------------------------

    similarity_columns = [
        "similarity_1",
        "similarity_2",
        "similarity_3",
    ]

    positive = pd.to_numeric(
        df["positive_similarity"],
        errors="coerce",
    )

    negatives = df[similarity_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    negative_1 = negatives["similarity_1"]

    valid = (
        positive.notna()
        & negative_1.notna()
    )

    positive_valid = positive[valid]
    negative_1_valid = negative_1[valid]

    margin = (
        positive_valid
        - negative_1_valid
    )

    hard_cases = (
        negative_1_valid
        > positive_valid
    )

    print(
        f"\nValid similarity rows: "
        f"{valid.sum()}"
    )

    if valid.any():

        print(
            f"Mean positive similarity: "
            f"{positive_valid.mean():.6f}"
        )

        print(
            f"Mean negative-1 similarity: "
            f"{negative_1_valid.mean():.6f}"
        )

        print(
            f"Mean margin "
            f"(positive - negative-1): "
            f"{margin.mean():.6f}"
        )

        print(
            f"Queries with negative-1 > positive: "
            f"{hard_cases.sum()}/{len(hard_cases)} "
            f"({100 * hard_cases.mean():.2f}%)"
        )

        print(
            f"Queries with negative-1 <= positive: "
            f"{(~hard_cases).sum()}/{len(hard_cases)} "
            f"({100 * (~hard_cases).mean():.2f}%)"
        )

    # ---------------------------------------------------------------
    # Category distribution
    # ---------------------------------------------------------------

    category_columns = [
        "negative_1_category",
        "negative_2_category",
        "negative_3_category",
    ]

    category_counts = (
        pd.concat(
            [
                df[column].astype(str)
                for column in category_columns
            ],
            ignore_index=True,
        )
        .value_counts()
    )

    print("\nNegative category distribution:")

    for cat, count in category_counts.items():

        percentage = (
            100.0
            * count
            / category_counts.sum()
        )

        print(
            f"  {cat:10s}: "
            f"{count:3d} "
            f"({percentage:6.2f}%)"
        )

    # ---------------------------------------------------------------
    # Per-query uniqueness
    # ---------------------------------------------------------------

    duplicate_negative_ids = []

    for _, row in df.iterrows():

        negatives = [
            str(row[f"negative_{i}"])
            for i in [1, 2, 3]
        ]

        if len(negatives) != len(set(negatives)):

            duplicate_negative_ids.append(
                (
                    row["query_id"],
                    negatives,
                )
            )

    print(
        f"\nRepeated negative ASINs within query: "
        f"{len(duplicate_negative_ids)}"
    )

    # ---------------------------------------------------------------
    # Final status
    # ---------------------------------------------------------------

    passed = (
        missing_negatives.sum() == 0
        and len(leakage_rows) == 0
        and len(duplicate_leakage) == 0
        and len(repeated_groups) == 0
        and len(duplicate_negative_ids) == 0
    )

    print(
        "\nAUDIT STATUS:",
        "PASS" if passed else "FAIL",
    )

    return {
        "category": category,
        "queries": len(df),
        "missing_negatives": int(
            missing_negatives.sum()
        ),
        "target_reference_leakage": len(
            leakage_rows
        ),
        "duplicate_group_leakage": len(
            duplicate_leakage
        ),
        "repeated_duplicate_groups": len(
            repeated_groups
        ),
        "repeated_negative_ids": len(
            duplicate_negative_ids
        ),
        "mean_positive": (
            float(positive_valid.mean())
            if valid.any()
            else float("nan")
        ),
        "mean_negative_1": (
            float(negative_1_valid.mean())
            if valid.any()
            else float("nan")
        ),
        "mean_margin": (
            float(margin.mean())
            if valid.any()
            else float("nan")
        ),
        "negative_1_gt_positive": int(
            hard_cases.sum()
        )
        if valid.any()
        else 0,
        "audit_pass": passed,
    }


def main():

    print("=" * 70)
    print("PHASE 6 HARD-NEGATIVE AUDIT")
    print("=" * 70)

    if not DUPLICATE_LOOKUP.exists():

        raise FileNotFoundError(
            f"Duplicate lookup not found:\n"
            f"{DUPLICATE_LOOKUP}"
        )

    asin_to_group = load_duplicate_lookup()

    print(
        f"\nLoaded duplicate lookup: "
        f"{len(asin_to_group)} ASINs"
    )

    results = []

    for category, csv_path in CSV_FILES.items():

        result = audit_file(
            category,
            csv_path,
            asin_to_group,
        )

        if result is not None:
            results.append(result)

    if not results:
        print("\nNo CSV files were audited.")
        return

    # ---------------------------------------------------------------
    # Overall summary
    # ---------------------------------------------------------------

    summary = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    print(
        summary[
            [
                "category",
                "queries",
                "missing_negatives",
                "target_reference_leakage",
                "duplicate_group_leakage",
                "repeated_duplicate_groups",
                "repeated_negative_ids",
                "mean_positive",
                "mean_negative_1",
                "mean_margin",
                "negative_1_gt_positive",
                "audit_pass",
            ]
        ].to_string(index=False)
    )

    all_pass = bool(
        summary["audit_pass"].all()
    )

    print("\n" + "=" * 70)

    if all_pass:
        print("OVERALL AUDIT: PASS")
        print(
            "The mined negatives are structurally safe "
            "for the next inspection stage."
        )
    else:
        print("OVERALL AUDIT: FAIL")
        print(
            "Fix the reported issue(s) before Phase 6 training."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()