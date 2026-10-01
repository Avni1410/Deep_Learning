from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DUPLICATE_CSV = (
    PROJECT_ROOT
    / "experiments"
    / "phase6"
    / "fashioniq_training_duplicate_groups.csv"
)

QUERY_FILES = [
    PROJECT_ROOT / "data" / "processed" / "fashioniq" / "train_queries_dress.csv",
    PROJECT_ROOT / "data" / "processed" / "fashioniq" / "train_queries_shirt.csv",
    PROJECT_ROOT / "data" / "processed" / "fashioniq" / "train_queries_toptee.csv",
]

OUTPUT_CSV = (
    PROJECT_ROOT
    / "experiments"
    / "phase6"
    / "duplicate_query_role_audit.csv"
)


def main():

    print("=" * 70)
    print("FashionIQ Duplicate Image / Query Role Audit")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Load duplicate groups
    # --------------------------------------------------------

    if not DUPLICATE_CSV.exists():
        raise FileNotFoundError(
            f"Duplicate report not found:\n{DUPLICATE_CSV}"
        )

    duplicates = pd.read_csv(DUPLICATE_CSV)

    required_duplicate_columns = {
        "category",
        "md5",
        "group_size",
        "asin",
    }

    missing = required_duplicate_columns - set(duplicates.columns)

    if missing:
        raise ValueError(
            "Duplicate report is missing columns: "
            f"{sorted(missing)}"
        )

    print(
        f"Duplicate groups loaded: "
        f"{duplicates[['category', 'md5']].drop_duplicates().shape[0]:,}"
    )

    print()

    # --------------------------------------------------------
    # Load all training queries
    # --------------------------------------------------------

    query_frames = []

    for csv_path in QUERY_FILES:

        print(f"Reading: {csv_path}")

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Training CSV not found:\n{csv_path}"
            )

        df = pd.read_csv(csv_path)

        required_query_columns = {
            "category",
            "split",
            "query_id",
            "candidate",
            "target",
        }

        missing = required_query_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{csv_path} is missing columns: "
                f"{sorted(missing)}"
            )

        query_frames.append(
            df[
                [
                    "category",
                    "split",
                    "query_id",
                    "candidate",
                    "target",
                ]
            ]
        )

    queries = pd.concat(
        query_frames,
        ignore_index=True,
    )

    print()
    print(
        f"Training queries loaded: {len(queries):,}"
    )

    # --------------------------------------------------------
    # Build ASIN -> query role index
    # --------------------------------------------------------

    role_records = []

    for row in queries.itertuples(index=False):

        role_records.append(
            {
                "category": row.category,
                "asin": str(row.candidate),
                "role": "candidate",
                "query_id": row.query_id,
            }
        )

        role_records.append(
            {
                "category": row.category,
                "asin": str(row.target),
                "role": "target",
                "query_id": row.query_id,
            }
        )

    roles = pd.DataFrame(role_records)

    # --------------------------------------------------------
    # Join duplicate ASINs with their query roles
    # --------------------------------------------------------

    audit = duplicates.merge(
        roles,
        on=["category", "asin"],
        how="left",
    )

    # Only duplicate images that actually appear in a query role.
    audit = audit.dropna(
        subset=["query_id"]
    )

    # --------------------------------------------------------
    # Determine whether a duplicate group spans roles.
    # --------------------------------------------------------

    group_summary = (
        audit
        .groupby(
            ["category", "md5"],
            as_index=False,
        )
        .agg(
            group_size=("group_size", "first"),
            asin_count=("asin", "nunique"),
            candidate_asins=(
                "asin",
                lambda x: ", ".join(
                    sorted(
                        set(
                            audit.loc[
                                audit["asin"].isin(x)
                                & (audit["role"] == "candidate"),
                                "asin",
                            ]
                        )
                    )
                ),
            ),
            target_asins=(
                "asin",
                lambda x: ", ".join(
                    sorted(
                        set(
                            audit.loc[
                                audit["asin"].isin(x)
                                & (audit["role"] == "target"),
                                "asin",
                            ]
                        )
                    )
                ),
            ),
            roles_present=(
                "role",
                lambda x: ", ".join(
                    sorted(set(x))
                ),
            ),
        )
    )

    # --------------------------------------------------------
    # A duplicate group is cross-role if both candidate and
    # target occur somewhere in the training queries.
    # --------------------------------------------------------

    group_summary["cross_role"] = (
        group_summary["roles_present"]
        .str.contains("candidate")
        &
        group_summary["roles_present"]
        .str.contains("target")
    )

    # --------------------------------------------------------
    # Save detailed audit
    # --------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    group_summary.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_duplicate_groups = len(group_summary)

    cross_role_groups = int(
        group_summary["cross_role"].sum()
    )

    candidate_only_groups = int(
        (
            group_summary["roles_present"]
            == "candidate"
        ).sum()
    )

    target_only_groups = int(
        (
            group_summary["roles_present"]
            == "target"
        ).sum()
    )

    print()
    print("=" * 70)
    print("Role Audit Summary")
    print("=" * 70)

    print(
        f"Duplicate groups appearing in training queries: "
        f"{total_duplicate_groups:,}"
    )

    print(
        f"Candidate-only duplicate groups: "
        f"{candidate_only_groups:,}"
    )

    print(
        f"Target-only duplicate groups: "
        f"{target_only_groups:,}"
    )

    print(
        f"Cross-role duplicate groups: "
        f"{cross_role_groups:,}"
    )

    print()

    # --------------------------------------------------------
    # Print cross-role groups
    # --------------------------------------------------------

    if cross_role_groups == 0:

        print(
            "No duplicate image groups were found across "
            "candidate and target roles."
        )

    else:

        print("=" * 70)
        print("Cross-Role Duplicate Groups")
        print("=" * 70)

        cross_role = group_summary[
            group_summary["cross_role"]
        ]

        for row in cross_role.itertuples(index=False):

            print()
            print(
                f"Category: {row.category}"
            )

            print(
                f"MD5:      {row.md5}"
            )

            print(
                f"Group size: {row.group_size}"
            )

            print(
                f"Candidate ASINs: {row.candidate_asins}"
            )

            print(
                f"Target ASINs:    {row.target_asins}"
            )

    print()
    print("=" * 70)
    print("Audit complete.")
    print("=" * 70)

    print()
    print(
        f"Detailed report:\n{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()