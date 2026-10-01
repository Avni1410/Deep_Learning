from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset


class FashionIQHardNegativeDataset(Dataset):
    """
    Loads FashionIQ CIR queries together with
    their mined hard negatives.
    """

    def __init__(
        self,
        csv_path,
        image_root,
        quality_exclusions=None,
    ):
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)

        df = pd.read_csv(self.csv_path)

        if quality_exclusions is not None:
            quality_exclusions = set(quality_exclusions)
            df = df[
                ~df["query_id"].isin(quality_exclusions)
            ]

        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        category = str(row["category"])

        target = str(row["target"])
        candidate = str(row["candidate"])

        negatives = [
            str(row["negative_1"]),
            str(row["negative_2"]),
            str(row["negative_3"]),
        ]

        negative_categories = [
            str(row["negative_1_category"]),
            str(row["negative_2_category"]),
            str(row["negative_3_category"]),
        ]

        target_path = (
            self.image_root
            / category
            / f"{target}.jpg"
        )

        candidate_path = (
            self.image_root
            / category
            / f"{candidate}.jpg"
        )

        negative_paths = [
            self.image_root
            / neg_category
            / f"{neg}.jpg"
            for neg, neg_category
            in zip(negatives, negative_categories)
        ]

        return {
            "query_id": str(row["query_id"]),
            "category": category,
            "target": target,
            "candidate": candidate,
            "negatives": negatives,
            "negative_categories": negative_categories,
            "target_path": str(target_path),
            "candidate_path": str(candidate_path),
            "negative_paths": [
                str(p) for p in negative_paths
            ],
        }


def load_quality_exclusions(path):
    path = Path(path)

    if not path.exists():
        return []

    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]
