import sys
from pathlib import Path
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

for csv_name in ["data/processed/deepfashion2/alignment_train.csv",
                  "data/processed/deepfashion2/alignment_val.csv"]:
    df = pd.read_csv(REPO_ROOT / csv_name)
    bad = []
    for _, row in df.iterrows():
        for col, subdir in [("user_image", "user"), ("shop_image", "shop")]:
            path = REPO_ROOT / "data/processed/deepfashion2/images" / subdir / row[col]
            try:
                with Image.open(path) as im:
                    im.convert("RGB").load()
            except Exception as e:
                bad.append((str(path), type(e).__name__, str(e)))
    print(f"{csv_name}: {len(bad)} bad images out of {len(df) * 2} checked")
    for p, etype, emsg in bad[:20]:
        print(f"  {p} -> {etype}: {emsg}")
