import json
from pathlib import Path
from collections import Counter

PAIR_FILE = Path("data/raw/deepfashion2/matching_pairs.json")

print("=" * 80)
print("DeepFashion2 Pair Analysis")
print("=" * 80)

with open(PAIR_FILE, "r", encoding="utf-8") as f:
    pairs = json.load(f)

print(f"\nTotal pair records: {len(pairs):,}")

# ------------------------------------------------------------------
# Basic statistics
# ------------------------------------------------------------------

pair_ids = [p["pair_id"] for p in pairs]
styles = [p["style"] for p in pairs]
categories = [p["category_name"] for p in pairs]
category_ids = [p["category_id"] for p in pairs]

print("\n" + "=" * 80)
print("PAIR STATISTICS")
print("=" * 80)

print(f"Unique pair IDs:       {len(set(pair_ids)):,}")
print(f"Unique styles:         {len(set(styles)):,}")
print(f"Unique categories:     {len(set(categories)):,}")
print(f"Unique category IDs:   {len(set(category_ids)):,}")

# ------------------------------------------------------------------
# Category distribution
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("CATEGORY DISTRIBUTION")
print("=" * 80)

category_counts = Counter(categories)

for category, count in category_counts.most_common():
    print(f"{category:<30} {count:>10,}")

# ------------------------------------------------------------------
# Style distribution
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("STYLE DISTRIBUTION")
print("=" * 80)

style_counts = Counter(styles)

for style, count in sorted(style_counts.items()):
    print(f"style={style:<5} {count:>10,}")

# ------------------------------------------------------------------
# Image paths
# ------------------------------------------------------------------

shop_paths = [p["shop_image"]["path"] for p in pairs]
user_paths = [p["user_image"]["path"] for p in pairs]

print("\n" + "=" * 80)
print("IMAGE PATH ANALYSIS")
print("=" * 80)

print(f"Unique shop images: {len(set(shop_paths)):,}")
print(f"Unique user images: {len(set(user_paths)):,}")

all_paths = shop_paths + user_paths

train_count = sum("/train/" in x.replace("\\", "/") for x in all_paths)
val_count = sum("/validation/" in x.replace("\\", "/") for x in all_paths)
test_count = sum("/test/" in x.replace("\\", "/") for x in all_paths)

print(f"\nTrain references:      {train_count:,}")
print(f"Validation references: {val_count:,}")
print(f"Test references:       {test_count:,}")

# ------------------------------------------------------------------
# Duplicate image usage
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("IMAGE REUSE")
print("=" * 80)

shop_counts = Counter(shop_paths)
user_counts = Counter(user_paths)

duplicate_shop = sum(1 for _, count in shop_counts.items() if count > 1)
duplicate_user = sum(1 for _, count in user_counts.items() if count > 1)

print(f"Shop images appearing in multiple records: {duplicate_shop:,}")
print(f"User images appearing in multiple records: {duplicate_user:,}")

# ------------------------------------------------------------------
# Category + style combinations
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("CATEGORY + STYLE COMBINATIONS")
print("=" * 80)

combination_counts = Counter(
    (p["category_name"], p["style"])
    for p in pairs
)

for (category, style), count in combination_counts.most_common(30):
    print(
        f"{category:<30} "
        f"style={style:<4} "
        f"{count:>10,}"
    )

# ------------------------------------------------------------------
# Sample records
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("SAMPLE RECORDS")
print("=" * 80)

for i, pair in enumerate(pairs[:5], start=1):
    print(f"\nPair {i}")
    print(f"pair_id:       {pair['pair_id']}")
    print(f"style:         {pair['style']}")
    print(f"category:      {pair['category_name']}")
    print(f"shop image:    {pair['shop_image']['name']}")
    print(f"shop path:     {pair['shop_image']['path']}")
    print(f"user image:    {pair['user_image']['name']}")
    print(f"user path:     {pair['user_image']['path']}")

print("\n" + "=" * 80)
print("Analysis complete")
print("=" * 80)