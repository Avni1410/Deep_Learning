import json

path = r"phase8_package\experiments\phase8\results\toptee_val_phase8_ranked.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

queries = data["queries"]
n = len(queries)

print(f"Queries: {n}")

for k in [1, 5, 10, 50]:
    hits = sum(q["target_rank"] <= k for q in queries)
    recall = hits / n
    print(f"R@{k}: {recall:.4f} ({hits}/{n})")
