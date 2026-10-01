import torch

gallery_path = "experiments/phase6/cache/combined_train_gallery.pt"

g = torch.load(
    gallery_path,
    map_location="cpu",
    weights_only=False
)

idx = {gid: i for i, gid in enumerate(g["ids"])}

pairs = [
    ("B00C40W020", "B009RV9EAA"),
    ("B0091GQV3E", "B0091GQTNQ"),
]

for a, b in pairs:
    if a not in idx:
        print(f"{a}: NOT FOUND")
        continue

    if b not in idx:
        print(f"{b}: NOT FOUND")
        continue

    e1 = g["embeddings"][idx[a]]
    e2 = g["embeddings"][idx[b]]

    print()
    print(a, b)
    print("cosine:", (e1 @ e2).item())
    print("identical:", torch.allclose(e1, e2))