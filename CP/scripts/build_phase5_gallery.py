"""
Phase 5 - rebuild a gallery using the trained CrossDomainCIRModel's encode_visual
(CLIP + visual_adapter), since target embeddings now depend on the adapter.

The set of gallery images is identical to Phase 3/4's (same CSV, same unique-
target scan) - only the embeddings differ, because the encoder differs.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.fashioniq_dataset import FashionIQCIRDataset  # noqa: E402
from src.models.attribute_aware_cir import AttributeAwareCIRModel  # noqa: E402
from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig  # noqa: E402
from src.models.composed_retrieval_model import CrossDomainCIRModel, load_phase4_checkpoint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fiq-csv", type=Path, required=True)
    ap.add_argument("--fiq-root", type=Path, default=REPO_ROOT / "data/raw/fashioniq")
    ap.add_argument("--category", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--phase4-checkpoint", type=Path, required=True,
                    help="warm-start base for cross_attn/fusion")
    ap.add_argument("--phase5-checkpoint", type=Path, required=True,
                    help="trained cross_attn/fusion/visual_adapter for this variant")
    ap.add_argument("--variant", required=True, choices=["pair", "coral", "combined"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/processed/fashioniq_embeddings")
    args = ap.parse_args()

    ds = FashionIQCIRDataset(csv_path=args.fiq_csv, image_root=args.fiq_root)
    unique_targets: dict[str, Image.Image] = {}
    for sample in tqdm(ds, desc="Scanning for unique targets"):
        if sample["target"] not in unique_targets:
            unique_targets[sample["target"]] = sample["target_image"]
    ids = list(unique_targets.keys())
    images = [unique_targets[i] for i in ids]
    print(f"Unique target images: {len(ids)} (must match Phase 3/4 gallery size for this category)")

    clip = CLIPEncoder(CLIPEncoderConfig(model_name=args.model, device=args.device))
    phase4_model = AttributeAwareCIRModel(clip)
    load_phase4_checkpoint(phase4_model, args.phase4_checkpoint, clip.device)
    model = CrossDomainCIRModel(phase4_model).to(clip.device)

    ckpt = torch.load(args.phase5_checkpoint, map_location=clip.device, weights_only=False)
    model.cross_attn.load_state_dict(ckpt["trainable_state"]["cross_attn"])
    model.fusion.load_state_dict(ckpt["trainable_state"]["fusion"])
    model.visual_adapter.load_state_dict(ckpt["trainable_state"]["visual_adapter"])
    model.eval()
    print(f"Loaded Phase 5 [{args.variant}] checkpoint (epoch {ckpt['epoch']})")

    all_embeds = []
    t0 = time.time()
    with torch.no_grad():
        for start in tqdm(range(0, len(images), args.batch_size), desc="Encoding gallery (Phase 5)"):
            batch = images[start:start + args.batch_size]
            all_embeds.append(model.encode_visual(batch).detach().cpu())
    embeddings = torch.cat(all_embeds, dim=0)
    print(f"Encoded {len(ids)} images in {time.time() - t0:.1f}s")

    norms = embeddings.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3), "Gallery embeddings not unit-norm"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.category}_{args.split}_phase5_{args.variant}_gallery.pt"
    torch.save({
        "ids": ids, "embeddings": embeddings, "category": args.category, "split": args.split,
        "model": args.model, "variant": args.variant, "phase5_checkpoint": str(args.phase5_checkpoint),
        "embed_dim": clip.embed_dim,
    }, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()