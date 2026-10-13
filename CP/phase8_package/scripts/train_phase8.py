"""
Phase 8 - controlled hard-negative mining experiment, starting from the
validated Phase 4 checkpoint (NOT Phase 5 or Phase 6).

Corrects the diagnosed Phase 6 failure mode (mining negatives that already
outranked the positive in 95.7-96.9% of queries) with three changes:

  1. SEMI-HARD mining: a negative is only eligible if its similarity to the
     query is strictly below the positive's similarity at mining time.
  2. ITERATIVE RE-MINING: negatives are re-mined from the just-trained
     checkpoint after each round, not mined once and reused for all epochs.
  3. WARM-UP on the hard-negative loss weight (0 -> lambda_hn_target over
     the first --warmup-steps of each round), so the shared cross-attention/
     fusion parameters are not hit with a large hard-negative gradient the
     instant a (still partially undertrained) checkpoint starts a new round.

Each round = (re-mine from current checkpoint) + (train epochs_per_round
epochs). Default: 2 rounds x 1 epoch = a modest, controlled experiment,
matching the explicit instruction not to run huge, uncontrolled compute.

Does NOT touch experiments/phase4/phase4_deliverable/checkpoints/epoch_2.pt -
that file is only ever read (map_location), never written to. All Phase 8
checkpoints are saved under a separate experiments/phase8/ tree.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.fashioniq_dataset import FashionIQCIRDataset  # noqa: E402
from src.data.fashioniq_paths import load_fashioniq_image  # noqa: E402
from src.losses.contrastive import symmetric_contrastive_loss  # noqa: E402
from src.losses.semi_hard_loss import hard_negative_loss_variable  # noqa: E402
from src.models.attribute_aware_cir import AttributeAwareCIRModel  # noqa: E402
from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig  # noqa: E402
from src.models.composed_retrieval_model import load_phase4_checkpoint  # noqa: E402
from src.retrieval.hard_negative_miner import load_duplicate_lookup  # noqa: E402
from src.retrieval.semi_hard_miner import mine_semi_hard_for_query  # noqa: E402
from src.utils.checkpoint_compat import patch_path_compat  # noqa: E402

patch_path_compat()


# --------------------------------------------------------------------------- mining (inline, per round)
def mine_round(model, clip, fiq_csvs: list[Path], categories: list[str], fiq_root: Path,
                combined_gallery_path: Path, duplicate_lookup_path: Path,
                strategy: str, k_retrieval: int, n_hard: int, margin_buffer: float,
                limit: int | None, round_idx: int, out_dir: Path) -> dict[str, Path]:
    combined = torch.load(combined_gallery_path, map_location="cpu", weights_only=False)
    gallery_ids, gallery_categories, gallery_embeds = (
        combined["ids"], combined["categories"], combined["embeddings"]
    )
    asin_to_group, group_to_asins = load_duplicate_lookup(duplicate_lookup_path)

    out_paths = {}
    model.eval()
    for csv_path, category in zip(fiq_csvs, categories):
        ds = FashionIQCIRDataset(csv_path=csv_path, image_root=fiq_root)
        n = len(ds) if limit is None else min(limit, len(ds))
        rows = []
        shortfalls = 0
        with torch.no_grad():
            for i in tqdm(range(n), desc=f"Round {round_idx} mining ({category})"):
                s = ds[i]
                q = model.encode_query([s["reference_image"]], [s["caption_1"]], [s["caption_2"]])["query_embedding"]
                q = q.squeeze(0).cpu()
                m = mine_semi_hard_for_query(
                    q, s["query_id"], s["category"], s["target"], s["candidate"],
                    gallery_ids, gallery_embeds, gallery_categories,
                    asin_to_group, group_to_asins, k_retrieval, n_hard, strategy, margin_buffer,
                )
                if m.shortfall > 0:
                    shortfalls += 1
                rows.append(m)
        if shortfalls:
            print(f"  [round {round_idx}, {category}] {shortfalls}/{n} queries had < {n_hard} "
                  f"semi-hard negatives available (not padded).")

        out_path = out_dir / f"round{round_idx}_{category}_negatives.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        max_n = n_hard
        # Field names match what scripts/audit_phase8_semihard_negatives.py
        # expects (n_requested/n_found/shortfall, similarity_i), so the same
        # audit tool works on both standalone-mined and in-training-mined CSVs.
        fieldnames = ["query_id", "category", "target", "candidate", "positive_similarity",
                      "n_requested", "n_found", "shortfall"] + \
                     [f"negative_{i+1}" for i in range(max_n)] + \
                     [f"similarity_{i+1}" for i in range(max_n)] + \
                     [f"negative_{i+1}_category" for i in range(max_n)]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for m in rows:
                row = {"query_id": m.query_id, "category": m.category, "target": m.target,
                       "candidate": m.candidate, "positive_similarity": m.positive_similarity,
                       "n_requested": m.n_requested, "n_found": m.n_found, "shortfall": m.shortfall}
                for i in range(max_n):
                    row[f"negative_{i+1}"] = m.negative_ids[i] if i < len(m.negative_ids) else ""
                    row[f"similarity_{i+1}"] = m.negative_similarities[i] if i < len(m.negative_similarities) else ""
                    row[f"negative_{i+1}_category"] = m.negative_categories[i] if i < len(m.negative_categories) else ""
                w.writerow(row)
        out_paths[category] = out_path
        print(f"  Saved {len(rows)} rows -> {out_path}")
    return out_paths


# --------------------------------------------------------------------------- dataset for one training round
class SemiHardRoundDataset(Dataset):
    """Wraps one category's FashionIQCIRDataset together with that round's
    mined-negatives CSV. Negative image lists may be SHORTER than n_hard
    for some queries (semi-hard shortfall) - never padded with an invalid
    image; the collate function and loss handle variable length directly."""

    def __init__(self, fiq_csv: Path, fiq_root: Path, mined_csv: Path, category: str):
        self.base = FashionIQCIRDataset(csv_path=fiq_csv, image_root=fiq_root)
        self.fiq_root = Path(fiq_root)
        self.category = category
        mined = pd.read_csv(mined_csv, dtype=str, keep_default_na=False)
        self.mined_by_qid = {row["query_id"]: row for _, row in mined.iterrows()}
        # Build query_id -> base-dataset index once (one pass, matching the
        # precedent set in build_fashioniq_gallery.py's "scan for unique
        # targets" - a one-time setup cost, not a per-epoch cost).
        self.qid_to_index = {}
        for idx in range(len(self.base)):
            sample = self.base[idx]
            self.qid_to_index[sample["query_id"]] = idx
        self.query_ids = [qid for qid in self.mined_by_qid if qid in self.qid_to_index]

    def __len__(self) -> int:
        return len(self.query_ids)

    def __getitem__(self, i: int) -> dict:
        qid = self.query_ids[i]
        base_idx = self.qid_to_index[qid]
        sample = self.base[base_idx]
        mined_row = self.mined_by_qid[qid]

        negative_images: list[Image.Image] = []
        j = 1
        while f"negative_{j}" in mined_row:
            neg_id = mined_row[f"negative_{j}"]
            neg_cat = mined_row.get(f"negative_{j}_category", "")
            if neg_id:
                negative_images.append(load_fashioniq_image(self.fiq_root, neg_cat, neg_id))
            j += 1

        return {
            "query_id": qid, "category": sample["category"],
            "reference_image": sample["reference_image"], "target_image": sample["target_image"],
            "caption_1": sample["caption_1"], "caption_2": sample["caption_2"],
            "negative_images": negative_images,  # list, length 0..n_hard
        }


def round_collate(batch: list[dict]) -> dict:
    out = {k: [b[k] for b in batch] for k in batch[0].keys() if k != "negative_images"}
    out["negative_images"] = [b["negative_images"] for b in batch]  # list of lists, variable length
    return out


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-checkpoint", type=Path, required=True,
                    help="Phase 4 checkpoint - read-only, never modified")
    ap.add_argument("--fiq-train-csvs", type=Path, nargs="+", required=True)
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--fiq-root", type=Path, default=REPO_ROOT / "data/raw/fashioniq")
    ap.add_argument("--combined-gallery", type=Path, required=True)
    ap.add_argument("--duplicate-lookup", type=Path,
                    default=REPO_ROOT / "experiments/phase6/fashioniq_duplicate_lookup.csv")
    ap.add_argument("--strategy", default="similarity", choices=["similarity", "category_similarity"])
    ap.add_argument("--k-retrieval", type=int, default=50)
    ap.add_argument("--n-hard", type=int, default=3)
    ap.add_argument("--margin-buffer", type=float, default=0.0)
    ap.add_argument("--rounds", type=int, default=2, help="mine-then-train rounds")
    ap.add_argument("--epochs-per-round", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=2,
                    help="matches Phase 6's convention; effective batch = batch_size * grad_accum")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--lambda-hn-target", type=float, default=0.2,
                    help="down from Phase 6's 0.5, given semi-hard negatives produce a larger "
                         "per-unit-weight loss than impossible ones did")
    ap.add_argument("--warmup-steps", type=int, default=200, help="per round, lambda_hn ramps 0 -> target")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="cap queries per category, for smoke tests")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments/phase8")
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args()
    if len(args.fiq_train_csvs) != len(args.categories):
        raise SystemExit("--fiq-train-csvs and --categories must have matching length/order.")

    torch.manual_seed(args.seed)
    run_dir = args.out_dir / args.run_name
    mining_dir = run_dir / "hard_negatives"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["round", "epoch", "step", "loss_cir", "loss_hn", "hn_samples_used",
                                 "lambda_hn_current", "elapsed_s"])

    clip = CLIPEncoder(CLIPEncoderConfig(model_name=args.model, device=args.device))
    model = AttributeAwareCIRModel(clip)
    epoch4, _ = load_phase4_checkpoint(model, args.init_checkpoint, clip.device)
    model.to(clip.device)
    print(f"Initialized from Phase 4 checkpoint (epoch {epoch4}): {args.init_checkpoint}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} | "
          f"Frozen (CLIP): {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=args.lr, weight_decay=args.weight_decay)

    current_checkpoint_for_mining = args.init_checkpoint
    t0 = time.time()
    global_step = 0

    for round_idx in range(args.rounds):
        print(f"\n===== Round {round_idx}: mining from {current_checkpoint_for_mining} =====")
        mined_paths = mine_round(
            model, clip, args.fiq_train_csvs, args.categories, args.fiq_root,
            args.combined_gallery, args.duplicate_lookup, args.strategy,
            args.k_retrieval, args.n_hard, args.margin_buffer, args.limit, round_idx, mining_dir,
        )

        datasets = [
            SemiHardRoundDataset(csv_path, args.fiq_root, mined_paths[cat], cat)
            for csv_path, cat in zip(args.fiq_train_csvs, args.categories)
        ]
        round_dataset = ConcatDataset(datasets)
        loader = DataLoader(round_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=round_collate)
        print(f"Round {round_idx} training set: {len(round_dataset)} queries, {len(loader)} batches/epoch")

        use_amp = clip.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        model.train()
        round_step = 0
        optimizer.zero_grad()
        for epoch in range(args.epochs_per_round):
            for step, batch in enumerate(loader):
                lambda_hn = args.lambda_hn_target * min(1.0, round_step / max(1, args.warmup_steps))

                with torch.autocast(device_type=clip.device.type, enabled=use_amp):
                    out = model(batch["reference_image"], batch["caption_1"], batch["caption_2"], batch["target_image"])
                    cir_loss = symmetric_contrastive_loss(
                        out["query_embedding"], out["target_embedding"], args.temperature
                    )

                    neg_embeds_list = []
                    for neg_imgs in batch["negative_images"]:
                        if len(neg_imgs) == 0:
                            neg_embeds_list.append(None)
                        else:
                            # CLIP is frozen (CLIPEncoder wraps this in its own
                            # internal no_grad when freeze=True), but the
                            # resulting embedding still needs to be part of
                            # this step's graph so the hard-negative loss can
                            # backprop into cross_attn/fusion via the query
                            # embedding side of the dot product.
                            neg_embeds_list.append(model.clip.encode_image(neg_imgs).global_embed)
                    hn_loss, n_used = hard_negative_loss_variable(
                        out["query_embedding"], out["target_embedding"], neg_embeds_list, args.temperature
                    )

                    total = cir_loss + lambda_hn * hn_loss
                    loss_to_backward = total / args.grad_accum

                scaler.scale(loss_to_backward).backward()
                if (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                if round_step % 20 == 0:
                    elapsed = time.time() - t0
                    print(f"round {round_idx} epoch {epoch} step {round_step}/{len(loader)} "
                          f"cir={cir_loss.item():.4f} hn={hn_loss.item():.4f} "
                          f"(used {n_used}/{len(batch['query_id'])}) lambda_hn={lambda_hn:.3f} "
                          f"elapsed={elapsed:.0f}s")
                    with open(log_path, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([round_idx, epoch, round_step, cir_loss.item(),
                                                 hn_loss.item(), n_used, lambda_hn, elapsed])
                round_step += 1
                global_step += 1

        ckpt_path = ckpt_dir / f"round{round_idx}.pt"
        torch.save({
            "round": round_idx, "global_step": global_step,
            "trainable_state": {
                "cross_attn": model.cross_attn.state_dict(),
                "fusion": model.fusion.state_dict(),
                "attribute_heads": model.attribute_heads.state_dict(),
            },
            "optimizer_state": optimizer.state_dict(),
            "config": vars(args),
        }, ckpt_path)
        print(f"Saved round {round_idx} checkpoint: {ckpt_path}")
        current_checkpoint_for_mining = ckpt_path  # next round mines from THIS checkpoint

    print(f"\nPhase 8 training complete in {time.time() - t0:.0f}s. "
          f"Final checkpoint: {current_checkpoint_for_mining}")


if __name__ == "__main__":
    main()
