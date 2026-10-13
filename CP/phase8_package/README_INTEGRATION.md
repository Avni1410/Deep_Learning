# Phase 8 Integration Guide — Controlled Semi-Hard Negative Mining

## 0. What this fixes, in one sentence

Phase 6's hard-negative mining selected the top-K most similar gallery images regardless
of whether they ranked above or below the true positive — the audit showed this happened
in **95.7–96.9%** of queries (`negative_1 > positive`), meaning Phase 6 trained against
the model's own top ranking *errors*, not fair "hard but still correct" negatives. Phase 8
fixes this by construction: a candidate is only eligible as a negative if its similarity
is **strictly below** the positive's similarity at mining time (*semi-hard* mining), and
negatives are **re-mined after each training round** from the updated checkpoint rather
than mined once and reused.

## 1. Where to extract

Extract the zip so its contents merge into your existing repo root:

```
C:\Users\selot\OneDrive\Documents\5SEM-CP\DL\Deep_Learning\CP\
```

That is: `phase8_package/src/...` → `CP\src\...`, `phase8_package/scripts/...` → `CP\scripts\...`.
No existing file is overwritten — every path below is new.

## 2. Files this creates

```
CP\
├── src\
│   ├── data\
│   │   └── fashioniq_paths.py              (NEW)
│   ├── losses\
│   │   └── semi_hard_loss.py               (NEW)
│   ├── retrieval\
│   │   └── semi_hard_miner.py              (NEW)
│   └── utils\
│       ├── __init__.py                     (NEW, empty)
│       └── checkpoint_compat.py            (NEW)
├── scripts\
│   ├── mine_phase8_hard_negatives.py       (NEW)
│   ├── train_phase8.py                     (NEW)
│   ├── evaluate_phase8.py                  (NEW)
│   ├── compare_phase8_phase4.py            (NEW)
│   ├── audit_phase8_semihard_negatives.py  (NEW)
│   └── test_phase8_pipeline.py             (NEW)
└── docs\
    └── phase_8_semi_hard_mining.md         (NEW, template — fill in real results after running)
```

**Create one empty file first** (needed for the `src/utils` package to import correctly):
```powershell
New-Item -ItemType File -Path "src\utils\__init__.py" -Force
```

## 3. What this depends on (must already exist in your repo)

- `experiments/phase4/phase4_deliverable/checkpoints/epoch_2.pt` — **read-only**, never modified
- `experiments/phase6/cache/combined_train_gallery.pt` — the 16,695-item combined gallery you already built
- `experiments/phase6/fashioniq_duplicate_lookup.csv` — the duplicate-group lookup you already built
- `src/data/fashioniq_dataset.py` (`FashionIQCIRDataset`)
- `src/models/attribute_aware_cir.py` (`AttributeAwareCIRModel`)
- `src/models/clip_encoder.py` (`CLIPEncoder`, `CLIPEncoderConfig`)
- `src/models/composed_retrieval_model.py` (`load_phase4_checkpoint` — format-agnostic checkpoint loader)
- `src/retrieval/cosine_retrieval.py` (`retrieve_top_k`, `rank_of_target`)
- `src/retrieval/hard_negative_miner.py` (`load_duplicate_lookup`, `duplicate_exclusion_ids` — **reused, not duplicated**)
- `src/losses/contrastive.py` (`symmetric_contrastive_loss`)
- `scripts/evaluate_baseline.py` (the Recall@K scorer used since Phase 3)
- `data/processed/fashioniq_embeddings/{dress,shirt,toptee}_val_gallery.pt` — existing Phase 3/4 galleries, reused as-is for evaluation

## 4. What this does NOT modify

- `experiments/phase4/phase4_deliverable/checkpoints/epoch_2.pt` (read-only)
- `experiments/phase5/**`, `experiments/phase6/**` (read-only — Phase 6's duplicate lookup and gallery are *read*, nothing there is overwritten)
- `CP\ui-demo` — untouched entirely
- Any existing `src/` or `scripts/` file — everything above is a new file

## 5. Environment check first

```powershell
conda activate cir
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If this prints `True` for CUDA, use this environment for everything below (ignore any separate
`2.14.0+cpu` environment reported earlier — that was a different/unconfirmed environment).
If it prints `False`, see Section 9 (Colab) — do not attempt full training on CPU.

## 6. Run order — ALWAYS this order, do not skip the sanity test

### Step 1 — sanity test (seconds, catches wiring bugs before anything expensive)

```powershell
python -m py_compile src\utils\checkpoint_compat.py src\data\fashioniq_paths.py src\losses\semi_hard_loss.py src\retrieval\semi_hard_miner.py scripts\mine_phase8_hard_negatives.py scripts\train_phase8.py scripts\evaluate_phase8.py scripts\compare_phase8_phase4.py scripts\audit_phase8_semihard_negatives.py scripts\test_phase8_pipeline.py

python scripts\test_phase8_pipeline.py `
  --fiq-csv "data\processed\fashioniq\train_queries_dress.csv" `
  --phase4-checkpoint "experiments\phase4\phase4_deliverable\checkpoints\epoch_2.pt" `
  --combined-gallery "experiments\phase6\cache\combined_train_gallery.pt"
```

All checks should pass, especially **"Semi-hard guarantee holds (no negative >= positive)"** —
this is the core fix; if this fails, stop and report the output before proceeding.

### Step 2 — smoke test the full mine-then-train loop on a tiny slice

```powershell
python scripts\train_phase8.py `
  --init-checkpoint "experiments\phase4\phase4_deliverable\checkpoints\epoch_2.pt" `
  --fiq-train-csvs "data\processed\fashioniq\train_queries_dress.csv" "data\processed\fashioniq\train_queries_shirt.csv" "data\processed\fashioniq\train_queries_toptee.csv" `
  --categories dress shirt toptee `
  --combined-gallery "experiments\phase6\cache\combined_train_gallery.pt" `
  --rounds 1 --epochs-per-round 1 --limit 20 --run-name smoke_test
```

Expect this to finish in well under a minute given `--limit 20` per category.

### Step 3 — audit the mined negatives before trusting them

```powershell
python scripts\audit_phase8_semihard_negatives.py `
  --hard-negative-dir "experiments\phase8\smoke_test\hard_negatives" --round 0
```

**Check `Semi-hard GUARANTEE VIOLATIONS: 0`** for every category. This must read 0 — it is
the entire point of Phase 8. If it is not 0, do not proceed to full training; report the output.

### Step 4 — the real run (no `--limit`)

Given your 4GB VRAM / 7.7GB system RAM constraints established in Phase 5, start conservative:

```powershell
python scripts\train_phase8.py `
  --init-checkpoint "experiments\phase4\phase4_deliverable\checkpoints\epoch_2.pt" `
  --fiq-train-csvs "data\processed\fashioniq\train_queries_dress.csv" "data\processed\fashioniq\train_queries_shirt.csv" "data\processed\fashioniq\train_queries_toptee.csv" `
  --categories dress shirt toptee `
  --combined-gallery "experiments\phase6\cache\combined_train_gallery.pt" `
  --rounds 2 --epochs-per-round 1 --batch-size 2 --grad-accum 2 `
  --run-name phase8_semihard_v1
```

This is **2 rounds** (mine → train epoch → re-mine from the new checkpoint → train epoch again) —
a controlled, modest experiment per the explicit "do not run huge compute" instruction. Close
background apps first (same RAM constraint as Phase 5 — see `docs/phase_5_cross_domain_alignment.md`
Section 12 if you need the reminder of why).

**This will take a while** — each round mines ~16,854 queries (similar cost to Phase 6's mining,
plus gallery lookup) and then trains one epoch. Expect a similar order of magnitude to Phase 6's
original local run unless you move to Colab (Section 9).

### Step 5 — resume / check progress

Checkpoints save after each round at:
```
experiments\phase8\phase8_semihard_v1\checkpoints\round0.pt
experiments\phase8\phase8_semihard_v1\checkpoints\round1.pt
```
There is no `--resume-from` flag in this version — if interrupted mid-round, you'd restart that
round (mining is deterministic given the same seed, so re-mining an interrupted round reproduces
the same negatives). If you need resume support added, ask before relying on a long unattended run.

### Step 6 — evaluate the final checkpoint

```powershell
python scripts\evaluate_phase8.py `
  --phase8-checkpoint "experiments\phase8\phase8_semihard_v1\checkpoints\round1.pt" `
  --fiq-csv "data\processed\fashioniq\val_queries_dress.csv" `
  --gallery "data\processed\fashioniq_embeddings\dress_val_gallery.pt" `
  --category dress --split val

python scripts\evaluate_phase8.py `
  --phase8-checkpoint "experiments\phase8\phase8_semihard_v1\checkpoints\round1.pt" `
  --fiq-csv "data\processed\fashioniq\val_queries_shirt.csv" `
  --gallery "data\processed\fashioniq_embeddings\shirt_val_gallery.pt" `
  --category shirt --split val

python scripts\evaluate_phase8.py `
  --phase8-checkpoint "experiments\phase8\phase8_semihard_v1\checkpoints\round1.pt" `
  --fiq-csv "data\processed\fashioniq\val_queries_toptee.csv" `
  --gallery "data\processed\fashioniq_embeddings\toptee_val_gallery.pt" `
  --category toptee --split val
```

### Step 7 — score and compare against Phase 4

```powershell
python scripts\evaluate_baseline.py --ranked `
  "experiments\phase8\results\dress_val_phase8_ranked.json" `
  "experiments\phase8\results\shirt_val_phase8_ranked.json" `
  "experiments\phase8\results\toptee_val_phase8_ranked.json" `
  --out "experiments\phase8\results.json"

python scripts\compare_phase8_phase4.py --phase8-results "experiments\phase8\results.json"
```

This prints a side-by-side table against the **fixed, hardcoded Phase 4 numbers** (never
recomputed, so there's no risk of silent drift from the validated baseline).

## 7. How to verify the experiment is working (don't skip this)

Before believing any retrieval numbers, check the mining audit from Step 3/each round:
`Semi-hard GUARANTEE VIOLATIONS: 0`. This is the one number that tells you the fix is real,
independent of whether retrieval improves or not. A clean audit + a result that still doesn't
beat Phase 4 is a valid, reportable outcome (see `docs/phase_8_semi_hard_mining.md`) — a dirty
audit (violations > 0) means something is broken in the mining pipeline and no retrieval number
from that run should be trusted.

## 8. Comparing against Phase 4

Already built into `compare_phase8_phase4.py` (Step 7). Read the printed note: a "better" delta
only matters if it holds across R@1 **and** R@10, not just one K — and check
`docs/phase_8_semi_hard_mining.md`'s margin/quality-stats sections once filled in.

## 9. Running on Google Colab instead

1. Upload this entire merged repo (or just the new files on top of what you already uploaded for
   Phase 6) to Google Drive, same layout as Phase 6's Colab setup.
2. Every script here already calls `patch_path_compat()` (from `src/utils/checkpoint_compat.py`)
   before loading any checkpoint — this is the same Windows→Linux `WindowsPath`/`PosixPath` fix
   your Phase 6 Colab session needed, now built in rather than hand-patched per script.
3. Mount Drive, `cd` into the repo root, and run the exact same commands from Section 6 — paths
   are all relative to repo root via `REPO_ROOT = Path(__file__).resolve().parents[1]`, so nothing
   Colab-specific needs to change in the commands themselves.
4. Watch for session disconnects on long runs (this bit you during the Phase 6 Colab attempt) —
   consider running one round at a time rather than `--rounds 2` unattended, checking in between.

## 10. What to commit to GitHub

**Commit these:**
```
src/data/fashioniq_paths.py
src/losses/semi_hard_loss.py
src/retrieval/semi_hard_miner.py
src/utils/__init__.py
src/utils/checkpoint_compat.py
scripts/mine_phase8_hard_negatives.py
scripts/train_phase8.py
scripts/evaluate_phase8.py
scripts/compare_phase8_phase4.py
scripts/audit_phase8_semihard_negatives.py
scripts/test_phase8_pipeline.py
docs/phase_8_semi_hard_mining.md          (after you fill in real results)
experiments/phase8/results.json           (small, aggregated metrics only)
```

**Do NOT commit:**
```
experiments/phase8/*/checkpoints/*.pt                 (large model weights)
experiments/phase8/*/hard_negatives/*.csv              (large, regeneratable from code + seed)
experiments/phase8/results/*_ranked.json                (large per-query files)
```

Add to `.gitignore` (append, don't replace):
```
experiments/phase8/*/checkpoints/
experiments/phase8/*/hard_negatives/
experiments/phase8/results/*_ranked.json
```

Exact commands:
```powershell
Add-Content .gitignore "`nexperiments/phase8/*/checkpoints/`nexperiments/phase8/*/hard_negatives/`nexperiments/phase8/results/*_ranked.json"

git status
git add src/data/fashioniq_paths.py src/losses/semi_hard_loss.py src/retrieval/semi_hard_miner.py `
  src/utils/__init__.py src/utils/checkpoint_compat.py `
  scripts/mine_phase8_hard_negatives.py scripts/train_phase8.py scripts/evaluate_phase8.py `
  scripts/compare_phase8_phase4.py scripts/audit_phase8_semihard_negatives.py scripts/test_phase8_pipeline.py `
  docs/phase_8_semi_hard_mining.md experiments/phase8/results.json .gitignore
git status
git commit -m "Implement controlled semi-hard negative mining for Phase 8"
git push origin main
```

Double-check `git status` shows nothing from `experiments/phase8/*/checkpoints/` or `hard_negatives/`
before committing.
