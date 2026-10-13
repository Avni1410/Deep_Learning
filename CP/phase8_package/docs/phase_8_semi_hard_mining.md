# Phase 8 — Controlled Semi-Hard Negative Mining

Fill in the TBD sections after running; nothing in this template is a real result yet.

## 1. Motivation

Phase 6 implemented offline hard-negative mining and fine-grained contrastive learning,
but a 1-epoch experiment underperformed the Phase 4 baseline at every Recall@K. An audit
of the mined negatives (`scripts/audit_phase6_hard_negatives.py`) revealed the likely
cause: `negative_1` similarity exceeded `positive_similarity` in 95.7–96.9% of queries
across all three categories — meaning Phase 6 mined the model's own top-ranking *errors*,
not fair near-miss hard negatives. Phase 8 is a controlled, minimal correction targeting
this specific, diagnosed mechanism.

## 2. What Phase 6 did (for reference)

- Mining: top-K similarity retrieval from a combined gallery, excluding the true target,
  reference/candidate, and exact-image duplicate groups (duplicate handling was itself a
  later correction within Phase 6, confirmed structurally leak-free by audit).
- Loss: margin-based hinge loss, `max(0, margin - pos_sim + neg_sim)`, margin=0.10,
  `total_loss = CIR_loss + 0.5 * hard_negative_loss`.
- Training: 1 epoch, batch_size=2, grad_accum=2, from the Phase 4 epoch-2 checkpoint.
- Result: R@1 -0.39pp, R@5 -0.36pp, R@10 -0.70pp, R@50 -1.65pp vs. Phase 4 (uniform
  regression across all K — consistent with destabilizing gradients from
  already-mis-ranked "negatives", not a targeted fine-grained-discrimination effect).

## 3. Why Phase 6 likely failed

A candidate that already outranks the true positive is not a "hard negative" in the
metric-learning sense (a near-miss, slightly-below-positive confusable item) — it is the
model's own ranking mistake. Training a margin loss against such candidates, from a
still-modest base encoder (Phase 4 R@1 ≈ 5.7%), produces large, one-directional gradients
on nearly every query simultaneously. With only ~9M trainable parameters (frozen CLIP) and
one epoch at lr=1e-4, this plausibly perturbed the shared cross-attention/fusion weights
enough to hurt retrieval broadly, without enough signal/time to actually fix the targeted
ranking errors. This is a documented failure mode in metric learning (naive hard-negative
mining from an undertrained encoder yields "impossible" triplets — Schroff et al., FaceNet,
2015), not a sign that hard-negative mining is inherently ineffective for this task.

## 4. Phase 8 correction

1. **Semi-hard mining**: a candidate is only eligible as a negative if
   `similarity(q, candidate) < similarity(q, positive)` at mining time. Enforced by
   construction in `src/retrieval/semi_hard_miner.py`; verified empirically via
   `scripts/audit_phase8_semihard_negatives.py` (see Section 6 for the real numbers).
2. **Iterative re-mining**: negatives are re-mined after each training round from the
   just-updated checkpoint, not mined once from Phase 4 and reused for all training.
3. **Reduced, warmed-up hard-negative weight**: `lambda_hn` ramps 0 → 0.2 (down from
   Phase 6's flat 0.5) over the first `--warmup-steps` of each round.
4. Loss formulation: InfoNCE-style (`src/losses/semi_hard_loss.py`), not the margin hinge
   loss Phase 6 used — a deliberate choice, since the InfoNCE form naturally handles
   variable negative counts per query (semi-hard mining can yield fewer than N_hard
   eligible negatives for some queries — documented as `shortfall`, never padded with an
   invalid negative) more cleanly than a margin loss would.

## 5. Why this should have a better chance

This directly removes the one concretely diagnosed mechanism behind Phase 6's regression:
by construction, Phase 8 never trains against a negative the model already prefers over
the truth. It does not change the architecture, the dataset, or the evaluation protocol —
only the mining criterion, the re-mining schedule, and the loss weight/warm-up.

## 6. Mining audit results (fill in from `audit_phase8_semihard_negatives.py` output)

| Category | Queries | Shortfall queries | Guarantee violations | Mean margin (pos - neg1) |
|---|---:|---:|---:|---:|
| Dress | TBD | TBD | TBD (must be 0) | TBD |
| Shirt | TBD | TBD | TBD (must be 0) | TBD |
| Toptee | TBD | TBD | TBD (must be 0) | TBD |

Compare the mean margin directly against Phase 6's measured -0.0759 (dress), -0.0730
(shirt), -0.0759 (toptee) — this should now be positive.

## 7. Training configuration

| Parameter | Value |
|---|---|
| Init checkpoint | `experiments/phase4/phase4_deliverable/checkpoints/epoch_2.pt` |
| Rounds | TBD |
| Epochs per round | TBD |
| Batch size / grad accum | TBD |
| Learning rate | TBD |
| lambda_hn target / warmup steps | TBD |
| Strategy | TBD (similarity / category_similarity) |
| n_hard / k_retrieval | TBD |
| Seed | TBD |

## 8. Results

| Method | R@1 | R@5 | R@10 | R@50 |
|---|---:|---:|---:|---:|
| Phase 4 baseline (overall) | 0.0570 | 0.1722 | 0.2547 | 0.5251 |
| Phase 6 (1 epoch, overall) | ~0.0531 | ~0.1686 | ~0.2477 | ~0.5086 |
| Phase 8 (overall) | TBD | TBD | TBD | TBD |

Category-wise: TBD (fill from `compare_phase8_phase4.py` output).

## 9. Interpretation

TBD — write honestly from the actual numbers. Do not declare success unless the
improvement holds consistently across R@1 and R@10 (not just one K favorably).

## 10. Limitations

1. Only a small number of rounds/epochs tested (compute-constrained), same caveat as
   every prior phase on this hardware.
2. Semi-hard filtering necessarily yields fewer than n_hard negatives for some queries
   (see Section 6's shortfall numbers) — a documented trade-off of the safety guarantee,
   not a bug.
3. No hyperparameter sweep on lambda_hn_target, warmup_steps, or rounds — single
   configuration, per the explicit instruction not to tune post-hoc to chase better numbers.
4. If results are still negative or inconclusive, that remains a valid, reportable outcome.

## 11. What is NOT part of Phase 8

- Phase 5's cross-domain alignment (separately documented, not combined here)
- Attribute classification (deferred since Phase 4 — no reliable annotation source)
- Any UI/deployment changes
