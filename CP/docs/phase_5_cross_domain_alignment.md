# Phase 5 — Cross-Domain Representation Alignment

## 1. Motivation

Phase 4 trained text-to-image cross-attention and fusion to compose a FashionIQ
reference image and a modification into a query embedding, using only frozen
CLIP ViT-B/16 features. That representation was learned entirely from FashionIQ's
own reference/target image pairs, which come from a relatively narrow visual
domain (catalogue-style product photography). Real-world composed retrieval may
need to work across more varied visual conditions (e.g. a consumer photo as the
reference, matched against commercial catalogue targets). Phase 5 asks whether
making the underlying visual representation more consistent across such domains
helps or hurts the retrieval task Phase 4 established.

## 2. Research gap addressed

This project does not claim cross-domain representation alignment is a new
research problem — prior work has studied domain generalization and alignment
extensively. The contribution here is the specific combination of:

    composed image retrieval
          +
    fine-grained product retrieval
          +
    cross-domain representation alignment
          +
    attribute-aware (text-conditioned) multimodal composition

evaluated together on FashionIQ, rather than any one of these in isolation.
Phase 5 is described as "cross-domain representation alignment for the proposed
composed image retrieval framework," not as a novel cross-domain method.

## 3. Role of DeepFashion2

DeepFashion2 provides a 5,000-pair, category-balanced subset of consumer
("user") and commercial ("shop") image pairs
(`data/processed/deepfashion2/domain_alignment_pairs_5000.csv`), used purely as
an auxiliary signal: encouraging `f(user_image) ≈ f(shop_image)` for the same
underlying item, while FashionIQ remains the primary training and evaluation
signal for the actual retrieval task.

## 4. Why DeepFashion2 is not used as CIR supervision

DeepFashion2 is used as an auxiliary source for cross-domain visual
representation alignment, while FashionIQ remains the primary dataset for
composed image retrieval supervision and evaluation. DeepFashion2 user/shop
pairs are never treated as (reference, modification, target) CIR triples, and
no synthetic captions were generated to force it into that role.

## 5. Mathematical formulation

Overall objective:

    L_total = λ_CIR · L_CIR + λ_domain · L_domain

Initial values used: `λ_CIR = 1.0` (implicit — `L_CIR` is added directly via its
own `backward()` call), `λ_domain = 0.1`.

## 6. Pair alignment loss

For a batch of B matched user/shop embeddings (both L2-normalized):

    L_pair = mean_i( 1 - cos(f_user_i, f_shop_i) )

Implemented in `src/losses/domain_alignment.py::paired_cosine_alignment_loss`.

## 7. CORAL loss (as corrected — see Section 16)

    C_user = cov(f_user_centered),  C_shop = cov(f_shop_centered)
    L_CORAL = ||C_user - C_shop||_F^2 / d

where `d = 512` is the feature dimension. The originally implemented version
used the Deep-CORAL paper's `/(4d^2)` normalizer; this was corrected to `/d`
after diagnosis — see Section 16 for the full investigation. Batches with
fewer than 2 samples per side return 0 rather than dividing by zero.

## 8. Combined loss

    L_domain = α · L_pair + β · L_CORAL,  α = 1.0, β = 0.1

## 9. Training strategy

CLIP ViT-B/16 remains frozen throughout (same frozen backbone as Phases 2–4).
`CrossDomainCIRModel` (`src/models/composed_retrieval_model.py`) warm-starts
`cross_attn` and `fusion` from the Phase 4 checkpoint
(`experiments/phase4/phase4_deliverable/checkpoints/epoch_2.pt`, lean
`trainable_state` format, confirmed by direct inspection rather than assumed)
and adds one new trainable module, `visual_adapter`
(512→512, GELU, dropout, 512→512, residual, L2-normalize), applied identically
to: the reference image embedding feeding fusion, the target image embedding
used for `L_CIR`, and the DeepFashion2 user/shop embeddings used for
`L_domain`. This shared adapter is what gives `L_domain`'s gradient an actual
path to parameters that also shape retrieval — computing the domain loss
directly on frozen CLIP output (a literal reading of the original spec) would
have produced zero gradient to any trainable parameter and zero influence on
`q`/`z`.

Optimization uses alternating batches with gradients accumulated into one
optimizer step per iteration (no `retain_graph` needed — the two forward
passes are independent graphs):

## Limitations (addendum)

The initial CORAL implementation used the original Deep-CORAL paper's /(4d^2)
normalizer, which assumes unnormalized deep-activation scale. Diagnosed via
inference-only testing on real DeepFashion2 features (B=2..64) using the
trained exp3_coral checkpoint: raw covariance differences were in a
physically plausible O(0.01-0.4) range, but /(4d^2) crushed this to ~1e-7
regardless of batch size - meaning the Combined experiment (exp4_combined)
was numerically indistinguishable from Pair-only (exp2_pair) by construction,
not due to a batch-size artifact. The normalizer was corrected to /d.
exp3_coral and exp4_combined results reported here used the UNCORRECTED
normalizer and should be treated as a null/inconclusive test of CORAL's
contribution, not evidence that CORAL doesn't help. A corrected rerun is
noted as pending future work.



Given your time constraint, don't chase this further right now. Two honest options, pick based on how much you care about a clean CORAL-only signal before Phase 6:

Fast path (recommended): leave beta=0.1 as the documented initial value (matches your original spec's §33 "start with these values, don't sweep yet"), note in the doc that CORAL's absolute scale remains small relative to pair-cosine alignment even after the normalizer fix, and move to Phase 6 now. This is scientifically honest — not every component needs to dominate equally.
If you want a real signal later: when you do rerun, bump --beta 10 (not now) so 0.1 × 10 × 3.8e-5 ≈ 4e-5 becomes at least directionally comparable — still a guess, not a sweep, consistent with "don't sweep yet."