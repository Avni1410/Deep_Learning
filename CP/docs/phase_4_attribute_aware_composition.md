# Phase 4 - Attribute-Aware Multimodal Composition

## 1. Research motivation
Phase 3's `q = normalize(r + t)` treats the reference image and modification text as
undifferentiated global vectors, unable to apply a targeted change while preserving
unrelated attributes.

## 2. Phase 3 limitation
Demonstrated qualitatively: large attribute changes were often ignored in favor of
retrieving reference-like items (see docs/phase_3_baseline.md examples).

## 3-5. Architecture, cross-attention, fusion
See src/models/attribute_aware_cir.py. Text tokens (concatenated caption_1 + caption_2)
attend to image patches:
  Q = T W_Q, K = V_p W_K, V = V_p W_V
  H = softmax(QK^T / sqrt(d)) V
Fusion: x = [g; h; g*h; g-h] -> MLP(2048->1024->512) -> normalize -> q

## 6. Attribute supervision — DEFERRED

Reliable image-level attribute annotations were not available in the downloaded
FashionIQ release (captions/, image_splits/, asin2url.*.txt, and product images only;
no attributes.csv / attribute_labels.json or equivalent). The earlier caption-derived
concept counts (color, length, sleeve, pattern, neckline, style, shape/fit, material)
are keyword-match statistics over natural-language captions, not per-image ground
truth, and per docs/attribute_mapping.md Section 7, caption-keyword matching alone is
explicitly not accepted as sufficient supervision.

Supervised attribute heads and the attribute loss (L_attr) are therefore DEFERRED,
not implemented with a substitute labeling scheme. Phase 4's attribute-awareness
comes entirely from the model learning fine-grained visual-text interaction via
cross-attention over image patches, not from explicit attribute classification.


## 7. Loss functions
L_CIR = 0.5*(L_q->t + L_t->q), symmetric contrastive, temperature=0.07 (initial).
L_attr: PENDING (see above). L_total = lambda_CIR * L_CIR + lambda_attr * L_attr,
with lambda_attr's term currently always 0 (ablation B) until attribute labels exist.

## 8. Training strategy
CLIP ViT-B/16 frozen throughout. Only cross-attention, fusion (+ attribute heads,
once added) are trained. AdamW, lr=1e-4, weight_decay=1e-4, mixed precision on CUDA.

## 9. Evaluation protocol
Identical to Phase 3: same per-category galleries (data/processed/fashioniq_embeddings/),
same Recall@1/5/10/50 implementation (src/evaluation/metrics.py).

## 10. Ablation protocol

A = Phase 3 baseline (q = normalize(r + t), no training)
B = Phase 4 deliverable: cross-attention + fusion, trained with contrastive loss only
    (this IS Phase 4's actual output, not an intermediate step toward a heavier "C")

Supervised attribute-head experiment (previously "ablation C") is deferred; see
Section 6 and Section 12.

## 11. Results
## 11. Results

Trained: 3 epochs, all three categories combined (dress+shirt+toptee train splits),
batch_size=4, lr=1e-4, AdamW, contrastive loss only (temperature=0.07), CLIP frozen.
Evaluated on the same FashionIQ validation galleries as Phase 3.

| Method                        | R@1    | R@5    | R@10   | R@50   |
|--------------------------------|--------|--------|--------|--------|
| Phase 3 baseline (overall)     | 0.0400 | 0.1505 | 0.2078 | 0.3847 |
| Phase 4 cross-attn+fusion (overall) | 0.0570 | 0.1722 | 0.2547 | 0.5251 |

Category breakdown (Phase 4):
| Category | R@1    | R@5    | R@10   | R@50   |
|----------|--------|--------|--------|--------|
| Dress    | 0.0398 | 0.1306 | 0.2047 | 0.4643 |
| Shirt    | 0.0567 | 0.1664 | 0.2427 | 0.5219 |
| Toptee   | 0.0745 | 0.2196 | 0.3171 | 0.5892 |

Phase 4 improves over Phase 3 at every K, in every category, on identical galleries
and evaluation protocol. This supports the core Phase 4 hypothesis: text-conditioned
attention over image patches, combined with learned fusion, captures composed-query
information that simple vector addition cannot.


## 12. Limitations

1. CLIP backbone frozen (no fine-tuning yet).
2. Supervised attribute prediction is deferred: no reliable image-level attribute
   annotation source exists in the downloaded FashionIQ release, and caption-keyword
   matching was deliberately not substituted as ground truth (see Section 6). This is
   a documented research limitation, not an oversight.
3. Caption combination concatenates two independently-encoded token sequences;
   caption_2's tokens carry position embeddings as if starting a new sentence
   (documented simplification).
4. In-batch negatives may include visually related, near-valid products (false
   negatives) - not addressed until Phase 6.
5. Cross-attention does not guarantee explicit attribute localization; any attention
   visualization would be qualitative evidence only, not proof of attribute grounding.
6. Cross-domain alignment (Phase 5) and hard-negative mining (Phase 6) not yet applied.