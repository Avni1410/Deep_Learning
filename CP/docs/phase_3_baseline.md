# Phase 3 - Baseline Composed Image Retrieval

Results below are placeholders until the experiment runs; the real numbers come from
`experiments/baseline/results.json`, never typed in by hand.

1. **Purpose.** Establish a zero-shot, pretrained-CLIP-only baseline (no learned
   parameters, no fusion, no attention) that later phases (4-6) are measured against.
2. **Dataset.** FashionIQ, categories dress/shirt/toptee, validation split.
3. **Model.** `openai/clip-vit-base-patch16` (frozen, from Phase 2).
4. **Query composition.**
   r = normalize(CLIP_image(reference))
   t1 = normalize(CLIP_text(caption_1)); t2 = normalize(CLIP_text(caption_2))
   t = normalize((t1 + t2) / 2)
   q = normalize(r + t)
5. **Gallery.** Unique target images per category+split (reference/candidate images excluded).
6. **Similarity.** s_i = q . z_i (cosine, since both are L2-normalized).
7. **Metrics.** Recall@1, @5, @10, @50, overall and per category.
8. **Validation protocol.** 10-query/category sanity run (`--limit`) before the full split.
9.9. **Category-wise results.**
   | Category | Gallery | R@1 | R@5 | R@10 | R@50 |
   |---|---|---|---|---|---|
   | Dress  | 1861 | 0.0253 | 0.1188 | 0.1709 | 0.3471 |
   | Shirt  | 1941 | 0.0520 | 0.1664 | 0.2179 | 0.3905 |
   | Toptee | 1867 | 0.0423 | 0.1655 | 0.2341 | 0.4162 |
10. **Overall results.** R@1 = 0.0400, R@5 = 0.1505, R@10 = 0.2078, R@50 = 0.3847 (n = 5,669 queries).
11. **Qualitative examples.** Not yet generated.
12. **Limitations.** No token-level image-text interaction; no explicit attribute
    modeling; no cross-domain alignment; no hard negatives; no learned composition.
    These motivate Phases 4-6. This baseline is not the proposed method.