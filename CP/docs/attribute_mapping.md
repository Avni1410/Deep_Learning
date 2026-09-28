# FashionIQ Attribute Mapping Specification

## 1. Purpose

This document defines how attribute information will be used in the
Attribute-Aware Cross-Domain Composed Image Retrieval project.

The purpose is to prevent assumptions about FashionIQ attributes and to
clearly distinguish:

1. attributes explicitly available in the dataset,
2. concepts inferred from modification captions, and
3. attributes that are reliable enough to be used as supervised model
   targets.

---

## 2. Source of Attribute Information

FashionIQ provides natural-language modification captions together with
visual attribute information.

The FashionIQ visual attribute vocabulary is organized around concepts
including:

- texture
- fabric
- shape
- part
- style

The project therefore does NOT assume that FashionIQ directly provides a
simple universal schema such as:

    color / pattern / material / shape

with complete labels for every image.

The final supervised attribute vocabulary must be determined from the
actual FashionIQ annotation files.

---

## 3. Caption-Derived Concepts

During Phase 1, modification captions were analyzed for recurring
attribute-related concepts.

Observed concept frequencies:

| Concept | Caption occurrences |
|---|---:|
| Color | 29,060 |
| Length | 14,584 |
| Sleeve | 14,475 |
| Pattern | 7,019 |
| Neckline | 3,110 |
| Style | 1,981 |
| Shape/Fit | 1,613 |
| Material | 240 |

These numbers represent occurrences in the natural-language modification
captions.

They are NOT:

- unique attribute values,
- image-level ground-truth labels,
- mutually exclusive categories,
- or the number of images possessing an attribute.

Therefore, these counts will be used for dataset analysis and vocabulary
planning, not directly as supervised targets.

---

## 4. Attribute Groups for the Proposed Model

### 4.1 Color

Status: PRIMARY CANDIDATE

Color is frequently expressed in FashionIQ modification captions and is
central to fine-grained product modification.

Examples of modification concepts may include:

- changing one color to another,
- making an item darker/lighter,
- modifying the dominant color.

Color should be used as a supervised attribute head only when a reliable
corresponding annotation/value can be established from the actual
FashionIQ annotation files.

---

### 4.2 Pattern / Texture

Status: PRIMARY CANDIDATE

Pattern and texture are important for distinguishing visually similar
products.

Examples include concepts such as:

- solid
- floral
- striped
- patterned
- textured

The exact supervised vocabulary must be derived from the available
FashionIQ annotations rather than manually assuming a fixed class list.

---

### 4.3 Material / Fabric

Status: PRIMARY CANDIDATE, SUBJECT TO LABEL COVERAGE

FashionIQ groups visual attributes under fabric-related concepts, making
material/fabric relevant to the research objective.

However, material-related concepts occurred much less frequently in the
caption analysis than color, length, sleeve, or pattern.

Therefore, material/fabric should only become a supervised attribute head
if sufficient reliable labels are available in the actual annotation
files.

If label coverage is insufficient, it will remain an analysis/evaluation
attribute rather than a mandatory training head.

---

### 4.4 Shape / Fit

Status: SECONDARY CANDIDATE

Shape/fit is relevant to fine-grained retrieval, but the project should
not assume that every FashionIQ image has a clean shape/fit label.

Potential concepts include:

- loose/fitted
- long/short silhouette
- shape-related modifications

Use as a supervised head only if the actual annotation structure provides
sufficient reliable labels.

---

### 4.5 Style / Detail

Status: SECONDARY CANDIDATE

Style-related concepts occur in the FashionIQ data and may contribute to
fine-grained discrimination.

However, "style" is broad and can overlap with several other visual
properties.

It should therefore not initially be treated as a single universal
classification head unless the available annotations provide a
well-defined vocabulary.

---

### 4.6 Length

Status: SECONDARY CANDIDATE

Length is frequent in modification captions.

It may be useful for:

- fine-grained evaluation,
- change/preserve analysis,
- and potentially supervised prediction.

However, the project will first prioritize the core attribute groups and
only add length when the annotation structure supports reliable labels.

---

### 4.7 Sleeve

Status: SECONDARY CANDIDATE

Sleeve-related modifications occur frequently in FashionIQ captions.

Sleeve information can be useful for fine-grained discrimination,
particularly for shirts and dresses.

It should be incorporated as a supervised attribute only after verifying
its annotation coverage and consistency.

---

### 4.8 Neckline

Status: SECONDARY / OPTIONAL

Neckline occurs in the caption analysis but is more category-specific.

It can be used for additional evaluation if reliable labels are available,
but it is not required for the first attribute-aware model.

---

## 5. Initial Attribute Priority

The initial model will prioritize:

1. Color
2. Pattern / Texture
3. Material / Fabric

These correspond to the project's central research objective of
understanding requested attribute changes while preserving relevant
unchanged properties.

Secondary attributes:

- Shape / Fit
- Style / Detail
- Length
- Sleeve
- Neckline

Secondary attributes will only be introduced when their annotation
coverage is sufficient.

---

## 6. Change vs Preserve Protocol

The central attribute-aware concept is:

    CHANGE requested attributes
    PRESERVE relevant unchanged attributes

For example:

Reference:
    red floral cotton shirt

Modification:
    "make it blue while keeping the floral pattern"

Desired behavior:

    Color:
        red -> blue
        CHANGE

    Pattern:
        floral -> floral
        PRESERVE

    Material:
        cotton -> cotton
        PRESERVE

The model should therefore not interpret the modification as an
instruction to reconstruct the entire product.

Instead, it should modify the requested semantic dimensions while
retaining relevant information from the reference image.

---

## 7. Attribute Supervision Rule

An attribute becomes a supervised model target only when:

1. its source annotation is clearly identified;
2. its label vocabulary is sufficiently consistent;
3. there is sufficient training coverage;
4. missing labels can be handled explicitly;
5. the labels can be mapped reproducibly from the original dataset.

Caption keyword matching alone is NOT considered sufficient ground-truth
supervision.

---

## 8. Missing Labels

Missing attribute annotations will not be treated as negative labels.

For an image/query where an attribute is unavailable:

    attribute loss = ignored / masked

rather than:

    attribute loss = negative class

This prevents missing annotations from introducing incorrect supervision.

---

## 9. Class Imbalance

Attribute distributions will be measured before training.

If an attribute has highly imbalanced classes, possible approaches include:

- class-weighted cross entropy,
- balanced sampling,
- minimum-frequency filtering,
- or excluding extremely sparse classes.

The selected approach will be documented after inspecting the actual
annotation distribution.

---

## 10. Relation to Caption Analysis

Caption analysis is used to identify what users commonly request.

Annotation analysis is used to determine what can be reliably supervised.

Therefore:

    Caption analysis
          ↓
    Understand modification concepts

    Actual annotations
          ↓
    Determine supervised labels

These two sources must not be treated as interchangeable.

---

## 11. DeepFashion2 Attribute Usage

DeepFashion2 is primarily used for natural consumer-to-commercial
cross-domain representation alignment.

It is NOT being treated as a second FashionIQ-style CIR dataset.

DeepFashion2 attributes/metadata may be used for:

- category-aware analysis,
- domain alignment,
- filtering,
- auxiliary analysis,
- and hard-negative experiments.

DeepFashion2 will not be forced into the FashionIQ modification-caption
task through large-scale synthetic caption generation.

---

## 12. Controlled Transformations

Controlled transformations are explicitly DEFERRED to Phase 8.

They will not be part of the Phase 1 dataset preparation pipeline.

The planned experiment will transform the FashionIQ reference image while
keeping:

- the original modification text unchanged,
- the original target image unchanged,
- and the original train/validation/test split unchanged.

Possible transformations include:

- brightness changes,
- contrast changes,
- color-temperature changes,
- mild shadows,
- JPEG compression,
- mild blur,
- resolution degradation,
- mild crop/scale,
- small rotation,
- mild perspective changes.

Transformations must not intentionally modify the underlying product
attributes.

This experiment will be treated as a controlled robustness evaluation,
not as a naturally occurring cross-domain dataset.

---

## 13. Current Decision

For the first attribute-aware implementation:

    Primary:
        Color
        Pattern / Texture
        Material / Fabric

    Secondary:
        Shape / Fit
        Style / Detail
        Length
        Sleeve
        Neckline

The final supervised heads will be selected only after inspecting the
actual FashionIQ annotation vocabulary and coverage.

No attribute vocabulary will be invented merely to make the model
architecture appear more complete.

---

## 14. Reproducibility Requirement

Any attribute vocabulary used in training must be reproducible from the
original FashionIQ files.

The project should record:

- source annotation file,
- attribute/concept name,
- class vocabulary,
- number of valid samples,
- missing-label count,
- class distribution,
- mapping rules,
- and any filtering thresholds.

This information will later be used to produce the experimental tables
and ablation results.

---

## 15. Phase 1 Boundary

Phase 1 is considered complete when:

- FashionIQ is prepared and validated;
- FashionIQ query manifests are available;
- DeepFashion2 domain-alignment subset is available;
- both dataset classes are implemented;
- both DataLoaders are verified;
- attribute interpretation is documented;
- controlled transformations are explicitly deferred to Phase 8.

The next implementation phase is Phase 2:

    CLIP / ViT visual and text representation