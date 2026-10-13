# Attribute-Aware Composed Image Retrieval for Fine-Grained Product Search

## Overview

This project focuses on **composed image retrieval (CIR)** for fine-grained product search. Instead of searching with only an image or a text query, users can provide a reference product image along with a natural-language instruction describing the changes they want.

For example, a user could provide an image of a shirt and request a similar product with a different colour or pattern. The system combines the visual information from the reference image with the modification text to create a composed representation, which is then used to retrieve and rank relevant products from a catalogue.

The project explores cross-attention-based feature composition and semi-hard negative mining to improve retrieval performance across different product categories.

## Objectives

- Develop a retrieval system that combines reference images with natural-language modifications.
- Extract meaningful visual and textual representations using pretrained encoders.
- Use cross-attention to connect text instructions with relevant image features.
- Train a composition network using contrastive learning.
- Investigate semi-hard negative mining to improve the model's ability to distinguish relevant products from challenging alternatives.
- Evaluate retrieval performance using standard Recall@K metrics.

## Key Features

- **Composed image retrieval:** Combines an input image and modification text for product search.
- **Visual feature extraction:** Uses CLIP ViT-B/16 to represent global image features and local image patches.
- **Text-conditioned visual composition:** Applies cross-attention to relate textual instructions to visual features.
- **Feature fusion:** Combines global image information and text-conditioned visual representations into a unified search embedding.
- **Contrastive learning:** Trains the composition network to improve the similarity between composed queries and their target products.
- **Semi-hard negative mining:** Explores challenging negative examples during training.
- **Category-wise evaluation:** Measures retrieval performance across Dress, Shirt, and Toptee categories.
- **Web interface:** Provides a Flask-based interface for interacting with the retrieval system.

## Methodology

The project follows a staged development process.

1. **Baseline retrieval:** Establishes a CLIP-based composed retrieval baseline using image and text representations.
2. **Attribute-aware composition:** Introduces cross-attention and feature fusion to generate representations conditioned on the requested modification.
3. **Contrastive training:** Fine-tunes the composition network using a symmetric contrastive objective.
4. **Negative mining experiments:** Investigates hard-negative and semi-hard-negative mining strategies to improve discrimination between target and non-target products.
5. **Retrieval evaluation:** Ranks catalogue products by similarity to the composed query and measures performance using Recall@K.

In this project, attribute awareness refers to text-conditioned visual feature learning. The model does not rely on supervised attribute-classification heads.

## Dataset

The primary dataset is **FashionIQ**, which provides reference product images and natural-language modification captions.

The experiments focus on three categories:

- Dress
- Shirt
- Toptee

The dataset is prepared for composed retrieval using reference images, modification text, and target products. Image availability and data-quality checks are performed as part of the preparation process.

**Evaluation note:** The reported retrieval experiments use the validation split. The official test split has not been evaluated.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning framework | PyTorch |
| Visual encoder | CLIP ViT-B/16 |
| Text encoder | CLIP Text Transformer |
| Feature processing | PyTorch, NumPy |
| Data handling | Pandas |
| Training objective | Symmetric contrastive loss |
| Negative mining | Custom semi-hard negative mining |
| Evaluation | Recall@1, Recall@5, Recall@10, Recall@50 |
| Web interface | Flask |
| Version control | Git and GitHub |

## Project Structure

```text
CP/
├── app.py
├── calculate_recall.py
├── .gitignore
├── docs/
│   └── attribute_mapping.md
├── src/
│   ├── losses/
│   │   ├── fine_grained.py
│   │   └── semi_hard_loss.py
│   ├── retrieval/
│   │   └── semi_hard_miner.py
│   └── utils/
├── scripts/
├── experiments/
├── phase8_package/
│   ├── README_INTEGRATION.md
│   ├── docs/
│   ├── scripts/
│   └── src/
└── ui-demo/
```

*Note: The structure above highlights the main project components; some directories may contain additional files.*

## Evaluation Metrics

Retrieval performance is measured using Recall@K:

- **Recall@1:** Whether the target product appears at rank 1.
- **Recall@5:** Whether the target appears among the top 5 results.
- **Recall@10:** Whether the target appears among the top 10 results.
- **Recall@50:** Whether the target appears among the top 50 results.

The experiments compare the baseline, cross-attention-based composition model, and negative-mining approaches. Comparisons should use the same evaluation population and protocol for a fair assessment.

## Running the Project

The repository contains the model components, training and evaluation scripts, and a Flask application entry point.

Before running the project:

1. Set up a compatible Python environment.
2. Install the dependencies required by the selected scripts.
3. Configure the FashionIQ dataset paths.
4. Ensure the required model checkpoint and catalogue embeddings are available.
5. Follow the instructions in `phase8_package/README_INTEGRATION.md` for the Phase 8 components.

The exact commands and paths may depend on your local dataset and checkpoint locations.

## Limitations

- Evaluation is currently limited to the validation split.
- The dataset release used in this project does not provide authoritative image-level attribute labels for supervised attribute classification.
- Cross-domain alignment and negative-mining configurations require careful, matched evaluation before drawing definitive conclusions.
- The current retrieval pipeline relies on available catalogue images and precomputed representations where configured.

## Future Scope

Possible extensions include:

- **Real-time camera-based product search:** Capture a product using a device camera and retrieve visually similar catalogue items.
- **Interactive retrieval:** Use user feedback on retrieved products to refine subsequent results.
- **Dynamic query modification:** Allow users to adjust natural-language instructions without restarting the search process.
- **Hybrid retrieval and reranking:** Combine visual similarity with product attributes, price, style, and availability.
- **Dynamic catalogue updates:** Refresh product representations as new items are added or existing items change.
- **Personalized ranking:** Adapt results to user preferences and search history.

## Repository

GitHub: [Avni1410/Deep_Learning](https://github.com/Avni1410/Deep_Learning)

## Acknowledgements

This project uses the FashionIQ dataset and pretrained CLIP representations for research and experimentation in composed image retrieval.
