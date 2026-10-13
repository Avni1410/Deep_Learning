from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from src.models.attribute_aware_cir import AttributeAwareCIRModel
from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig
from src.retrieval.cosine_retrieval import retrieve_top_k


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

CHECKPOINT = (
    ROOT
    / "experiments"
    / "phase4"
    / "phase4_deliverable"
    / "checkpoints"
    / "epoch_2.pt"
)

GALLERY_DIR = (
    ROOT
    / "data"
    / "processed"
    / "fashioniq_embeddings"
)

IMAGE_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "fashioniq"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Attribute-Aware Product Retrieval",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    .result-card {
        padding: 0.6rem;
        border-radius: 10px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
    }

    .rank-label {
        font-size: 1.05rem;
        font-weight: 600;
    }

    .score-label {
        color: #666;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    ckpt = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    clip = CLIPEncoder(
        CLIPEncoderConfig(
            model_name="openai/clip-vit-base-patch32",
            device=device,
        )
    )

    model = AttributeAwareCIRModel(
        clip,
        attribute_vocab=ckpt.get("attribute_vocab"),
    ).to(clip.device)

    model.cross_attn.load_state_dict(
        ckpt["trainable_state"]["cross_attn"]
    )

    model.fusion.load_state_dict(
        ckpt["trainable_state"]["fusion"]
    )

    model.attribute_heads.load_state_dict(
        ckpt["trainable_state"]["attribute_heads"]
    )

    model.eval()

    return model, device


# ============================================================
# GALLERY LOADING
# ============================================================

@st.cache_data
def load_gallery(category: str):

    gallery_path = (
        GALLERY_DIR
        / f"{category}_val_gallery.pt"
    )

    gallery = torch.load(
        gallery_path,
        map_location="cpu",
        weights_only=False,
    )

    return (
        gallery["ids"],
        gallery["embeddings"],
    )


# ============================================================
# IMAGE SEARCH
# ============================================================

def find_image(image_id: str) -> Path | None:

    jpg_candidates = list(
        IMAGE_ROOT.rglob(
            f"{image_id}.jpg"
        )
    )

    if jpg_candidates:
        return jpg_candidates[0]

    png_candidates = list(
        IMAGE_ROOT.rglob(
            f"{image_id}.png"
        )
    )

    if png_candidates:
        return png_candidates[0]

    return None


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    model,
    reference_image,
    modification_text,
    gallery_embeddings,
    gallery_ids,
    top_k,
):

    with torch.no_grad():

        output = model.encode_query(
            [reference_image],
            [modification_text],
            [""],
        )

    query_embedding = (
        output["query_embedding"]
        .squeeze(0)
        .cpu()
    )

    results = retrieve_top_k(
        query_embedding,
        gallery_embeddings,
        gallery_ids,
        k=top_k,
    )

    return results


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Attribute-Aware Composed Image Retrieval'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Fine-Grained Product Search using Reference Image + Modification Text'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Retrieval Settings")

    category = st.selectbox(
        "Product Category",
        [
            "dress",
            "shirt",
            "toptee",
        ],
    )

    top_k = st.slider(
        "Number of Results",
        min_value=1,
        max_value=10,
        value=5,
    )

    st.divider()

    st.subheader("Model")

    st.write(
        "Attribute-Aware CIR"
    )

    st.write(
        "Backbone: CLIP ViT-B/32"
    )

    device_text = (
        "CUDA GPU"
        if torch.cuda.is_available()
        else "CPU"
    )

    st.write(
        f"Device: {device_text}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner("Loading retrieval model..."):

    model, device = load_model()


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '1. Query Input'
    '</div>',
    unsafe_allow_html=True,
)

left, right = st.columns(
    [1, 1]
)


with left:

    uploaded_file = st.file_uploader(
        "Upload a reference product image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
    )


with right:

    modification_text = st.text_area(
        "Describe the desired modification",
        placeholder=(
            "Example: Make it black with a more formal style"
        ),
        height=150,
    )


# ============================================================
# REFERENCE IMAGE PREVIEW
# ============================================================

reference_image = None

if uploaded_file is not None:

    reference_image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.markdown(
        '<div class="section-title">'
        'Reference Image'
        '</div>',
        unsafe_allow_html=True,
    )

    st.image(
        reference_image,
        width=300,
    )


# ============================================================
# RETRIEVE BUTTON
# ============================================================

st.divider()

retrieve_button = st.button(
    "🔎 Retrieve Products",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RETRIEVAL
# ============================================================

if retrieve_button:

    if reference_image is None:

        st.warning(
            "Please upload a reference image."
        )

        st.stop()

    if not modification_text.strip():

        st.warning(
            "Please enter a modification instruction."
        )

        st.stop()

    with st.spinner(
        "Searching the product catalogue..."
    ):

        gallery_ids, gallery_embeddings = (
            load_gallery(category)
        )

        results = retrieve(
            model=model,
            reference_image=reference_image,
            modification_text=modification_text,
            gallery_embeddings=gallery_embeddings,
            gallery_ids=gallery_ids,
            top_k=top_k,
        )

    # ========================================================
    # RESULTS HEADER
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '2. Retrieved Products'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(
        f"Showing top {len(results.ranked_ids)} "
        f"results for **{category}** products."
    )

    # ========================================================
    # RESULT GRID
    # ========================================================

    columns = st.columns(
        min(top_k, 5)
    )

    for i, image_id in enumerate(
        results.ranked_ids
    ):

        column = columns[
            i % len(columns)
        ]

        image_path = find_image(
            image_id
        )

        with column:

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True,
            )

            if image_path is not None:

                image = Image.open(
                    image_path
                ).convert("RGB")

                st.image(
                    image,
                    use_container_width=True,
                )

            else:

                st.warning(
                    "Image unavailable"
                )

            score = float(
                results.scores[i]
            )

            st.markdown(
                f'<div class="rank-label">'
                f'Rank {i + 1}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="score-label">'
                f'Similarity: {score:.4f}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.caption(
                image_id
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True,
            )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

with st.expander(
    "About this system"
):

    st.write(
        """
        This research prototype performs composed image retrieval
        for fine-grained product search.

        The query consists of:

        • A reference product image

        • Natural-language modification instructions

        The system encodes the composed query and retrieves the
        most similar products from a precomputed FashionIQ gallery.

        The retrieval model uses CLIP visual and text representations
        together with attribute-aware cross-attention and feature
        fusion.
        """
    )

    st.write(
        f"Current device: **{device}**"
    )