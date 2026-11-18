"""Load the Hugging Face ViT image-emotion classifier with PyTorch."""

import streamlit as st
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from config.settings import IMAGE_MODEL_ID, IMAGE_MODEL_REVISION


@st.cache_resource(show_spinner="Loading image emotion model…")
def load_image_model():
    """Return the image processor, model, and inference device.

    The model is downloaded from Hugging Face on first use and cached by
    Streamlit for subsequent reruns.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(
        IMAGE_MODEL_ID,
        revision=IMAGE_MODEL_REVISION,
    )
    model = AutoModelForImageClassification.from_pretrained(
        IMAGE_MODEL_ID,
        revision=IMAGE_MODEL_REVISION,
    )
    model.to(device)
    model.eval()

    return processor, model, device
