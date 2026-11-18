"""Load the Hugging Face DistilRoBERTa emotion classifier with PyTorch."""

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.settings import TEXT_MODEL_ID, TEXT_MODEL_REVISION


@st.cache_resource(show_spinner="Loading text emotion model…")
def load_text_model():
    """Return the tokenizer, model, and inference device.

    The model is downloaded from Hugging Face on first use and cached by
    Streamlit for subsequent reruns.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_MODEL_ID,
        revision=TEXT_MODEL_REVISION,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        TEXT_MODEL_ID,
        revision=TEXT_MODEL_REVISION,
    )
    model.to(device)
    model.eval()

    return tokenizer, model, device
