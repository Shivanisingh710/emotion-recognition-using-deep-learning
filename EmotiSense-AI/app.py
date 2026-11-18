"""Streamlit entry point for EmotiSense AI."""

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, UnidentifiedImageError

from config.settings import APP_NAME, IMAGE_MODEL_ID, TEXT_MODEL_ID
from services.image_predictor import predict_image
from services.text_predictor import predict_text


st.set_page_config(page_title=APP_NAME, page_icon="🎭", layout="wide")
st.title("🎭 EmotiSense AI")
st.caption("Multimodal emotion classification from facial images and English text")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Image Emotion", "Text Emotion", "About Models"],
)

if page == "Home":
    st.subheader("Welcome")
    st.write(
        "Choose an image or enter a text sample to see the model's predicted "
        "emotion and class probabilities. Image and text predictions use "
        "separate pretrained models."
    )
    st.info(
        "These are model predictions, not proof of a person's actual internal "
        "emotional state. Facial-expression predictions can be uncertain and "
        "may not generalize equally across people or settings."
    )

elif page == "Image Emotion":
    st.subheader("Image-based Emotion Classification")
    uploaded_file = st.file_uploader(
        "Upload a face image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            st.error("The uploaded file could not be read as an image. Please try a valid JPG or PNG.")
        else:
            st.image(image, caption="Uploaded image", width=360)

            if st.button("Analyse Image", type="primary"):
                try:
                    with st.spinner("Analysing image…"):
                        results = predict_image(image)
                    if not results:
                        st.warning("The image model returned no class scores.")
                    else:
                        top_result = results[0]
                        st.metric(
                            "Top predicted expression",
                            top_result["label"].replace("_", " ").title(),
                            f"{top_result['score']:.1%} model score",
                        )
                        scores = pd.DataFrame(results)
                        scores["label"] = scores["label"].str.replace("_", " ").str.title()
                        fig = px.bar(
                            scores.sort_values("score"),
                            x="score",
                            y="label",
                            orientation="h",
                            range_x=[0, 1],
                            labels={"score": "Model score", "label": "Class"},
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as exc:
                    st.error(f"Image prediction failed: {exc}")

elif page == "Text Emotion":
    st.subheader("Text-based Emotion Classification")
    text_input = st.text_area(
        "Enter an English sentence or short paragraph",
        placeholder="Example: I am really excited about starting my new job!",
        height=140,
    )

    if st.button("Analyse Text", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text first.")
        else:
            try:
                with st.spinner("Analysing text…"):
                    results = predict_text(text_input)
                if not results:
                    st.warning("The text model returned no class scores.")
                else:
                    top_result = results[0]
                    st.metric(
                        "Top predicted emotion",
                        top_result["label"].replace("_", " ").title(),
                        f"{top_result['score']:.1%} model score",
                    )
                    scores = pd.DataFrame(results)
                    scores["label"] = scores["label"].str.replace("_", " ").str.title()
                    fig = px.bar(
                        scores.sort_values("score"),
                        x="score",
                        y="label",
                        orientation="h",
                        range_x=[0, 1],
                        labels={"score": "Model score", "label": "Class"},
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Text prediction failed: {exc}")

else:
    st.subheader("About the Models")
    st.markdown(
        f"**Image model:** [`{IMAGE_MODEL_ID}`](https://huggingface.co/{IMAGE_MODEL_ID})  \n"
        "Vision Transformer (ViT), loaded with Hugging Face Transformers and PyTorch."
    )
    st.markdown(
        f"**Text model:** [`{TEXT_MODEL_ID}`](https://huggingface.co/{TEXT_MODEL_ID})  \n"
        "DistilRoBERTa text-emotion classifier, loaded with Hugging Face Transformers and PyTorch."
    )
    st.caption(
        "The pretrained models have their own label sets. These are not the same "
        "as the six labels used in the original notebooks."
    )
