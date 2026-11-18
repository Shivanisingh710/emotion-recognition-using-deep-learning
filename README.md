# EmotiSense AI — Multimodal Emotion Recognition

EmotiSense AI is a Streamlit application that explores emotion classification through two independent input pipelines: **facial images** and **English text**. It uses pretrained Hugging Face Transformer models with PyTorch inference and presents the predicted class scores through an interactive web interface.

🚀 **Live Demo:** [Open EmotiSense AI](https://emotion-recognition-using-deep-learning-aaejxtn6rqgboaztfamiyc.streamlit.app/)

## Project Overview

The application provides two ways to explore model predictions:

- **Image Emotion:** Upload a JPG, JPEG, or PNG image containing a visible face to classify facial expressions.
- **Text Emotion:** Enter an English sentence or short paragraph to classify its expressed emotion.

The two inputs are processed independently using separate pretrained models.

## Key Features

- Image-based emotion classification.
- Text-based emotion classification.
- Predicted emotion label and model score.
- Interactive Plotly probability visualizations.
- Separate image and text model pipelines.
- Hugging Face pretrained models with PyTorch inference.
- Streamlit-based interactive interface.

## Models Used

| Input | Model | Architecture |
|---|---|---|
| Image | dima806/facial_emotions_image_detection | Vision Transformer (ViT) |
| Text | j-hartmann/emotion-english-distilroberta-base | DistilRoBERTa |

## Technology Stack

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- Vision Transformer (ViT)
- DistilRoBERTa
- Pillow
- Pandas and NumPy
- Plotly

## Original Deep Learning Experiments

The project also preserves two original notebooks:

**Image Emotion Classification**
- Custom CNN and transfer-learning experiments.
- Six classes: Anger, Disgust, Fear, Happy, Pain and Sad.
- VGG16 experiment with a reported 51% test accuracy.

**Text Emotion Classification**
- Text cleaning and preprocessing.
- Tokenization and sequence padding.
- LSTM-based classification.
- Six classes: Anger, Fear, Joy, Love, Sadness and Surprise.
- Reported test accuracy: 87%.

These notebook results belong to the original TensorFlow/Keras experiments and are not the performance metrics of the pretrained models used in the Streamlit application.

## Run Locally

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app.py
```

## Responsible Use

Model predictions are not proof of a person's actual internal emotional state. Results may be affected by image quality, lighting, pose, language and context. This project is intended for learning and demonstration, not clinical or high-stakes decision-making.

## Author

**Shivani Singh**

- [GitHub](https://github.com/Shivanisingh710)
- [Live Application](https://emotion-recognition-using-deep-learning-aaejxtn6rqgboaztfamiyc.streamlit.app/)
