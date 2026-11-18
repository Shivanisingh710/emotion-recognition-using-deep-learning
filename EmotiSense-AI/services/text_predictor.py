"""Text preprocessing and emotion prediction."""

import torch

from config.settings import MAX_TEXT_LENGTH
from models.text_model import load_text_model


def predict_text(text: str) -> list[dict[str, float | str]]:
    """Predict emotion scores for an English text input.

    Returns a descending list of dictionaries with ``label`` and ``score``.
    Inputs longer than MAX_TEXT_LENGTH tokens are truncated by the tokenizer.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Please enter some text before running prediction.")

    tokenizer, model, device = load_text_model()
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TEXT_LENGTH,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()

    id2label = model.config.id2label
    results = [
        {
            "label": str(id2label.get(index, id2label.get(str(index), index))),
            "score": float(score),
        }
        for index, score in enumerate(probabilities)
    ]

    return sorted(results, key=lambda item: item["score"], reverse=True)
