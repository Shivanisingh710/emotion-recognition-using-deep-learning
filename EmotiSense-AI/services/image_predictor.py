"""Image preprocessing and emotion prediction."""

import torch
from PIL import Image

from models.image_model import load_image_model


def predict_image(image: Image.Image) -> list[dict[str, float | str]]:
    """Predict emotion scores for a PIL image.

    Returns a descending list of dictionaries with ``label`` and ``score``.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image object")

    image = image.convert("RGB")
    processor, model, device = load_image_model()
    inputs = processor(images=image, return_tensors="pt")
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
