"""Offline unit tests for predictor logic (no Hugging Face downloads)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from services import image_predictor, text_predictor


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeImageProcessor:
    def __call__(self, images, return_tensors):
        if images.mode != "RGB":
            raise AssertionError("Image should be converted to RGB")
        if return_tensors != "pt":
            raise AssertionError("Expected PyTorch tensors")
        return {"pixel_values": torch.zeros((1, 3, 224, 224))}


class FakeImageModel:
    config = SimpleNamespace(id2label={0: "sad", 1: "happy", 2: "angry"})

    def __call__(self, **kwargs):
        return FakeOutput(torch.tensor([[0.1, 2.0, 0.3]]))


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        self.last_text = text
        self.last_kwargs = kwargs
        return {"input_ids": torch.tensor([[1, 2, 3]])}


class FakeTextModel:
    config = SimpleNamespace(id2label={0: "anger", 1: "joy", 2: "sadness"})

    def __call__(self, **kwargs):
        return FakeOutput(torch.tensor([[0.1, 2.0, 0.3]]))


class PredictorTests(unittest.TestCase):
    def test_image_prediction_returns_sorted_probabilities(self):
        with patch.object(
            image_predictor,
            "load_image_model",
            return_value=(FakeImageProcessor(), FakeImageModel(), torch.device("cpu")),
        ):
            image = Image.new("RGBA", (32, 32), (255, 0, 0, 128))
            results = image_predictor.predict_image(image)

        self.assertEqual(results[0]["label"], "happy")
        self.assertEqual(results, sorted(results, key=lambda item: item["score"], reverse=True))
        self.assertAlmostEqual(sum(item["score"] for item in results), 1.0, places=6)

    def test_image_prediction_rejects_non_pillow_input(self):
        with self.assertRaises(TypeError):
            image_predictor.predict_image("not an image")

    def test_text_prediction_strips_text_and_returns_sorted_probabilities(self):
        tokenizer = FakeTokenizer()
        with patch.object(
            text_predictor,
            "load_text_model",
            return_value=(tokenizer, FakeTextModel(), torch.device("cpu")),
        ):
            results = text_predictor.predict_text("  I feel happy  ")

        self.assertEqual(tokenizer.last_text, "I feel happy")
        self.assertTrue(tokenizer.last_kwargs["truncation"])
        self.assertEqual(tokenizer.last_kwargs["max_length"], 512)
        self.assertEqual(results[0]["label"], "joy")
        self.assertEqual(results, sorted(results, key=lambda item: item["score"], reverse=True))
        self.assertAlmostEqual(sum(item["score"] for item in results), 1.0, places=6)

    def test_text_prediction_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            text_predictor.predict_text("   ")

    def test_text_prediction_rejects_non_string_input(self):
        with self.assertRaises(TypeError):
            text_predictor.predict_text(None)


if __name__ == "__main__":
    unittest.main()
