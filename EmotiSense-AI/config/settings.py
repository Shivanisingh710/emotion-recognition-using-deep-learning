"""Central configuration for EmotiSense AI."""

APP_NAME = "EmotiSense AI"

# Pin model revisions so a future update to a Hugging Face main branch does
# not silently change the models used by this project.
IMAGE_MODEL_ID = "dima806/facial_emotions_image_detection"
IMAGE_MODEL_REVISION = "e0334e9c554ce5435b684c711516f049bc7e7bb9"

TEXT_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
TEXT_MODEL_REVISION = "cdc18393b6e2e92fe122e225173c95cbd9cc2b19"

# Maximum number of tokens passed to the text classifier.
MAX_TEXT_LENGTH = 512
