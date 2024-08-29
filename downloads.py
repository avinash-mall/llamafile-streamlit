# download_model.py

from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

model_name = os.getenv("MODEL_NAME")
model_path = os.getenv("MODEL_PATH")

# Check if the model already exists
if not os.path.exists(model_path) or not os.listdir(model_path):
    # Download and save the model locally
    model = SentenceTransformer(model_name)
    os.makedirs(model_path, exist_ok=True)
    model.save(model_path)
    print(f"Model '{model_name}' downloaded and saved to '{model_path}'.")
else:
    print(f"Model '{model_name}' already exists at '{model_path}'.")

# download_tokenizer.py

from transformers import AutoTokenizer
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

tokenizer_name = os.getenv("TOKENIZER_NAME")
tokenizer_path = os.getenv("TOKENIZER_PATH")

# Check if the tokenizer already exists
if not os.path.exists(tokenizer_path) or not os.listdir(tokenizer_path):
    # Download and save the tokenizer locally
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer '{tokenizer_name}' downloaded and saved to '{tokenizer_path}'.")
else:
    print(f"Tokenizer '{tokenizer_name}' already exists at '{tokenizer_path}'.")

# EasyOCR model downloads

import easyocr

EASYOCR_MODELS_PATH = os.getenv("EASYOCR_MODELS_PATH", "./models/easyocr")
# Create the models directory if it doesn't exist
models_dir = EASYOCR_MODELS_PATH
os.makedirs(models_dir, exist_ok=True)

# List of supported languages in EasyOCR
languages = [
    'abq', 'ady', 'af', 'ang', 'ar', 'as', 'ava', 'az', 'be', 'bg', 'bh', 'bho', 'bn',
    'bs', 'ch_sim', 'ch_tra', 'che', 'cs', 'cy', 'da', 'dar', 'de', 'en', 'es', 'et',
    'fa', 'fr', 'ga', 'gom', 'hi', 'hr', 'hu', 'id', 'inh', 'is', 'it', 'ja', 'kbd',
    'kn', 'ko', 'ku', 'la', 'lbe', 'lez', 'lt', 'lv', 'mah', 'mai', 'mi', 'mn', 'mr',
    'ms', 'mt', 'ne', 'new', 'nl', 'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'ru', 'rs_cyrillic',
    'rs_latin', 'sck', 'sk', 'sl', 'sq', 'sv', 'sw', 'ta', 'tab', 'te', 'th', 'tjk',
    'tl', 'tr', 'ug', 'uk', 'ur', 'uz', 'vi'
]

# Download models for each language and store them in the ./models directory
for lang in languages:
    try:
        # Check if model directory for the language already exists and is not empty
        lang_model_dir = os.path.join(models_dir, lang)
        if not os.path.exists(lang_model_dir) or not os.listdir(lang_model_dir):
            print(f"Downloading models for {lang}...")
            reader = easyocr.Reader([lang], model_storage_directory=models_dir, download_enabled=True)
            print(f"Model for {lang} downloaded and stored in '{lang_model_dir}'.")
        else:
            print(f"Model for {lang} already exists in '{lang_model_dir}'. Skipping download.")
    except Exception as e:
        print(f"Error downloading model for {lang}: {e}. Skipping to the next language.")

print("All models processed.")
