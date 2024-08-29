import os
import json
import re
import unicodedata

import fitz  # PyMuPDF
import docx
import numpy as np
import easyocr
import requests
from PIL import Image
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer
from semantic_text_splitter import TextSplitter
import streamlit as st
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError

# Load environment variables from .env file
load_dotenv()

# Access environment variables
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
MODEL_PATH = os.getenv("MODEL_PATH")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH")
MAX_TOKENS = int(os.getenv("MAX_TOKENS"))
EASYOCR_MODELS_PATH = os.getenv("EASYOCR_MODELS_PATH", "./models")

# Initialize model, tokenizer, and text splitter
model = SentenceTransformer(MODEL_PATH)
tokenizer = Tokenizer.from_file(os.path.join(TOKENIZER_PATH, "tokenizer.json"))
splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, MAX_TOKENS, trim=False)
client = OpenAI(base_url=f"{OPENAI_BASE_URL}/v1", api_key=OPENAI_API_KEY)

# Load OCR languages from the JSON file
with open('ocr_languages.json', 'r') as file:
    ocr_languages = json.load(file)
language_options = [lang["language"] for lang in ocr_languages]
language_codes = {lang["language"]: lang["code"] for lang in ocr_languages}

def clean_text(text):
    """Cleans and normalizes text."""
    if text is None:
        return ""
    if isinstance(text, list):
        text = " ".join(item['text'] if isinstance(item, dict) and 'text' in item else str(item) for item in text)
    if not isinstance(text, str):
        raise TypeError(f"Expected a string for cleaning, but got {type(text)}")
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u0600-\u06FF.,!?;:()\[\]{}\'"-]', '', text)
    return text.strip()

def split_text_semantically(text):
    """Splits text into semantic chunks."""
    return splitter.chunks(text)

def extract_text_from_pdf(uploaded_file, use_ocr=False, languages=None):
    """Extracts text from a PDF file using OCR or direct reading."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""

    if use_ocr and languages:
        selected_codes = [language_codes[lang] for lang in languages]
        reader = easyocr.Reader(selected_codes, model_storage_directory=EASYOCR_MODELS_PATH)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            result = reader.readtext(np.array(img), detail=0)
            text += "\n".join(result) + "\n"
    else:
        text = "".join([page.get_text() for page in doc])

    return text

def extract_text_from_image(uploaded_file, languages=None):
    """Extracts text from an image file using OCR."""
    if languages:
        selected_codes = [language_codes[lang] for lang in languages]
        reader = easyocr.Reader(selected_codes, model_storage_directory=EASYOCR_MODELS_PATH)
        result = reader.readtext(uploaded_file.getvalue(), detail=0)
        text = "\n".join(result)
        return text
    else:
        st.warning("Please select at least one language for OCR.")
        return None

def extract_text_from_file(uploaded_file, ocr_option=None, languages=None):
    """Extracts text from various file types."""
    file_type = uploaded_file.type
    if file_type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    elif file_type == "application/pdf":
        return extract_text_from_pdf(uploaded_file, use_ocr=(ocr_option == "Read With OCR"), languages=languages)
    elif file_type.startswith("image/"):
        return extract_text_from_image(uploaded_file, languages)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        st.error("Unsupported file type.")
        return None

def get_model_health():
    """Gets the model health status."""
    try:
        health_url = f"{OPENAI_BASE_URL}/health"
        response = requests.get(health_url)
        if response.status_code == 200:
            status = response.json().get("status", "unknown")
            return status
        else:
            return "unknown"
    except Exception as e:
        return f"error ({str(e)})"
