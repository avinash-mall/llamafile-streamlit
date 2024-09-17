import json
import os
import re
import unicodedata
import warnings
from datetime import datetime
from elasticsearch import Elasticsearch, exceptions, ElasticsearchWarning
import fitz  # PyMuPDF
import docx
import numpy as np
import easyocr
import psutil
import requests
from PIL import Image
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from streamlit_autorefresh import st_autorefresh
from tokenizers import Tokenizer
from semantic_text_splitter import TextSplitter
import streamlit as st

# Load environment variables from .env file
load_dotenv()
# Initialize Elasticsearch with authentication
es = Elasticsearch(
    hosts=[os.getenv('ES_HOST_URL')],
    basic_auth=(os.getenv('ES_USERNAME'), os.getenv('ES_PASSWORD'))
)

num_results = int(os.getenv("NUM_RESULTS", 10))  # Default to 10 if not provided
num_candidates = int(os.getenv("NUM_CANDIDATES", 100))
min_score = float(os.getenv("MIN_SCORE", 1.78))

# Suppress Elasticsearch system indices warnings
warnings.filterwarnings("ignore", category=ElasticsearchWarning)

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

# Load OCR languages from the JSON file
with open('ocr_languages.json', 'r') as file:
    ocr_languages = json.load(file)

# Create language options and language codes dictionaries
language_options = {lang["code"]: lang["language"] for lang in ocr_languages}
language_codes = {lang["language"]: lang["code"] for lang in ocr_languages}

# Load settings from settings.json file
with open('settings.json') as f:
    settings = json.load(f)

def load_advanced_settings():
    """Load and return default values for advanced settings when hidden."""
    return {
        "temperature": settings["temperature"]["default"],
        "top_p": settings["top_p"]["default"],
        "frequency_penalty": settings["frequency_penalty"]["default"],
        "presence_penalty": settings["presence_penalty"]["default"],
        "seed": settings["seed"]["default"],
        "logit_bias": settings["logit_bias"]["default"],
        "logprobs": settings["logprobs"]["default"],
        "top_logprobs": settings["top_logprobs"]["default"],
        "max_tokens": settings["max_tokens"]["default"],
        "n": settings["n"]["default"],
        "stop": json.dumps(settings["stop"]["default"]),
        "stream_t": settings["stream"]["default"],
    }

def display_advanced_settings(settings_visible):
    """Display advanced settings sliders and inputs when settings_visible is True."""
    if settings_visible:
        advanced_settings = {
            "temperature": st.sidebar.slider("Temperature", min_value=settings["temperature"]["min"], max_value=settings["temperature"]["max"],
                                             value=settings["temperature"]["default"], step=settings["temperature"]["step"], help=settings["temperature"]["help"]),
            "top_p": st.sidebar.slider("Top P", min_value=settings["top_p"]["min"], max_value=settings["top_p"]["max"],
                                       value=settings["top_p"]["default"], step=settings["top_p"]["step"], help=settings["top_p"]["help"]),
            "frequency_penalty": st.sidebar.slider("Frequency Penalty", min_value=settings["frequency_penalty"]["min"], max_value=settings["frequency_penalty"]["max"],
                                                   value=settings["frequency_penalty"]["default"], step=settings["frequency_penalty"]["step"], help=settings["frequency_penalty"]["help"]),
            "presence_penalty": st.sidebar.slider("Presence Penalty", min_value=settings["presence_penalty"]["min"], max_value=settings["presence_penalty"]["max"],
                                                  value=settings["presence_penalty"]["default"], step=settings["presence_penalty"]["step"], help=settings["presence_penalty"]["help"]),
            "seed": st.sidebar.number_input("Seed", value=settings["seed"]["default"], help=settings["seed"]["help"]),
            "logit_bias": st.sidebar.text_area("Logit Bias", settings["logit_bias"]["default"], help=settings["logit_bias"]["help"]),
            "logprobs": st.sidebar.toggle("Return Log Probabilities", value=settings["logprobs"]["default"], help=settings["logprobs"]["help"]),
            "top_logprobs": st.sidebar.number_input("Top Logprobs", min_value=settings["top_logprobs"]["min"], max_value=settings["top_logprobs"]["max"],
                                                    value=settings["top_logprobs"]["default"], help=settings["top_logprobs"]["help"]),
            "max_tokens": st.sidebar.number_input("Max Tokens", min_value=settings["max_tokens"]["min"], max_value=settings["max_tokens"]["max"],
                                                  value=settings["max_tokens"]["default"], help=settings["max_tokens"]["help"]),
            "n": st.sidebar.number_input("Number of Choices (n)", min_value=settings["n"]["min"], max_value=settings["n"]["max"],
                                         value=settings["n"]["default"], help=settings["n"]["help"]),
            "stop": st.sidebar.text_area("Stop Sequences", json.dumps(settings["stop"]["default"]), help=settings["stop"]["help"]),
            "stream_t": st.sidebar.toggle("Stream Output", value=settings["stream"]["default"], help=settings["stream"]["help"]),
            "debug": st.sidebar.toggle("Enable Debug Mode", value=False, help="Toggle to enable or disable debug mode for detailed insights.")
        }
        # Update session state for debug mode
        st.session_state["debug"] = advanced_settings["debug"]
        return advanced_settings
    else:
        return load_advanced_settings()

def toggle_display_metrics():
    """Toggle to enable/disable the display of CPU, Memory, and Model Health metrics."""
    return st.sidebar.toggle("Display CPU, Memory, and Model Health Metrics", value=False, help="Enable to show CPU, memory, and model health metrics")

def refresh_metrics(display_metrics):
    """Automatically refresh CPU, Memory, and Health status every 5 seconds if metrics display is enabled."""
    if display_metrics:
        st_autorefresh(interval=5000, key="status_refresh")

        # Update CPU, Memory, and Health status in real-time
        cpu_usage = psutil.cpu_percent(interval=1)
        mem_usage = psutil.virtual_memory().percent
        model_health = get_model_health()

        # Create rows and containers for CPU, Memory, and Health status
        row1 = st.columns(3)
        cpu_container = row1[0].container()
        mem_container = row1[1].container()
        health_container = row1[2].container()

        # Display the metrics inside their respective containers
        cpu_container.metric(label="CPU Usage", value=f"{cpu_usage}%")
        mem_container.metric(label="Memory Usage", value=f"{mem_usage}%")
        health_container.metric(label="Model Health", value=model_health)

def append_date_time_to_prompt(prompt_content, append_date_time):
    """Append date and time information to the selected prompt content if enabled."""
    if prompt_content is None:
        prompt_content = ""  # Initialize as an empty string if None

    if append_date_time:
        now = datetime.now().astimezone()
        current_time = now.strftime("%H:%M:%S")
        current_date = now.strftime("%Y-%m-%d")
        current_day = now.strftime("%A")
        current_timezone = now.tzname()

        date_time_info = f"The current time is: {current_time}, date: {current_date}, day: {current_day}, timezone: {current_timezone}."
        prompt_content += "\n" + date_time_info

    return prompt_content


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
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""

        if use_ocr and languages:
            selected_codes = [language_codes.get(lang, lang) for lang in languages]
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

    except RuntimeError as e:
        st.error("Failed to read the PDF file. Please ensure the file is a valid PDF.")
        st.error(f"Error details: {str(e)}")
        return None

    except Exception as e:
        st.error("An unexpected error occurred while processing the PDF file.")
        st.error(f"Error details: {str(e)}")
        return None


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

def search_elasticsearch(query, index_name):
    query_embedding = model.encode(query).tolist()

    try:
        response = es.search(index=index_name, body={
            "query": {
                "knn": {
                    "query_vector": query_embedding,
                    "field": "embedding",
                    "k": num_results,
                    "num_candidates": num_candidates
                }
            },
            "min_score": min_score
        })
        hits = response['hits']['hits']
        return hits
    except exceptions.RequestError as e:
        st.error(f"Request Error: {e.info}")
    except exceptions.ConnectionError as e:
        st.error(f"Connection Error: {e.errors}")
    except Exception as e:
        st.error(f"General Error: {str(e)}")


def display_debug_info(summary, prompt, messages):
    """Displays debug information in a table format if debug mode is enabled."""
    if st.session_state.get("debug", False):
        debug_data = {
            "Instruction Prompt": [st.session_state["system_instruction"]],
            "User Message": [prompt],
            "Assistant Message": [summary],
            "Message History": [str(messages)]
        }
        st.subheader("Debug Information")
        st.table(debug_data)

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

def clear_chat_history():
    """Clears the chat history stored in session state."""
    if "messages" in st.session_state:
        st.session_state["messages"] = []
    if "system_instruction" in st.session_state:
        st.session_state["system_instruction"] = None
