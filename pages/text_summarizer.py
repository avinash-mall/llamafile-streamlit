import os
import re
import unicodedata
import time
import json
import fitz  # PyMuPDF
import streamlit as st
import easyocr
import docx
import numpy as np
from PIL import Image
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError
from dotenv import load_dotenv
from semantic_text_splitter import TextSplitter
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer

# Load environment variables from .env file
load_dotenv()

# Access variables from the environment
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
MODEL_PATH = os.getenv("MODEL_PATH")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH")
MAX_TOKENS = int(os.getenv("MAX_TOKENS"))
SYSTEM_INSTRUCTION = os.getenv("SYSTEM_INSTRUCTION", "You're a Text Summarizer...")  # Default instruction if not set
EASYOCR_MODELS_PATH = os.getenv("EASYOCR_MODELS_PATH", "./models")  # Default to './models' if not set

# Load settings from settings.json file
with open('settings.json') as f:
    settings = json.load(f)

# Initialize SentenceTransformer model and tokenizer
model = SentenceTransformer(MODEL_PATH)
tokenizer = Tokenizer.from_file(os.path.join(TOKENIZER_PATH, "tokenizer.json"))
splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, MAX_TOKENS, trim=False)

st.set_page_config(layout="wide")
st.title("💬 Text Summarizer")

# Initialize the OpenAI client
client = OpenAI(base_url=f"{OPENAI_BASE_URL}/v1", api_key=OPENAI_API_KEY)

# Initialize session state attributes specifically for this page
st.session_state["messages"] = []  # Reset messages when visiting this page
st.session_state["system_instruction"] = SYSTEM_INSTRUCTION  # Set from .env variable for this page

# Load OCR languages from the JSON file
with open('ocr_languages.json', 'r') as file:
    ocr_languages = json.load(file)

language_options = [lang["language"] for lang in ocr_languages]
language_codes = {lang["language"]: lang["code"] for lang in ocr_languages}

# Sidebar settings
st.sidebar.title("Settings")

# Toggle to show/hide advanced settings
settings_visible = st.sidebar.toggle("Show/Hide Advanced Settings", value=False)

if settings_visible:
    # Advanced settings sliders and inputs
    temperature = st.sidebar.slider("Temperature", min_value=settings["temperature"]["min"], max_value=settings["temperature"]["max"],
                                    value=settings["temperature"]["default"], step=settings["temperature"]["step"], help=settings["temperature"]["help"])

    top_p = st.sidebar.slider("Top P", min_value=settings["top_p"]["min"], max_value=settings["top_p"]["max"],
                              value=settings["top_p"]["default"], step=settings["top_p"]["step"], help=settings["top_p"]["help"])

    frequency_penalty = st.sidebar.slider("Frequency Penalty", min_value=settings["frequency_penalty"]["min"], max_value=settings["frequency_penalty"]["max"],
                                          value=settings["frequency_penalty"]["default"], step=settings["frequency_penalty"]["step"], help=settings["frequency_penalty"]["help"])

    presence_penalty = st.sidebar.slider("Presence Penalty", min_value=settings["presence_penalty"]["min"], max_value=settings["presence_penalty"]["max"],
                                         value=settings["presence_penalty"]["default"], step=settings["presence_penalty"]["step"], help=settings["presence_penalty"]["help"])

    seed = st.sidebar.number_input("Seed", value=settings["seed"]["default"], help=settings["seed"]["help"])

    logit_bias = st.sidebar.text_area("Logit Bias", settings["logit_bias"]["default"], help=settings["logit_bias"]["help"])

    logprobs = st.sidebar.toggle("Return Log Probabilities", value=settings["logprobs"]["default"], help=settings["logprobs"]["help"])

    top_logprobs = st.sidebar.number_input("Top Logprobs", min_value=settings["top_logprobs"]["min"], max_value=settings["top_logprobs"]["max"],
                                           value=settings["top_logprobs"]["default"], help=settings["top_logprobs"]["help"])

    max_tokens = st.sidebar.number_input("Max Tokens", min_value=settings["max_tokens"]["min"], max_value=settings["max_tokens"]["max"],
                                         value=settings["max_tokens"]["default"], help=settings["max_tokens"]["help"])

    n = st.sidebar.number_input("Number of Choices (n)", min_value=settings["n"]["min"], max_value=settings["n"]["max"],
                                value=settings["n"]["default"], help=settings["n"]["help"])

    stop = st.sidebar.text_area("Stop Sequences", json.dumps(settings["stop"]["default"]), help=settings["stop"]["help"])

    stream_t = st.sidebar.toggle("Stream Output", value=settings["stream"]["default"], help=settings["stream"]["help"])

    # Enable debug mode toggle inside advanced settings
    st.session_state["debug"] = st.sidebar.toggle("Enable Debug Mode", value=False, help="Toggle to enable or disable debug mode for detailed insights.")

# Layout for file upload and OCR option
uploaded_file = st.file_uploader(
    "Upload a file",
    type=["txt", "pdf", "docx", "epub", "mobi", "fb2", "svg", "png", "jpeg", "jpg"],
    key="file_uploader"
)
ocr_option = None
selected_languages = []

# Prompt for OCR options for PDF files
if uploaded_file and uploaded_file.type == "application/pdf":
    ocr_option = st.radio(
        "Select how to read the PDF",
        ["Read Without OCR", "Read With OCR"],
        index=None
    )
    if ocr_option == "Read With OCR":
        selected_languages = st.multiselect(
            "Select OCR Languages",
            options=language_options,
            default=["English", "Arabic"],
            max_selections=2,
            placeholder="Choose Language(s)"
        )

# Prompt for OCR options for image files
elif uploaded_file and uploaded_file.type.startswith("image/"):
    selected_languages = st.multiselect(
        "Select OCR Languages for the Image",
        options=language_options,
        default=["English", "Arabic"],
        max_selections=2,
        placeholder="Choose Language(s)"
    )

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
        if ocr_option:
            return extract_text_from_pdf(uploaded_file, use_ocr=(ocr_option == "Read With OCR"), languages=languages)
        else:
            st.warning("Please select an option to continue.")
            return None
    elif file_type.startswith("image/"):
        return extract_text_from_image(uploaded_file, languages)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        st.error("Unsupported file type.")
        return None

def display_debug_info(summary, prompt, messages):
    """Displays debug information in a table format if debug mode is enabled."""
    if st.session_state["debug"]:
        debug_data = {
            "Instruction Prompt": [st.session_state["system_instruction"]],
            "User Message": [prompt],
            "Assistant Message": [summary],
            "Message History": [str(messages)]
        }
        st.subheader("Debug Information")
        st.table(debug_data)

def extract_and_summarize(uploaded_file, ocr_option=None, languages=None):
    """Extracts text from the uploaded file and summarizes it."""
    text = extract_text_from_file(uploaded_file, ocr_option, languages)
    if not text:
        return

    chunks = split_text_semantically(clean_text(text))
    total_chunks = len(chunks)
    previous_summaries = ""
    start_time = time.time()

    # Initialize progress bar and chunk counter
    progress_bar = st.progress(0)
    chunk_counter = st.empty()

    # Container for summary outputs
    with st.container():
        st.header("Summary", divider=True)

        # Summarize each chunk based on previous summaries
        for i, chunk in enumerate(chunks):
            chunk_counter.markdown(f"**Chunks Processed: {i + 1}/{total_chunks}**")
            part_info = f"Part {i + 1} of {total_chunks}"
            prompt = f"Context:\n{previous_summaries}\n\n{part_info}\n\nText:\n{chunk}"

            try:
                with st.spinner("Thinking..."):
                    with st.chat_message("assistant"):
                        # Prepare messages for LLM API
                        messages = [{"role": "system", "content": st.session_state["system_instruction"]}]
                        # Include only the current chunk as the user message
                        messages.append({"role": "user", "content": chunk})
                        # Add previous summaries to the assistant context if they exist
                        if previous_summaries.strip():
                            messages.append({"role": "assistant", "content": previous_summaries.strip()})
                        stream = client.chat.completions.create(
                            model=OPENAI_MODEL,
                            messages=messages,
                            temperature=temperature,
                            top_p=top_p,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                            seed=seed,
                            logit_bias=eval(logit_bias),
                            logprobs=logprobs,
                            top_logprobs=top_logprobs if logprobs else None,
                            max_tokens=max_tokens,
                            n=n,
                            stop=json.loads(stop),
                            stream=stream_t,
                        )
                        response = st.write_stream(stream)
                    st.session_state["messages"].append({"role": "assistant", "content": response})
                    display_debug_info(response, prompt, st.session_state["messages"])
                    previous_summaries += response + "\n"  # Accumulate summaries for context

            except APIConnectionError as e:
                st.error("The server could not be reached.")
                st.error(f"Details: {e.__cause__}")
            except RateLimitError as e:
                st.error("Rate limit exceeded; please try again later.")
            except APIStatusError as e:
                st.error(f"An error occurred: {e.status_code}")
                st.error(f"Response: {e.response}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

            # Update progress bar
            progress_bar.progress((i + 1) / total_chunks)

    end_time = time.time()
    elapsed_time = end_time - start_time
    st.markdown(f"Summarization completed in :blue[{elapsed_time:.2f}] seconds.")

# Display start button to initiate processing
if uploaded_file:
    if st.button("Start Processing"):
        extract_and_summarize(uploaded_file, ocr_option, selected_languages)

# Chat input for additional text
if prompt := st.chat_input("Or enter text here to get a summary"):
    extract_and_summarize(st.file_uploader("Upload a file", key="chat_file_uploader"))
