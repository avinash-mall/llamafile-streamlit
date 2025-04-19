import hashlib
import json
import os
import re
import unicodedata
import warnings
from datetime import datetime
from urllib.parse import urlencode
from scipy.spatial.distance import cosine
from datetime import datetime
import pandas as pd
import pytz
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
from yaml.loader import SafeLoader
import smtplib
from email.message import EmailMessage
import yaml
import streamlit as st
import streamlit_authenticator as stauth
from urllib.parse import urlencode
import secrets
# Suppress Elasticsearch system indices warnings
warnings.filterwarnings("ignore", category=ElasticsearchWarning)

def send_email(subject, body, to_email, config):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = config['smtp']['username']
    msg['To'] = to_email

    with smtplib.SMTP(config['smtp']['server'], config['smtp']['port']) as server:
        #if config['smtp']['use_tls']:
        #    server.starttls()
        #server.login(config['smtp']['username'], config['smtp']['password'])
        server.send_message(msg)


def send_reset_password_email(name, new_password, to_email, config):
    subject = "Your New Password"
    body = f"Hey {name},\n\nHere is your new password:\n\n {new_password}\n\nPlease change it once you log in."

    send_email(subject, body, to_email, config)


def send_forgot_username_email(name, username, to_email, config):
    subject = "Your Username Reminder"
    body = f"Hey {name},\n\nYour username is: \n\n{username}\n\n"

    send_email(subject, body, to_email, config)

# Helper to determine if password is alredy hashed
def is_bcrypt_hash(s):
    return s.startswith(('$2a$', '$2b$', '$2x$', '$2y$')) and len(s) == 60


# Hash new plaintext passwords only
def hash_plaintext_passwords(config):
    plaintext_passwords = {}
    for user, details in config['credentials']['usernames'].items():
        # Check if the password is not a bcrypt hash
        if not is_bcrypt_hash(details['password']):
            plaintext_passwords[user] = details['password']

    if plaintext_passwords:
        hashed_passwords = stauth.Hasher(list(plaintext_passwords.values())).generate()
        for user, hashed_pw in zip(plaintext_passwords.keys(), hashed_passwords):
            config['credentials']['usernames'][user]['password'] = hashed_pw

    return config

# ——————————————————————————————————————————————
# CONFIG LOADING & AUTHENTICATOR INSTANTIATION
# ——————————————————————————————————————————————

def load_auth_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def update_yaml(config: dict, path: str = "config.yaml") -> None:
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

# Load once at import time
_config = load_auth_config()
if "hashed_done" not in st.session_state:
    # Pre‑hash any plaintext passwords in-place
    _config["credentials"] = stauth.Hasher.hash_passwords(_config["credentials"])
    update_yaml(_config)                            # persist the hashes
    st.session_state.hashed_done = True

authenticator = stauth.Authenticate(
    credentials=_config["credentials"],
    cookie_name=_config["cookie"]["name"],
    key=_config["cookie"]["key"],
    cookie_expiry_days=_config["cookie"]["expiry_days"],
    auto_hash=False    # <- do NOT auto-hash again
)

# In‑memory store for users who have registered but not yet verified
unverified_users: dict[str, dict] = {}

# Login function
def login(authenticator):
    custom_fields = {
        'Form name': 'Please Login',
        'Username': 'Enter Username',
        'Password': 'Enter Password',
        'Login': 'Sign In'
    }

    name, authentication_status, username = authenticator.login(
        location='main',
        max_concurrent_users=5,
        max_login_attempts=3,
        fields=custom_fields,
        captcha=False,
        clear_on_submit=True,
        key='LoginWidget'
    )

    return name, authentication_status, username


# Logout function
def logout(authenticator):
    try:
        authenticator.logout('Logout', location='main', key='LogoutWidget')
        #st.toast('User logged out successfully', icon="✅")
    except Exception as e:
        st.toast(e, icon="⚠️")

# Generate Token
def generate_verification_token():
    # Generate a unique token (e.g., use the secrets module)
    return secrets.token_urlsafe(16)


# Temporary storage for unverified users (can be a dictionary or external storage)
unverified_users = {}


# ——————————————————————————————————————————————
# MANUAL REGISTRATION & VERIFICATION LOGIC
# ——————————————————————————————————————————————

def register_user(authenticator, config):
    """
    1) Render the Streamlit‑Authenticator registration form.
    2) On submit, stash the new user in `unverified_users` and
       email them a verification link.
    """
    email, username, name = authenticator.register_user(
        location="sidebar",
        pre_authorized=None,       # open signup
        clear_on_submit=True,
        key="RegisterWidget"
    )
    if email:
        # generate token & store pending user
        token = secrets.token_urlsafe(16)
        unverified_users[email] = {
            "username": username,
            "name": name,
            "password": config["credentials"]["usernames"][username]["password"],
            "token": token
        }
        # build & send verification link
        params = urlencode({"token": token, "email": email})
        link = f"{config['app']['domain']}?&{params}"
        subject = "🔒 Complete your registration"
        body = (
            f"Hi {name},\n\n"
            f"Thanks for registering. Please click the link below to verify your email and activate your account:\n\n"
            f"{link}\n\n"
            "If you didn't sign up, just ignore this email."
        )
        send_email(subject, body, email, config)
        st.sidebar.success("✅ Check your inbox for a verification link.")

# This is the verification handler
def verify_email(token, email, config):
    #print(unverified_users)
    for user_email, user_data in unverified_users.items():
        if user_email == email and user_data['verification_token'] == token:
            # Move user from unverified to verified list (add to config.yaml)
            config['credentials']['usernames'][user_data['username']] = {
                'email': user_email,
                'name': user_data['name'],
                'password': user_data['password']  # Assuming password is already hashed
            }
            # Update config.yaml
            update_yaml(config)
            st.toast('User Verified Successfully.', icon="✅")
            # Remove user from unverified users
            del unverified_users[user_email]
            return True
        else:
            return False


# Streamlit function to check URL parameters
def verify_page(config=load_auth_config()):
    """
    Call this in app.py when you detect `token` and `email` in URL params.
    Moves user from `unverified_users` into `config.yaml`.
    """
    tp = st.experimental_get_query_params()
    token = tp.get("token", [None])[0]
    email = tp.get("email", [None])[0]
    if token and email and email in unverified_users:
        entry = unverified_users[email]
        if entry["token"] == token:
            # commit to config
            config["credentials"]["usernames"][entry["username"]] = {
                "email": email,
                "name": entry["name"],
                "password": entry["password"]
            }
            update_yaml(config)
            st.success("🎉 Email verified! You can now log in.")
            del unverified_users[email]
            return True
    st.error("Invalid or expired verification link.")
    return False

# Reset password function
def reset_password(authenticator, config):
    """
    Allows a logged‑in user to change their password in‑app.
    """
    current_user = st.session_state.get("username")
    if current_user:
        try:
            ok = authenticator.reset_password(
                current_user,
                location="sidebar",
                clear_on_submit=True,
                key="ResetPasswdWidget"
            )
            if ok:
                update_yaml(config)  # write new hash back to file
                st.sidebar.success("🔑 Password updated!")
        except Exception as e:
            st.sidebar.error(f"Reset failed: {e}")


# ——————————————————————————————————————————————
# PASSWORD RESET & USERNAME RECOVERY (SMTP)
# ——————————————————————————————————————————————

def forgot_password(authenticator, config):
    """
    Renders the widget, then emails the new random password.
    """
    user, email, new_pw = authenticator.forgot_password(
        location="sidebar",
        captcha=True,
        send_email=False,      # we will email ourselves
        clear_on_submit=True,
        key="ForgotPasswdWidget"
    )
    if user:
        # email via your SMTP helper
        send_reset_password_email(user, new_pw, email, config)
        st.sidebar.success(f"✅ New password sent to {email}")

# Forgot username function
def forgot_username(authenticator, config):
    """
    Renders the widget, then emails the recovered username.
    """
    user, email = authenticator.forgot_username(
        location="sidebar",
        captcha=True,
        send_email=False,
        clear_on_submit=True,
        key="ForgotUserWidget"
    )
    if user:
        send_forgot_username_email(user, user, email, config)
        st.sidebar.success(f"✅ Username sent to {email}")

# Forgot username function
def update_user(authenticator, config):
    """
    Allows a logged‑in user to update their name/email.
    """
    current_user = st.session_state.get("username")
    if current_user:
        try:
            ok = authenticator.update_user_details(
                current_user,
                location="sidebar",
                clear_on_submit=True,
                key="UpdateUserWidget"
            )
            if ok:
                update_yaml(config)
                st.sidebar.success("✅ Profile updated!")
        except Exception as e:
            st.sidebar.error(f"Update failed: {e}")

# Load environment variables from .env file
load_dotenv()
# Initialize Elasticsearch with authentication
es = Elasticsearch(
    hosts=[os.getenv('ES_HOST_URL')],
    basic_auth=(os.getenv('ES_USERNAME'), os.getenv('ES_PASSWORD')),
    verify_certs=False
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

def append_date_time_to_prompt(prompt_content, append_date_time, name):
    """Append date and time information to the selected prompt content if enabled."""
    if prompt_content is None:
        prompt_content = ""  # Initialize as an empty string if None

    if append_date_time:
        now = datetime.now().astimezone()
        current_time = now.strftime("%H:%M:%S")
        current_date = now.strftime("%Y-%m-%d")
        current_day = now.strftime("%A")
        current_timezone = now.tzname()

        date_time_info = f"The current time is: {current_time}, date: {current_date}, day: {current_day}, timezone: {current_timezone}, Username: {name}"
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
        st.toast("Failed to read the PDF file. Please ensure the file is a valid PDF.", icon="⚠️")
        st.toast(f"Error details: {str(e)}", icon="⚠️")
        return None

    except Exception as e:
        st.toast("An unexpected error occurred while processing the PDF file.", icon="⚠️")
        st.toast(f"Error details: {str(e)}", icon="⚠️")
        return None


def extract_text_from_image(uploaded_file, languages):
    """Extracts text from an image file using OCR."""
    if languages:
        selected_codes = languages
        reader = easyocr.Reader(selected_codes, model_storage_directory=EASYOCR_MODELS_PATH)
        image = Image.open(uploaded_file)
        result = reader.readtext(np.array(image), detail=0)
        text = "\n".join(result)
        return text
    else:
        st.toast("Please select at least one language for OCR.", icon="⚠️")
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
        st.toast("Unsupported file type.", icon="⚠️")
        return None

def search_elasticsearch(query, index_name):
    query_embedding = model.encode(query).tolist()

    try:
        query_es = {
            "query_vector": query_embedding,
            "field": "embedding",
            "k": num_results,
            "num_candidates": num_candidates
        }
        response = es.search(index=index_name, knn=query_es, min_score=min_score)
        hits = response['hits']['hits']
        return hits
    except exceptions.RequestError as e:
        st.toast(f"Request Error: {e.info}", icon="⚠️")
    except exceptions.ConnectionError as e:
        st.toast(f"Connection Error: {e.errors}", icon="⚠️")
    except Exception as e:
        st.toast(f"General Error: {str(e)}", icon="⚠️")

def format_timestamp(iso_timestamp):
    """Format the ISO timestamp to a human-readable format."""
    dt = datetime.fromisoformat(iso_timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def hybrid_search(query, index_name, alpha=0.5, num_results=10):
    """Perform hybrid search combining BM25 and neural reranking."""
    query_embedding = model.encode(query).tolist()

    # Step 1: Retrieve BM25 Results
    bm25_query = {
        "query": {
            "match": {
                "text": query
            }
        },
        "size": num_results
    }
    bm25_results = es.search(index=index_name, body=bm25_query)["hits"]["hits"]

    # Step 2: Retrieve Dense Vector (Neural) Results
    knn_query = {
        "query_vector": query_embedding,
        "field": "embedding",
        "k": num_results,
        "num_candidates": num_candidates
    }
    knn_results = es.search(index=index_name, knn=knn_query)["hits"]["hits"]

    # Step 3: Combine Scores using Weighted Sum
    hybrid_results = {}

    for hit in bm25_results:
        doc_id = hit["_id"]
        bm25_score = hit["_score"]
        source = hit["_source"]
        document_name = source.get("document_name", "Unknown Document")  # Get actual document name
        timestamp = source.get("timestamp", None)
        formatted_timestamp = format_timestamp(timestamp) if timestamp else "Unknown"

        hybrid_results[doc_id] = {
            "bm25_score": bm25_score,
            "neural_score": 0,
            "text": source["text"],
            "document_name": document_name,
            "timestamp": formatted_timestamp
        }

    for hit in knn_results:
        doc_id = hit["_id"]
        source = hit["_source"]
        neural_score = 1 - cosine(query_embedding, source["embedding"])  # Cosine similarity
        document_name = source.get("document_name", "Unknown Document")
        timestamp = source.get("timestamp", None)
        formatted_timestamp = format_timestamp(timestamp) if timestamp else "Unknown"

        if doc_id in hybrid_results:
            hybrid_results[doc_id]["neural_score"] = neural_score
        else:
            hybrid_results[doc_id] = {
                "bm25_score": 0,
                "neural_score": neural_score,
                "text": source["text"],
                "document_name": document_name,
                "timestamp": formatted_timestamp
            }

    # Step 4: Normalize Scores
    min_bm25 = min([r["bm25_score"] for r in hybrid_results.values()], default=0)
    max_bm25 = max([r["bm25_score"] for r in hybrid_results.values()], default=1)

    min_neural = min([r["neural_score"] for r in hybrid_results.values()], default=0)
    max_neural = max([r["neural_score"] for r in hybrid_results.values()], default=1)

    for doc_id in hybrid_results:
        hybrid_results[doc_id]["bm25_score"] = (hybrid_results[doc_id]["bm25_score"] - min_bm25) / (max_bm25 - min_bm25 + 1e-5)
        hybrid_results[doc_id]["neural_score"] = (hybrid_results[doc_id]["neural_score"] - min_neural) / (max_neural - min_neural + 1e-5)

        # Final hybrid score (weighted sum)
        hybrid_results[doc_id]["final_score"] = alpha * hybrid_results[doc_id]["bm25_score"] + (1 - alpha) * hybrid_results[doc_id]["neural_score"]

    # Step 5: Return Sorted Results as List of Dictionaries
    sorted_results = sorted(hybrid_results.items(), key=lambda x: x[1]["final_score"], reverse=True)

    return [{"doc_id": doc[0], **doc[1]} for doc in sorted_results[:num_results]]

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


def create_index(index_name):
    field_name = "embedding"
    similarity_type = os.getenv("SIMILARITY_TYPE", "cosine")
    default_dims = int(os.getenv("DEFAULT_DIMS", 1024))

    if not es.indices.exists(index=index_name):
        mappings = {
            "properties": {
                field_name: {
                    "type": "dense_vector",
                    "dims": default_dims,
                    "index": "true",
                    "similarity": similarity_type,
                }
            }
        }
        es.indices.create(index=index_name, body={"mappings": mappings})
        st.success(
            f"Index '{index_name}' with {default_dims} dimensions and similarity '{similarity_type}' created successfully")
    else:
        st.error("Index already exists")



def delete_index(index_name):
    if not es.indices.exists(index=index_name):
        st.error("Index not found")
        return
    es.indices.delete(index=index_name)
    st.success(f"Index '{index_name}' deleted successfully")

def generate_unique_id(text):
    hash_object = hashlib.sha256()
    hash_object.update(text.encode('utf-8'))
    unique_id = hash_object.hexdigest()
    return unique_id

def index_text(index_name, text, document_name, total_files, file_number):
    clean_text_content = clean_text(text)
    local_time = datetime.now()
    utc_time = local_time.astimezone(pytz.utc)
    timestamp = utc_time.isoformat()
    chunks = split_text_semantically(clean_text_content)
    total_chunks = len(chunks)
    progress_text = f"Indexing document {file_number}/{total_files}. Please wait..."
    my_bar = st.progress(0, text=progress_text)

    for i, chunk in enumerate(chunks):
        if chunk:
            doc_id = generate_unique_id(chunk)
            embedding = model.encode(chunk).tolist()
            body = {
                "text": chunk,
                "embedding": embedding,
                "document_name": document_name,
                "timestamp": timestamp
            }
            es.index(index=index_name, id=doc_id, body=body)
            my_bar.progress((i + 1) / total_chunks, text=progress_text)

    my_bar.empty()


def list_documents(index_name):
    query = {
        "size": 10000,
        "_source": ["document_name", "timestamp"],
        "query": {
            "match_all": {}
        }
    }
    response = es.search(index=index_name, body=query)

    document_data = {}
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        doc_name = source.get("document_name", "Unknown Document")
        timestamp = source.get("timestamp", "No Timestamp")

        if doc_name in document_data:
            document_data[doc_name]["number_of_chunks"] += 1
        else:
            document_data[doc_name] = {
                "document_name": doc_name,
                "number_of_chunks": 1,
                "timestamp": timestamp
            }

    if not document_data:
        return pd.DataFrame(columns=["document_name", "number_of_chunks", "date_time_added"])

    document_df = pd.DataFrame(document_data.values())
    document_df["timestamp"] = pd.to_datetime(document_df["timestamp"], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    document_df = document_df.rename(columns={"timestamp": "date_time_added"})

    return document_df

# ——————————————————————————————————————————————
# LOGIN / LOGOUT & HIGH‑LEVEL MENU
# ——————————————————————————————————————————————
def setup_authentication_menu():
    config = load_auth_config()

    # ─── Initialize session‐state keys ──────────────────────
    if "logout" not in st.session_state:
        st.session_state["logout"] = False
    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = None

    auth_status = st.session_state["authentication_status"]

    # ─── If already logged in: show welcome/logout/manage only ───
    if auth_status:
        st.sidebar.markdown(f"**Welcome, {st.session_state['name']}!**")

        # Logout button
        authenticator.logout("Logout", "sidebar", key="LogoutWidget")
        st.session_state["logout"] = True

        # Manage Account toggle
        if st.sidebar.toggle("Manage Account", value=False):
            reset_password(authenticator, config)
            update_user(authenticator, config)

        # Return the logged‑in user’s name for app.py
        return st.session_state["name"]

    # ─── Not logged in yet: show the auth menu ───────────────────
    choice = st.sidebar.radio(
        "🔐 Authentication",
        ["Login", "Register", "Forgot Password", "Forgot Username"],
        index=0
    )

    if choice == "Login":
        result = authenticator.login(
            location="sidebar",
            max_concurrent_users=5,
            max_login_attempts=3,
            clear_on_submit=True,
            key="LoginWidget"
        )
        if result:
            name, new_status, username = result
            st.session_state["authentication_status"] = new_status
            if new_status:
                st.session_state["name"] = name
                st.session_state["username"] = username

    elif choice == "Register":
        register_user(authenticator, config)

    elif choice == "Forgot Password":
        forgot_password(authenticator, config)

    else:  # "Forgot Username"
        forgot_username(authenticator, config)

    # ─── After an attempted login, show errors/info ────────────
    auth_status = st.session_state["authentication_status"]
    if auth_status is False:
        st.sidebar.error("❌ Wrong username or password")
    elif auth_status is None:
        st.sidebar.info("ℹ️ Please log in or register")

    return None
