import json
import os
import re
import unicodedata
import warnings
from datetime import datetime
from urllib.parse import urlencode

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
import yaml
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
import smtplib
from email.message import EmailMessage
import secrets
from urllib.parse import parse_qs

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

# Load config.yaml
def load_auth_config():
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
    return config


# Update config.yaml after any changes (e.g., registration or password reset)
def update_yaml(config):
    with open('config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


# Authentication setup
def setup_authenticator():
    config = load_auth_config()
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['pre-authorized']
    )
    return authenticator, config


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


# Register function
# def register_user(authenticator, config):
#     try:
#         email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(
#             location='main',
#             pre_authorization=False,
#             clear_on_submit=True,
#             key='RegisterWidget'
#         )
#         if email_of_registered_user:
#             st.toast('User registered successfully', icon="✅")
#             update_yaml(config)
#     except Exception as e:
#         st.toast(e, icon="⚠️")


# Temporary storage for unverified users (can be a dictionary or external storage)
unverified_users = {}


def register_user(authenticator, config):
    try:
        email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(
            location='main',
            pre_authorization=False,
            clear_on_submit=True,
            key='RegisterWidget'
        )
        if email_of_registered_user:
            st.toast('User registration initiated. Please check your email to verify.', icon="✅")
            # Generate a verification token
            verification_token = generate_verification_token()
            # Store the user temporarily with the token
            unverified_users[email_of_registered_user] = {
                'username': username_of_registered_user,
                'name': name_of_registered_user,
                'password': config['credentials']['usernames'][username_of_registered_user]['password'],
                # Assuming password is hashed
                'verification_token': verification_token
            }

            # Load the domain from config.yaml
            app_domain = config['app']['domain']

            # URL encode the parameters (token and email)
            query_params = urlencode({
                'token': verification_token,
                'email': email_of_registered_user
            })

            # Create the verification link, now including the page parameter
            verification_link = f"{app_domain}?&{query_params}"

            # Send verification email
            subject = "Email Verification"
            body = f"Hello {name_of_registered_user},\n\nPlease verify your email by clicking the link below:\n{verification_link}"
            send_email(subject, body, email_of_registered_user, config)

    except Exception as e:
        st.toast(e, icon="⚠️")


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
    if st.query_params.token or st.query_params.email:
        token = st.query_params.token  # Extract token
        email = st.query_params.email  # Extract email
        print(token, email)
        # Call the verification function
        if verify_email(token, email, config):
            st.success("Your email has been verified successfully. You can now log in.")
        else:
            st.error("Invalid or expired verification link.")
    else:
        st.error("Invalid request. Token or email is missing.")


# Reset password function
def reset_password(authenticator, username, config):
    try:
        if authenticator.reset_password(username, location='main', clear_on_submit=True, key='ResetPasswdWidget'):
            st.toast('Password modified successfully', icon="✅")
            update_yaml(config)
    except Exception as e:
        st.toast(e, icon="⚠️")


# Forgot password function
def forgot_password(authenticator, config):
    try:
        username_of_forgotten_password, email_of_forgotten_password, new_random_password = authenticator.forgot_password(location='main', captcha=True, key='ForgotPasswdWidget')
        if username_of_forgotten_password:
            user_name = config['credentials']['usernames'][username_of_forgotten_password][
                'name']  # Assuming you store the name in the config
            update_yaml(config)
            send_reset_password_email(user_name, new_random_password, email_of_forgotten_password, config)
            st.toast('New password sent securely', icon="✅")
        elif username_of_forgotten_password == False:
            st.toast('Username not found', icon="⚠️")
    except Exception as e:
        st.toast(e, icon="⚠️")

# Forgot username function
def forgot_username(authenticator, config):
    try:
        username_of_forgotten_username, email_of_forgotten_username = authenticator.forgot_username(location='main', captcha=True, key='ForgotUserWidget')
        if username_of_forgotten_username:
            user_name = config['credentials']['usernames'][username_of_forgotten_username]['name']
            update_yaml(config)
            send_forgot_username_email(user_name, username_of_forgotten_username, email_of_forgotten_username, config)
            st.toast('Username sent securely', icon="✅")
        elif username_of_forgotten_username == False:
            st.toast('Email not found', icon="⚠️")
    except Exception as e:
        st.toast(e, icon="⚠️")

# Forgot username function
def update_user(authenticator, config):
    try:
        if authenticator.update_user_details(st.session_state['username'], location='main', clear_on_submit=True, key='UpdateUserWidget'):
            st.toast('Entries updated successfully', icon="✅")
            update_yaml(config)
    except Exception as e:
        st.toast(e, icon="⚠️")


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


def extract_text_from_image(uploaded_file, languages=None):
    """Extracts text from an image file using OCR."""
    if languages:
        selected_codes = [language_codes[lang] for lang in languages]
        reader = easyocr.Reader(selected_codes, model_storage_directory=EASYOCR_MODELS_PATH)
        result = reader.readtext(uploaded_file.getvalue(), detail=0)
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
        st.toast(f"Request Error: {e.info}", icon="⚠️")
    except exceptions.ConnectionError as e:
        st.toast(f"Connection Error: {e.errors}", icon="⚠️")
    except Exception as e:
        st.toast(f"General Error: {str(e)}", icon="⚠️")


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

def setup_authentication_menu():
    # Setup Authenticator
    authenticator, config = setup_authenticator()

    if 'hashed_done' not in st.session_state:
        config = hash_plaintext_passwords(config)
        update_yaml(config)
        st.session_state.hashed_done = True

    col1, col2, col3, col4 = st.columns(4)

    # Login
    with col1:
        with st.popover("Login"):
            name, authentication_status, username = login(authenticator)

    # Handle Authentication Logic
    if st.session_state["authentication_status"]:
        st.markdown(f':thumbsup: Welcome *{name}*')
        with col2:
            # If the user is authenticated
            logout(authenticator)
        with col3:
            # Reset password for authenticated users
            with st.popover("Reset Password"):
                reset_password(authenticator, username, config)
        with col4:
            # Update details for authenticated users
            with st.popover("Update User Details"):
                update_user(authenticator, config)
        return name
    elif st.session_state["authentication_status"] == False:
        st.toast('Incorrect Username or Password', icon="⚠️")
        with col2:
            with st.popover("Register"):
                register_user(authenticator, config)
        with col3:
            with st.popover("Forgot Password"):
                forgot_password(authenticator, config)
    elif st.session_state["authentication_status"] == None:
        st.toast('Please enter your username and password', icon="⚠️")
        with col2:
            # Handle Registration
            with st.popover("Register"):
                register_user(authenticator, config)
        with col3:
            with st.popover("Forgot Password"):
                forgot_password(authenticator, config)
        with col4:
            with st.popover("Forgot Username"):
                forgot_username(authenticator, config)
