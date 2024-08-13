import json
import psutil
import requests
import streamlit as st
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError
from datetime import datetime
from io import StringIO
import pytz
import os
import fitz  # PyMuPDF
from docx import Document
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

# Load environment variables from .env file
load_dotenv()

# Access variables from the environment
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

st.set_page_config(layout="wide")
st.title("💬 LLamafile Chatbot")

# Initialize the OpenAI client
client = OpenAI(base_url=f"{OPENAI_BASE_URL}/v1", api_key=OPENAI_API_KEY)

# Function to get model health status
def get_model_health():
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

# Function to extract text from uploaded file using PyMuPDF (fitz)
def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.type

    # Handle plain text files separately
    if file_type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")

    # Handle PDF, XPS, EPUB, MOBI, FB2, CBZ, SVG, DOCX, XLSX, PPTX, HWPX, Images
    elif file_type in ["application/pdf", "application/epub+zip", "application/x-mobipocket-ebook", "application/vnd.amazon.ebook", "application/fb2+xml", "application/vnd.comicbook+zip", "image/svg+xml", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        document = fitz.open(stream=uploaded_file.read(), filetype=file_type.split('/')[-1])
        text = ""
        for page in document:
            text += page.get_text()
        return text
    elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
        document = fitz.open(stream=uploaded_file.read(), filetype="image")
        text = ""
        for page in document:
            text += page.get_text("text")
        return text
    else:
        return "Unsupported file type."

# Create rows and containers for CPU, Memory, and Health status
row1 = st.columns(3)

cpu_container = row1[0].container()
mem_container = row1[1].container()
health_container = row1[2].container()

# Initialize session state variables
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = OPENAI_MODEL

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "system_instruction" not in st.session_state:
    st.session_state["system_instruction"] = ""

# Load system prompts from JSON file
with open('system_prompts.json') as f:
    system_prompts = json.load(f)

# Sidebar settings
st.sidebar.title("Settings")

# Dropdown for selecting system instruction prompt
prompt_names = [prompt['name'] for prompt in system_prompts]
default_prompt = next((prompt for prompt in system_prompts if prompt['name'] == "Assistant"), system_prompts[0])
selected_prompt_name = st.sidebar.selectbox("Select System Instruction Prompt", prompt_names, index=prompt_names.index("Assistant"))

# Retrieve the selected prompt content and description
selected_prompt = next((prompt for prompt in system_prompts if prompt['name'] == selected_prompt_name), {})
selected_prompt_content = selected_prompt.get("prompt", "")
selected_prompt_description = selected_prompt.get("description", "")

# Display the selected prompt description as a caption
st.caption(f"🚀 {selected_prompt_description}")

# Setting to enable/disable appending date and time to the prompt
append_date_time = st.sidebar.toggle("Append Date and Time to Prompt", value=True)

if append_date_time:
    # Get the current time, date, day, and timezone
    now = datetime.now(pytz.timezone('UTC')).astimezone()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%Y-%m-%d")
    current_day = now.strftime("%A")
    current_timezone = now.tzname()

    # Append date and time information to the selected prompt content
    date_time_info = f"The current time is: {current_time}, date: {current_date}, day: {current_day}, timezone: {current_timezone}."
    selected_prompt_content += "\n" + date_time_info

# Store the selected system instruction prompt in session state
st.session_state["system_instruction"] = selected_prompt_content

# Toggle for advanced settings visibility
settings_visible = st.sidebar.toggle("Show/Hide Advanced Settings", value=False)

# Default values for advanced settings when hidden
temperature = 0.8
top_p = 0.95
frequency_penalty = 0.0
presence_penalty = 0.0
seed = -1
logit_bias = "{}"

if settings_visible:
    # Advanced settings sliders and inputs
    temperature = st.sidebar.slider(
        "Temperature (Adjust the randomness of the generated text)",
        min_value=0.0, max_value=1.0, value=0.8, step=0.1, help="Default: 0.8"
    )
    top_p = st.sidebar.slider(
        "Top P (Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P)",
        min_value=0.0, max_value=1.0, value=0.95, step=0.01, help="Default: 0.95"
    )
    frequency_penalty = st.sidebar.slider(
        "Frequency Penalty (Control the repetition of token sequences in the generated text)",
        min_value=0.0, max_value=2.0, value=0.0, step=0.1, help="Default: 0.0"
    )
    presence_penalty = st.sidebar.slider(
        "Presence Penalty (Repeat alpha presence penalty)",
        min_value=0.0, max_value=2.0, value=0.0, step=0.1, help="Default: 0.0"
    )
    seed = st.sidebar.number_input(
        "Seed (Set the random number generator (RNG) seed)",
        value=-1, help="Default: -1 (random seed)"
    )
    logit_bias = st.sidebar.text_area(
        "Logit Bias (Modify the likelihood of a token appearing in the generated text completion. JSON format)",
        "{}", help='Default: []'
    )

# Modify chat input based on the selected prompt
chat_input_label = "How can I help you?"

if selected_prompt_name == "Text Summarizer":
    chat_input_label = "Please paste the text or upload the file to get a summary"
    uploaded_file = st.file_uploader("Upload a text, PDF, or Word document", type=["txt", "pdf", "docx", "epub", "mobi", "fb2", "cbz", "svg", "png", "jpeg", "jpg"])
    if uploaded_file is not None:
        file_text = extract_text_from_file(uploaded_file)
        st.session_state.messages.append({"role": "user", "content": file_text})
        with st.chat_message("user"):
            st.markdown(file_text)

        # Automatically trigger the assistant to process the uploaded text
        try:
            with st.spinner("Thinking..."):
                with st.chat_message("assistant"):
                    messages = [{"role": "system", "content": st.session_state.system_instruction}] if st.session_state.system_instruction else []
                    messages += [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]
                    stream = client.chat.completions.create(
                        model=st.session_state["openai_model"],
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        seed=seed,
                        logit_bias=eval(logit_bias),  # Converting JSON string to dict
                        stream=True,
                    )
                    response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})
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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(chat_input_label):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.spinner("Thinking..."):
            with st.chat_message("assistant"):
                messages = [{"role": "system", "content": st.session_state.system_instruction}] if st.session_state.system_instruction else []
                messages += [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
                stream = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    seed=seed,
                    logit_bias=eval(logit_bias),  # Converting JSON string to dict
                    stream=True,
                )
                response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
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

# Automatically refresh the CPU, Memory, and Health status every 5 seconds
count = st_autorefresh(interval=5000, key="status_refresh")

# Update CPU, Memory, and Health status in real-time
cpu_usage = psutil.cpu_percent(interval=1)
mem_usage = psutil.virtual_memory().percent
model_health = get_model_health()

# Display the metrics inside their respective containers
cpu_container.metric(label="CPU Usage", value=f"{cpu_usage}%")
mem_container.metric(label="Memory Usage", value=f"{mem_usage}%")
health_container.metric(label="Model Health", value=model_health)
