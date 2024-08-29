import json
import os
import psutil
import requests
import streamlit as st
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError
from datetime import datetime
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh
from utils import get_model_health

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

# Load system prompts from JSON file
with open('system_prompts.json') as f:
    system_prompts = json.load(f)

# Load settings from settings.json file
with open('settings.json') as f:
    settings = json.load(f)

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
    now = datetime.now().astimezone()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%Y-%m-%d")
    current_day = now.strftime("%A")
    current_timezone = now.tzname()

    # Append date and time information to the selected prompt content
    date_time_info = f"The current time is: {current_time}, date: {current_date}, day: {current_day}, timezone: {current_timezone}."
    selected_prompt_content += "\n" + date_time_info

# Store the selected system instruction prompt in session state
st.session_state["system_instruction"] = selected_prompt_content

# Initialize session state variables
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = OPENAI_MODEL

if "messages" not in st.session_state:
    st.session_state["messages"] = []

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

# Toggle to enable/disable the display of CPU, Memory, and Model Health metrics
display_metrics = st.sidebar.toggle("Display CPU, Memory, and Model Health Metrics", value=False, help="Enable to show CPU, memory, and model health metrics")

# Chat input
if prompt := st.chat_input("How can I help you?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.spinner("Thinking..."):
            with st.chat_message("assistant"):
                messages = [{"role": "system", "content": st.session_state.system_instruction}] if st.session_state.system_instruction else []
                messages += [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                stream = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    seed=seed,
                    logit_bias=eval(logit_bias),  # Converting JSON string to dict
                    logprobs=logprobs,
                    top_logprobs=top_logprobs if logprobs else None,
                    max_tokens=max_tokens,
                    n=n,
                    stop=json.loads(stop),
                    stream=stream_t,
                )
                response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Display debug information if debug mode is enabled
            if st.session_state.get("debug", False):
                st.write(f"Debug Info: Messages: {st.session_state['messages']}")
    except APIConnectionError as e:
        st.error("The server could not be reached.")
        st.error(f"Details: {e.__cause__}")
    except RateLimitError as e:
        st.error("Rate limit exceeded; please try again later.")
    except APIStatusError as e:
        st.error(f"An error occurred: {e.status_code}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

# Automatically refresh CPU, Memory, and Health status every 5 seconds if metrics display is enabled
if display_metrics:
    count = st_autorefresh(interval=5000, key="status_refresh")

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
