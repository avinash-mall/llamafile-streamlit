import streamlit as st
import json
from datetime import datetime
import os
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError
from utils import (
    get_model_health,
    append_date_time_to_prompt,
    load_advanced_settings,
    display_advanced_settings,
    refresh_metrics,
    toggle_display_metrics,
    clear_chat_history
)

# Load environment variables
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

# Add a button to clear the chat history
if st.sidebar.button("Clear Chat History"):
    clear_chat_history()
    st.success("Chat history cleared.")

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
selected_prompt_content = append_date_time_to_prompt(selected_prompt_content, append_date_time)

# Store the selected system instruction prompt in session state
st.session_state["system_instruction"] = selected_prompt_content

# Initialize session state variables
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = OPENAI_MODEL

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "debug" not in st.session_state:
    st.session_state["debug"] = False


# Load advanced settings
settings_visible = st.sidebar.toggle("Show/Hide Advanced Settings", value=False)
advanced_settings = display_advanced_settings(settings_visible) if settings_visible else load_advanced_settings()

# Toggle to enable/disable the display of CPU, Memory, and Model Health metrics
display_metrics = toggle_display_metrics()

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
                    temperature=advanced_settings["temperature"],
                    top_p=advanced_settings["top_p"],
                    frequency_penalty=advanced_settings["frequency_penalty"],
                    presence_penalty=advanced_settings["presence_penalty"],
                    seed=advanced_settings["seed"],
                    logit_bias=eval(advanced_settings["logit_bias"]),
                    logprobs=advanced_settings["logprobs"],
                    top_logprobs=advanced_settings["top_logprobs"] if advanced_settings["logprobs"] else None,
                    max_tokens=advanced_settings["max_tokens"],
                    n=advanced_settings["n"],
                    stop=json.loads(advanced_settings["stop"]),
                    stream=advanced_settings["stream_t"],
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
refresh_metrics(display_metrics)
