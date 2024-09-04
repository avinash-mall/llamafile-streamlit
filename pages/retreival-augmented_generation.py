import streamlit as st
import json
import os
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError
from elasticsearch import Elasticsearch, exceptions, ElasticsearchWarning
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from datetime import datetime

from utils import (
    append_date_time_to_prompt,
    load_advanced_settings,
    display_advanced_settings,
    refresh_metrics,
    toggle_display_metrics,
    search_elasticsearch,
    display_debug_info
)

es = Elasticsearch(
    hosts=[os.getenv('ES_HOST_URL')],
    basic_auth=(os.getenv('ES_USERNAME'), os.getenv('ES_PASSWORD'))
)
# Function to format timestamp
def format_timestamp(iso_timestamp):
    dt = datetime.fromisoformat(iso_timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")
# Load environment variables
load_dotenv()

# Initialize SentenceTransformer model
model = SentenceTransformer(os.getenv("MODEL_PATH"))

# Access OpenAI API environment variables
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
instruction_prompt = os.getenv("INSTRUCTION_PROMPT")

# Set page configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")

# Initialize the OpenAI client
client = OpenAI(base_url=f"{OPENAI_BASE_URL}/v1", api_key=OPENAI_API_KEY)

# Initialize session state variables if they don't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = OPENAI_MODEL

if "system_instruction" not in st.session_state:
    st.session_state["system_instruction"] = instruction_prompt

# Sidebar settings
st.sidebar.title("Settings")

# Toggle for appending date and time to the prompt
append_date_time = st.sidebar.toggle("Append Date and Time to Prompt", value=True)
st.session_state["system_instruction"] = append_date_time_to_prompt(st.session_state["system_instruction"], append_date_time)

# Toggle to show/hide advanced settings
settings_visible = st.sidebar.toggle("Show/Hide Advanced Settings", value=False)
advanced_settings = display_advanced_settings(settings_visible) if settings_visible else load_advanced_settings()

# Toggle to enable/disable the display of CPU, Memory, and Model Health metrics
display_metrics = toggle_display_metrics()

# Dropdown for selecting Elasticsearch index
indexes = [index for index in es.indices.get_alias(index="*").keys() if not index.startswith('.')]
selected_index = st.sidebar.selectbox("Select Elasticsearch Index", options=indexes)

# Title of the app
st.title("💬 RAG Chatbot")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How can I help you?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.spinner("Thinking..."):
            with st.chat_message("assistant") as assistant_message:
                # Retrieve context from Elasticsearch
                hits = search_elasticsearch(prompt, selected_index)
                context_chunks = []
                document_names = []
                timestamps = []
                if hits:
                    context_chunks = [hit['_source']['text'] for hit in hits]
                    document_names = [hit['_source']['document_name'] for hit in hits]
                    timestamps = [hit['_source']['timestamp'] for hit in hits]
                    # Remove duplicates (in case the same document appears multiple times)
                    unique_document_names = list(set(document_names))
                    unique_timestamps = list(set(timestamps))

                # Format timestamps for better readability
                unique_timestamps = [format_timestamp(ts) for ts in unique_timestamps]

                context = "\n".join(context_chunks)

                # Prepare the full conversation history
                messages = [{"role": message["role"], "content": message["content"]}
                            for message in st.session_state.messages]

                # Adding the structured context to the messages
                if context_chunks:
                    structured_context = "\n".join(
                        [f"Document Name: {doc_name}\nTimestamp: {timestamp}\nContext: {chunk}"
                         for chunk, doc_name, timestamp in zip(context_chunks, document_names, timestamps)]
                    )
                    messages.append(
                        {"role": "assistant", "content": f"{structured_context}"})

                # Include the new user prompt
                messages.append({"role": "user", "content": prompt})

                # Call the LLM API with the full conversation and new context
                # noinspection PyTypeChecker
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

                # Format the reference and added on information
                reference_info = "**Reference**: " + ", ".join(unique_document_names)
                added_on_info = "**Added on**: " + ", ".join(unique_timestamps)

                # Display the information before the LLM response
                st.markdown(reference_info)
                st.markdown(added_on_info)

                response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Display Debug Information
            display_debug_info(summary=response, prompt=prompt, messages=messages)


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

# Automatically refresh CPU, Memory, and Health status every 5 seconds if metrics display is enabled
refresh_metrics(display_metrics)
