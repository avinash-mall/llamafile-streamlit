import streamlit as st
import json
import os
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError
from elasticsearch import Elasticsearch, exceptions, ElasticsearchWarning
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from utils import (
    append_date_time_to_prompt,
    load_advanced_settings,
    display_advanced_settings,
    refresh_metrics,
    toggle_display_metrics,
    search_elasticsearch,
    display_debug_info,
    clear_chat_history,
    setup_authentication_menu
)

# Suppress Elasticsearch system indices warnings
warnings.filterwarnings("ignore", category=ElasticsearchWarning)

es = Elasticsearch(
    hosts=[os.getenv('ES_HOST_URL')],
    basic_auth=(os.getenv('ES_USERNAME'), os.getenv('ES_PASSWORD')),
    verify_certs=False
)
# Function to format timestamp
def format_timestamp(iso_timestamp):
    dt = datetime.fromisoformat(iso_timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")
# Load environment variables
load_dotenv()

# Call the clear_chat_history function whenever the page loads
clear_chat_history()

# Initialize SentenceTransformer model
model = SentenceTransformer(os.getenv("MODEL_PATH"))

# Access OpenAI API environment variables
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
selected_prompt_content = os.getenv("INSTRUCTION_PROMPT")

ADMIN_EMAIL = os.getenv('ADMIN_EMAIL')

# Set page configuration
st.set_page_config(page_icon="💬", layout="wide")
# Title of the app
st.title("📚 Document Chatbot")

# Setup Authenticator
name, config = setup_authentication_menu()

if st.session_state["authentication_status"]:
    # Initialize the OpenAI client
    client = OpenAI(base_url=f"{OPENAI_BASE_URL}/v1", api_key=OPENAI_API_KEY)

    user_email = config['credentials']['usernames'][st.session_state["username"]]['email']
    user_index = f"user_{user_email.replace('@', '_').replace('.', '_')}"

    # Initialize session state variables if they don't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    else:
        # Display previous messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = OPENAI_MODEL

    # Sidebar settings
    st.sidebar.title("Settings")

    # Add a button to clear the chat history
    if st.sidebar.button("Clear Chat History"):
        clear_chat_history()
        st.success("Chat history cleared.")

    # Setting to enable/disable appending date and time to the prompt
    append_date_time = st.sidebar.toggle("Append Date and Time to Prompt", value=True)
    selected_prompt_content = append_date_time_to_prompt(selected_prompt_content, append_date_time, name)

    # Store the selected system instruction prompt in session state
    st.session_state["system_instruction"] = selected_prompt_content

    # Toggle to show/hide advanced settings
    settings_visible = st.sidebar.toggle("Show/Hide Advanced Settings", value=False)
    advanced_settings = display_advanced_settings(settings_visible) if settings_visible else load_advanced_settings()

    # Toggle to enable/disable the display of CPU, Memory, and Model Health metrics
    display_metrics = toggle_display_metrics()

    # Dropdown for selecting Elasticsearch index
    if user_email == ADMIN_EMAIL:
        indexes = [index for index in es.indices.get_alias(index="*").keys() if not index.startswith('.')]
        selected_index = st.sidebar.selectbox("Select Elasticsearch Index", options=indexes)
    else:
        selected_index = user_index

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
                    # Initialize variables to avoid errors if no hits are found
                    unique_document_names = []
                    unique_timestamps = []
                    if hits:
                        context_chunks = [hit['_source']['text'] for hit in hits]
                        document_names = [hit['_source']['document_name'] for hit in hits]
                        timestamps = [hit['_source']['timestamp'] for hit in hits]
                        # Create the DataFrame
                        df = pd.DataFrame({
                            'Document Name': document_names,
                            'ISO Timestamp': timestamps,
                            'Context Chunks': context_chunks
                        })
                        # Apply the format_timestamp function on 'ISO Timestamp' and store the result in a new column 'Formatted Timestamp'
                        df['Timestamp'] = df['ISO Timestamp'].apply(format_timestamp)
                        # Remove duplicates based on 'Document Name' and 'Formatted Timestamp'
                        df_unique = df.drop_duplicates(subset=['Document Name', 'Timestamp'])
                        # Optionally drop the original 'ISO Timestamp' if no longer needed
                        df_unique = df_unique.drop(columns=['ISO Timestamp'])
                        df_unique = df_unique.drop(columns=['Context Chunks'])
                    #context = "\n".join(context_chunks) if context_chunks else ""

                    # Prepare the full conversation history
                    messages = [{"role": message["role"], "content": message["content"]}
                                for message in st.session_state.messages]

                    # Adding the structured context to the messages
                    if context_chunks:
                        structured_context = "\n".join(
                            [
                                f"Document Name: {row['Document Name']}\nTimestamp: {row['Timestamp']}\nContext: {row['Context Chunks']}"
                                for _, row in df.iterrows()
                            ]
                        )
                        messages.append({"role": "assistant", "content": f"{structured_context}"})

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
                    st.header("References", divider=True)
                    # Format the reference and added on information
                    st.dataframe(df_unique)
                    st.header("Answer", divider=True)
                    # Display the information before the LLM response
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


else:
    st.markdown(':red[Please Login or Register]')
