import os
import json
import time
import streamlit as st
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError
from dotenv import load_dotenv
from utils import (
    clean_text,
    split_text_semantically,
    extract_text_from_file,
    display_debug_info,
    load_advanced_settings,
    display_advanced_settings,
    append_date_time_to_prompt,
    toggle_display_metrics,
    refresh_metrics,
    get_model_health,
    clear_chat_history,
    setup_authentication_menu
)

# Load environment variables from .env file
load_dotenv()

# Access variables from the environment
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
SUMMARIZER_INSTRUCTION = os.getenv("SUMMARIZER_INSTRUCTION", "You're a Text Summarizer...")  # Default instruction if not set

st.set_page_config(layout="wide")
st.title("📄 Document Summarizer")

# Call the clear_chat_history function whenever the page loads
clear_chat_history()


# Setup Authenticator
name = setup_authentication_menu()

if st.session_state["authentication_status"]:
    # Initialize the OpenAI client
    client = OpenAI(base_url=f"{OPENAI_BASE_URL}/v1", api_key=OPENAI_API_KEY)

    # Initialize session state attributes specifically for this page
    st.session_state["messages"] = []  # Reset messages when visiting this page
    selected_prompt_content = SUMMARIZER_INSTRUCTION  # Set from .env variable for this page

    # Initialize session state variables
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = OPENAI_MODEL

    if "debug" not in st.session_state:
        st.session_state["debug"] = False

    # Load OCR languages from the JSON file
    with open('ocr_languages.json', 'r') as file:
        ocr_languages = json.load(file)

    language_options = [lang["language"] for lang in ocr_languages]
    language_codes = {lang["language"]: lang["code"] for lang in ocr_languages}

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
    if settings_visible:
        advanced_settings = display_advanced_settings(settings_visible)
    else:
        advanced_settings = load_advanced_settings()

    # Toggle to enable/disable the display of CPU, Memory, and Model Health metrics
    display_metrics = toggle_display_metrics()

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
                                stop=json.loads(advanced_settings["stop"]) if advanced_settings["stop"] else None,
                                stream=advanced_settings["stream_t"],
                            )
                            response = st.write_stream(stream)
                        st.session_state["messages"].append({"role": "assistant", "content": response})
                        # Display debug information if debug mode is enabled
                        if st.session_state.get("debug", False):
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

    # Refresh CPU, Memory, and Health status if metrics display is enabled
    refresh_metrics(display_metrics)

else:
    st.markdown(':red[Please Login or Register]')
