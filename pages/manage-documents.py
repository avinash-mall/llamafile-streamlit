import streamlit as st
import os
import hashlib
from datetime import datetime
from elasticsearch import Elasticsearch, ElasticsearchWarning
from dotenv import load_dotenv
import warnings
import pytz
import pandas as pd
from utils import (
    model,
    split_text_semantically,
    clean_text,
    extract_text_from_pdf,
    extract_text_from_file,
    language_options,
    language_codes,
    refresh_metrics,
    display_advanced_settings,
    toggle_display_metrics,
    setup_authentication_menu,
    create_index,
    delete_index,
    list_documents,
    index_text
)
# Suppress Elasticsearch system indices warnings
warnings.filterwarnings("ignore", category=ElasticsearchWarning)

# Load environment variables from .env file
load_dotenv()
ADMIN_EMAIL = os.getenv('ADMIN_EMAIL')

# Initialize Elasticsearch with authentication
es = Elasticsearch(
    hosts=[os.getenv('ES_HOST_URL')],
    basic_auth=(os.getenv('ES_USERNAME'), os.getenv('ES_PASSWORD'))
)

# Set page configuration for a wide layout
st.set_page_config(page_icon="🔧", layout="wide")

# Admin page title
st.title("🔧 Document Management")

# Setup Authenticator
name, config = setup_authentication_menu()

def enhanced_indexing_progress(uploaded_files, extraction_method, selected_languages, user_index):
    """Indexes multiple uploaded files with enhanced progress indicators"""

    total_files = len(uploaded_files)
    global_progress_bar = st.progress(0)  # Global progress for overall process
    global_status = st.empty()  # Display for global status messages

    for file_number, uploaded_file in enumerate(uploaded_files, start=1):
        file_progress = st.progress(0)  # Progress for individual file
        status_message = st.empty()  # Message display for individual file

        # Simulate file processing
        start_time = time.time()

        # Determine the extraction method based on the file type
        file_type = uploaded_file.type
        if file_type == "application/pdf":
            file_text = extract_text_from_pdf(uploaded_file, extraction_method == "Read Using OCR", selected_languages)
        elif file_type.startswith("image/"):
            file_text = extract_text_from_file(uploaded_file, languages=selected_languages)
        else:
            file_text = extract_text_from_file(uploaded_file)

        # Process the file in chunks to simulate a more complex operation and show progress
        if file_text:
            chunks = file_text.split('\n')  # Example chunking, adjust based on your logic
            total_chunks = len(chunks)

            # Loop through chunks and update the individual file progress bar
            for i, chunk in enumerate(chunks):
                time.sleep(0.05)  # Simulate time delay for processing
                percent_complete = int((i + 1) / total_chunks * 100)
                file_progress.progress(percent_complete)

                elapsed_time = time.time() - start_time
                remaining_time = elapsed_time / (i + 1) * (total_chunks - (i + 1))

                status_message.text(f"Processing file {file_number}/{total_files}... {percent_complete}% complete, "
                                    f"Estimated time remaining: {int(remaining_time)} seconds")

            # After all chunks of the file are processed, index the file
            index_text(user_index, file_text, uploaded_file.name, total_files=total_files, file_number=file_number)
            file_progress.progress(100)
            status_message.success(f"File {file_number}/{total_files} processing complete!")

        else:
            status_message.error(f"Error processing file {file_number}: {uploaded_file.name}")

        # Update global progress
        global_progress_bar.progress(file_number / total_files)

    # Once all files are done
    global_status.success("All files have been processed and indexed successfully!")

if st.session_state["authentication_status"]:
    user_email = config['credentials']['usernames'][st.session_state["username"]]['email']
    user_index = f"user_{user_email.replace('@', '_').replace('.', '_')}"

    # Sidebar settings
    st.sidebar.title("Admin Settings")

    # Advanced settings toggle
    settings_visible = st.sidebar.toggle("Show/Hide Advanced Settings", value=False)
    advanced_settings = display_advanced_settings(settings_visible) if settings_visible else {}

    # Toggle to enable/disable the display of CPU, Memory, and Model Health metrics
    display_metrics = toggle_display_metrics()

    # Dropdown for index management actions
    index_action = st.sidebar.selectbox("Select Action", ["Create Index", "Delete Index", "List Documents"])

    if index_action == "Create Index":
        st.sidebar.subheader("Create a New Index")
        if user_email == ADMIN_EMAIL:
            new_index_name = st.sidebar.text_input("New Index Name")
        else:
            new_index_name = user_index

        if st.sidebar.button("Create Index"):
            if not new_index_name.strip():
                st.sidebar.error("Index name cannot be empty.")
            elif any(char in new_index_name for char in r'\/:*?"<>|'):
                st.sidebar.error("Index name contains invalid characters.")
            else:
                create_index(new_index_name)
                st.sidebar.success(f"Index '{new_index_name}' created successfully.")

    elif index_action == "Delete Index":
        st.sidebar.subheader("Delete an Index")
        if user_email == ADMIN_EMAIL:
            indexes = [index for index in es.indices.get_alias(index="*").keys() if not index.startswith('.')]
            selected_index_name = st.sidebar.selectbox("Select Index to Delete", options=indexes)
        else:
            selected_index_name = user_index
        if st.sidebar.button("Delete Index"):
            delete_index(selected_index_name)

    elif index_action == "List Documents":
        st.sidebar.subheader("List Documents in an Index")
        if user_email == ADMIN_EMAIL:
            indexes = [index for index in es.indices.get_alias(index="*").keys() if not index.startswith('.')]
            selected_index_name = st.sidebar.selectbox("Select Index to List Documents", options=indexes)
        else:
            selected_index_name = user_index

        if st.sidebar.button("List Documents"):
            document_df = list_documents(selected_index_name)
            if not document_df.empty:
                st.write(f"Documents indexed in '{selected_index_name}':")
                st.table(document_df)
            else:
                st.write(f"No documents found in index '{selected_index_name}'.")

    uploaded_files = st.file_uploader("Upload text, PDF, Image, or Word documents", accept_multiple_files=True)

    if uploaded_files:
        # Admin can choose an index, otherwise the user's own index is selected
        if user_email == ADMIN_EMAIL:
            indexes = [index for index in es.indices.get_alias(index="*").keys() if not index.startswith('.')]
            index_for_upload = st.selectbox("Select Index to Upload Document", options=indexes)
        else:
            index_for_upload = user_index

        # Choose extraction method
        extraction_method = st.selectbox("Choose extraction method for all PDFs", ["Read Normal", "Read Using OCR"])

        # OCR language selection
        if extraction_method == "Read Using OCR":
            selected_languages = st.multiselect(
                "Choose OCR Languages for all Documents",
                options=list(language_options.keys()),  # Language options
                format_func=lambda x: language_options.get(x, x)
            )
            if not selected_languages:
                st.error("Please select at least one language.")
        else:
            selected_languages = None

        # Start indexing when button is clicked
        if st.button("Index All Documents"):
            enhanced_indexing_progress(uploaded_files, extraction_method, selected_languages, index_for_upload)

    # Automatically refresh CPU, Memory, and Health status every 5 seconds if metrics display is enabled
    refresh_metrics(display_metrics)

else:
    st.markdown(':red[Please Login or Register]')
