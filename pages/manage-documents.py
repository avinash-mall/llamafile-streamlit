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
    setup_authentication_menu
)

# Suppress Elasticsearch system indices warnings
warnings.filterwarnings("ignore", category=ElasticsearchWarning)

# Load environment variables from .env file
load_dotenv()

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
name = setup_authentication_menu()

if st.session_state["authentication_status"]:
    # Sidebar settings
    st.sidebar.title("Admin Settings")

    # Advanced settings toggle
    settings_visible = st.sidebar.toggle("Show/Hide Advanced Settings", value=False)
    advanced_settings = display_advanced_settings(settings_visible) if settings_visible else {}

    # Toggle to enable/disable the display of CPU, Memory, and Model Health metrics
    display_metrics = toggle_display_metrics()

    # Dropdown for index management actions
    index_action = st.sidebar.selectbox("Select Action", ["Create Index", "Delete Index", "List Documents"])

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

    def create_index(index_name):
        field_name = "embedding"
        similarity_type = os.getenv("SIMILARITY_TYPE", "cosine")
        default_dims = int(os.getenv("DEFAULT_DIMS", 1024))

        if es.indices.exists(index=index_name):
            st.error("Index already exists")
            return

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
        st.success(f"Index '{index_name}' with {default_dims} dimensions and similarity '{similarity_type}' created successfully")

    def delete_index(index_name):
        if not es.indices.exists(index=index_name):
            st.error("Index not found")
            return
        es.indices.delete(index=index_name)
        st.success(f"Index '{index_name}' deleted successfully")

    if index_action == "Create Index":
        st.sidebar.subheader("Create a New Index")
        new_index_name = st.sidebar.text_input("New Index Name")

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
        indexes = [index for index in es.indices.get_alias(index="*").keys() if not index.startswith('.')]
        selected_index_name = st.sidebar.selectbox("Select Index to Delete", options=indexes)
        if st.sidebar.button("Delete Index"):
            delete_index(selected_index_name)

    elif index_action == "List Documents":
        st.sidebar.subheader("List Documents in an Index")
        indexes = [index for index in es.indices.get_alias(index="*").keys() if not index.startswith('.')]
        selected_index_name = st.sidebar.selectbox("Select Index to List Documents", options=indexes)
        if st.sidebar.button("List Documents"):
            document_df = list_documents(selected_index_name)
            if not document_df.empty:
                st.write(f"Documents indexed in '{selected_index_name}':")
                st.table(document_df)
            else:
                st.write(f"No documents found in index '{selected_index_name}'.")

    uploaded_files = st.file_uploader("Upload text, PDF, or Word documents", type=["txt", "pdf", "docx"], accept_multiple_files=True)

    if uploaded_files:
        indexes = [index for index in es.indices.get_alias(index="*").keys() if not index.startswith('.')]
        index_for_upload = st.selectbox("Select Index to Upload Document", options=indexes)

        extraction_method = st.selectbox("Choose extraction method for all PDFs", ["Read Normal", "Read Using PDF OCR"])

        if extraction_method == "Read Using PDF OCR":
            selected_languages = st.multiselect(
                "Choose OCR Languages for all PDFs",
                options=list(language_options.keys()),  # Correct language codes
                format_func=lambda x: language_options.get(x, x)
            )

            if selected_languages:
                st.write(f"Selected languages: {', '.join(selected_languages)}")
            else:
                st.error("Please select at least one language.")
        else:
            selected_languages = None

        if st.button("Index All Documents"):
            total_files = len(uploaded_files)
            for file_number, uploaded_file in enumerate(uploaded_files, start=1):
                file_type = uploaded_file.type
                progress_text = f"Processing document {file_number}/{total_files}. Please wait..."
                my_bar = st.progress(0, text=progress_text)

                if file_type == "application/pdf":
                    file_text = extract_text_from_pdf(uploaded_file, extraction_method == "Read Using PDF OCR",
                                                      selected_languages)
                else:
                    file_text = extract_text_from_file(uploaded_file)

                if file_text:
                    my_bar.progress(50, text=f"Document {file_number}/{total_files} - Text extracted successfully.")
                    index_text(index_for_upload, file_text, uploaded_file.name, total_files=total_files,
                               file_number=file_number)
                    my_bar.progress(100, text=f"Document {file_number}/{total_files} - Indexed successfully.")
                    st.success(f"Document '{uploaded_file.name}' indexed successfully in '{index_for_upload}'.")

            st.success(f"All documents have been indexed successfully in '{index_for_upload}'.")

    # Automatically refresh CPU, Memory, and Health status every 5 seconds if metrics display is enabled
    refresh_metrics(display_metrics)

else:
    st.markdown(':red[Please Login or Register]')
