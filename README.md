# Chat Assistant Application

This repository contains a Streamlit application for interacting with an OpenAI compatible model. The application supports various file uploads, real-time system monitoring, and customizable prompts for dynamic interaction with the model.

It offers three main features:
1. **AI Chatbot**: An interactive chatbot interface.
2. **Document Summarizer**: A tool for summarizing text from various file formats.
3. **Document Chat**: A tool to chat with your documents per user.
4. **User Authentication**: A tool for users to register, login, update.

# Key Files:

-    app.py: Main application entry point. Handles system prompt selection, page authentication, and overall UI.
-    document-summarize.py: Allows users to upload files and summarizes document content.
-    document-chat.py: Enables chat functionality with document content using OpenAI and Elasticsearch.
-    manage-documents.py: Admin interface for indexing, managing, and deleting documents.
-    utils.py: Contains utility functions for authentication, email notifications, and Elasticsearch management.
-    downloads.py: Downloads models like SentenceTransformer and EasyOCR for local use.
-    config.yaml: Stores configuration settings for authentication, email, and more. **Note**: admin user password is admin.
-    settings.json: Configuration file for OpenAI API settings.

# Configure settings:
    Update .env, config.yaml and settings.json with your own values for OpenAI, Elasticsearch, and other settings.
   
## Features

- **OpenAI API Integration**: Easily interact with the OpenAI compatible model api using customizable prompts.
- **Chatbot**: Interact with the AI model using a conversational interface.
- **Document Summarizer**: Upload files in different formats (PDF, DOCX, images, etc.) and get concise summaries.
- **OCR Support**: Extract text from images and scanned PDFs using EasyOCR.
- **Advanced Settings**: Customize parameters such as temperature, top-p, frequency penalty, presence penalty, and more.
- **Real-time Metrics**: Display CPU, memory usage, and model health status.
- **Document Management**: Manage your documents, indices.
- **User Management**: Manage Users.

## Installation

### Run in Docker

1. **Clone the Repository**:
```bash
git clone https://github.com/avinash-mall/llamafile-streamlit.git
cd llamafile-streamlit
```
2. **Change the settings**
   a. Edit the .env and change at minimum the following:
       ```env
           OPENAI_BASE_URL=http://<your ip address>:8080 # Replace with your OPENAI API host URL
           ES_HOST_URL=http://<your ip address>:9200  # Replace with your Elasticsearch host URL
           ES_USERNAME=elastic  # Replace with your Elasticsearch username
           ES_PASSWORD=changeme  # Replace with your Elasticsearch password
       ```
   b. Edit the config.yaml and at minimum the following:
       ```env
       smtp:
          port: 25
          server: smtp.freesmtpservers.com
          use_tls: false
          username: test@myapp.com
       app:
          domain: http://<your ip address or host for app>:8501/
       ```
4. **Run the app**
```bash
docker compose up -d
```

### Run from source

### Prerequisites

- Python 3.8 or higher
- [pip](https://pip.pypa.io/en/stable/installation/)

1. **Clone the Repository**:
```bash
git clone https://github.com/avinash-mall/llamafile-streamlit.git
cd llamafile-streamlit
```
    
2. **Install the Required Packages**:
Install the required Python packages using `pip`:
```bash
pip install -r requirements.txt 
```

3. **Set Up Environment Variables**:
Edit `.env` file in the root directory of the project with the following content.
 ```env
 OPENAI_BASE_URL="https://api.openai.com"
 OPENAI_API_KEY="your_openai_api_key"
 OPENAI_MODEL="llama3.1" # or the model of your choice
 MODEL_PATH="path_to_your_model"
 TOKENIZER_PATH="path_to_tokenizer"
 MAX_TOKENS="131072"
 SYSTEM_INSTRUCTION="Your system instruction here for only text summarizer."
 EASYOCR_MODELS_PATH="./models/easyocr"
 ```
Edit the config.yaml and at minimum the following:
```env
smtp:
    port: 25
    server: smtp.freesmtpservers.com
    use_tls: false
    username: test@myapp.com
app:
    domain: http://<your ip address or host for app>:8501/
```
Replace the placeholders with your actual OpenAI API information.

4. **Download The Models**:
```python
python downloads.py
```

5. **Run the Application**:

Start the Streamlit app with:
```bash
streamlit run app.py
```
The application will start and you can access it in your web browser at `http://localhost:8501`.

6. **Run the LLamafile**:
Download and run the llamafile from the Hugging Face model `Mozilla/Meta-Llama-3.1-8B-Instruct-llamafile`:
```bash
./Meta-Llama-3.1-8B-Instruct.Q8_0.llamafile -c 0 --server --host 0.0.0.0 --nobrowser --mlock
```
This command will start the Llama server in a non-browser mode.
Use the context according to the memory you have. c 8192 needs around 8GB, c 0 needs around 16GB

## Customization
You can customize the system prompts and other settings through the settings.json and system_prompts.json. Adjust parameters like temperature, top_p, and others to tailor the model's behavior to your needs.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue on GitHub.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
