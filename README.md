# Chat Assistant Application

This repository contains a Streamlit application for interacting with an OpenAI model. The application supports various file uploads, real-time system monitoring, and customizable prompts for dynamic interaction with the model.

## Features

- **OpenAI API Integration**: Easily interact with the OpenAI model using customizable prompts.
- **File Upload and Text Extraction**: Supports multiple file types (PDF, DOCX, images, etc.) and extracts text for processing.
- **Real-time System Monitoring**: Monitors and displays CPU usage, memory usage, and model health status.
- **Advanced Settings**: Fine-tune model parameters such as temperature, top_p, frequency_penalty, and presence_penalty.

## Installation

### Prerequisites

- Python 3.8 or higher
- [pip](https://pip.pypa.io/en/stable/installation/)


### Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt 
```
### Environment Variables

Create a `.env` file in the root directory of the project with the following content:

```
OPENAI_BASE_URL=<your-openai-base-url>
OPENAI_API_KEY=<your-openai-api-key>
OPENAI_MODEL=<your-openai-model>` 
```

Replace the placeholders with your actual OpenAI API information.

## Usage

### Run the Application

To run the Streamlit application, use the following command:

```bash
streamlit run app.py
```
The application will start and you can access it in your web browser at `http://localhost:8501`.

### Run the .llamafile

Download and run the llamafile from the Hugging Face model `Mozilla/Meta-Llama-3.1-8B-Instruct-llamafile`:

```bash
./Meta-Llama-3.1-8B-Instruct.Q8_0.llamafile -c 0 --server --host 0.0.0.0 --nobrowser --mlock
```
This command will start the Llama server in a non-browser mode.
Use the context according to the memory you have. c 8192 needs around 8GB, c 0 needs around 16GB

## Customization

You can customize the system prompts and other settings through the sidebar in the Streamlit app. Adjust parameters like temperature, top_p, and others to tailor the model's behavior to your needs.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue on GitHub.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
