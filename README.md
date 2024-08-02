# Chat Assistant Application

This application is a chat interface that interacts with an AI assistant, providing helpful, detailed, and polite responses to user queries. It uses Streamlit for the frontend, `httpx` for asynchronous HTTP requests, and various settings to customize the AI's responses.

## Features

- **Chat Interface**: Interact with the AI assistant through a user-friendly chat interface.
- **Customizable Settings**: Adjust various settings for AI response generation, including temperature, top_k, top_p, and more.
- **Response Timing**: Optionally display the response time for each AI-generated reply.
- **Debug Mode**: Enable debug mode to view the prompt sent to the AI API for troubleshooting.

## Configuration

The application is configured using environment variables and Streamlit widgets. The key configuration variables include:

- `API_URL`: The URL of the AI API (default: `http://127.0.0.1:8080`).
- `TIMEOUT`: Request timeout in seconds (default: 600 seconds or 10 minutes).
- `INSTRUCTION`: Initial instruction to the AI assistant.

## Setup

1. **Install Dependencies**: Ensure you have the necessary dependencies installed. You can install them using pip:
    ```sh
    pip install streamlit httpx pytz
    ```
2. **Run the .llamafile**: Download and run the llamafile from huggingface model Mozilla/Meta-Llama-3.1-8B-Instruct-llamafile.
   ```sh
   ./Meta-Llama-3.1-8B-Instruct.Q8_0.llamafile -c 8192 --server --nobrowser
   ```
2. **Environment Variables**: Set the `API_URL` environment variable if you are not using the default API URL.

3. **Run the Application**: Launch the Streamlit application:
    ```sh
    streamlit run app.py
    ```

## Code Overview

### Helper Functions

- **trim(text)**: Trims leading and trailing whitespace from the text.
- **trim_trailing(text)**: Trims trailing whitespace from the text.
- **format_prompt(messages, include_datetime)**: Formats the chat messages into a prompt for the AI, optionally including the current date and time.
- **get_current_datetime_info()**: Returns the current date, time, day, and time zone information.
- **chat_completion(question, messages, settings, debug_mode, display_timing, include_datetime)**: Sends a chat completion request to the AI API and streams the response.
- **reset_chat()**: Resets the chat history and session state.

### Streamlit Interface

- **Session State**: Manages chat messages and partial responses.
- **Sidebar Settings**: Provides various settings to customize AI response generation, including general settings, generation settings, penalty settings, and advanced settings.
- **Chat Display**: Displays chat history and user input.

## Example Usage

1. **Start a New Chat**: Click the "Start New Chat" button in the sidebar to reset the chat history.
2. **Adjust Settings**: Use the sidebar to adjust settings such as temperature, top_k, top_p, and more.
3. **Ask a Question**: Enter your question in the chat input box and receive a response from the AI assistant.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
