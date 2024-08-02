import os
import httpx
import json
import streamlit as st
import asyncio
import time
from datetime import datetime
import pytz

# Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8080")
TIMEOUT = 600  # Timeout in seconds (10 minutes)
INSTRUCTION = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

def trim(text):
    """Trim leading and trailing whitespace from the text."""
    return text.strip()

def trim_trailing(text):
    """Trim trailing whitespace from the text."""
    return text.rstrip()

def format_prompt(messages, include_datetime):
    """Format the chat messages into a prompt for the AI."""
    system_info = ""
    if include_datetime:
        system_info = get_current_datetime_info() + "\n"
    chat_text = "\n".join([f"### {'Human' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in messages])
    return f"{INSTRUCTION}\n{system_info}{chat_text}\n### Human:"

def get_current_datetime_info():
    """Get current date, time, day, and time zone."""
    now = datetime.now(pytz.timezone('Asia/Dubai'))
    current_time = now.strftime('%H:%M')
    day = now.strftime('%A')
    date = now.strftime('%B %d, %Y')
    time_zone = now.tzinfo.zone
    return f"The current time is {current_time} on {day}, {date}, Time zone is {time_zone}."

async def chat_completion(question, messages, settings, debug_mode, display_timing, include_datetime):
    """Send a chat completion request to the AI API and stream the response."""
    messages_copy = messages + [{"role": "user", "content": question}]
    prompt = trim_trailing(format_prompt(messages_copy, include_datetime) + "\n### Assistant:")
    data = {
        "prompt": prompt,
        "temperature": settings["temperature"],
        "top_k": settings["top_k"],
        "top_p": settings["top_p"],
        "min_p": settings["min_p"],
        "n_predict": settings["n_predict"],
        "n_keep": settings["n_keep"],
        "typical_p": settings["typical_p"],
        "repeat_penalty": settings["repeat_penalty"],
        "repeat_last_n": settings["repeat_last_n"],
        "penalize_nl": settings["penalize_nl"],
        "presence_penalty": settings["presence_penalty"],
        "frequency_penalty": settings["frequency_penalty"],
        "penalty_prompt": settings["penalty_prompt"],
        "mirostat": settings["mirostat"],
        "mirostat_tau": settings["mirostat_tau"],
        "mirostat_eta": settings["mirostat_eta"],
        "seed": settings["seed"],
        "ignore_eos": settings["ignore_eos"],
        "n_probs": settings["n_probs"],
        "slot_id": settings["slot_id"],
        "cache_prompt": settings["cache_prompt"],
        "stream": True,
        "stop": ["\n### Human:"]
    }

    if debug_mode:
        st.write("Prompt sent to API:")
        st.json(data)

    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            async with client.stream("POST", f"{API_URL}/completion", headers={"Content-Type": "application/json"}, json=data) as response:
                answer = ""
                message_container = st.empty()
                async for line in response.aiter_lines():
                    if line and line.startswith("data:"):
                        content = json.loads(line[5:])["content"]
                        answer += content
                        with message_container.container():
                            st.markdown(f"**Assistant**: {answer}")
                end_time = time.time()
                if display_timing:
                    st.markdown(f":blue[Response Time: {end_time - start_time:.2f} seconds]")
                return trim(answer)
    except httpx.ReadTimeout:
        st.error("Request timed out. Please try again.")
        return ""
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

def reset_chat():
    """Reset the chat history and session state."""
    st.session_state.messages = []
    st.session_state.partial_answer = ""

# Streamlit app setup
if "messages" not in st.session_state:
    st.session_state.messages = []

if "partial_answer" not in st.session_state:
    st.session_state.partial_answer = ""

# Sidebar settings
with st.sidebar:
    if st.button("Start New Chat"):
        reset_chat()

    st.header("Settings")

    st.markdown("### General Settings")
    debug_mode = st.toggle("Debug Mode", value=False, help="Enable to display the prompt sent to the API for debugging purposes.")
    display_timing = st.toggle("Display Response Timing", value=True, help="Enable to display the response timing for the AI.")
    include_datetime = st.toggle("Include Current DateTime", value=True, help="Enable to include the current date, time, day, and time zone in the system prompt.")

    st.divider()

    st.markdown("### Generation Settings")
    temperature = st.slider("Temperature:", 0.0, 1.0, 0.8, 0.1, help="Adjust the randomness of the generated text (default: 0.8).")
    n_predict = st.selectbox("Number of predictions:", [-1, 512, 1024, 1536, 2048, 4096, 8192], 0, help="Set the maximum number of tokens to predict when generating text (default: -1 for infinity).")
    top_k = st.slider("Top_k:", 0, 100, 40, help="Limit the next token selection to the K most probable tokens (default: 40).")
    top_p = st.slider("Top_p:", 0.0, 1.0, 0.95, help="Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P (default: 0.95).")
    min_p = st.slider("Min_p:", 0.0, 1.0, 0.05, help="The minimum probability for a token to be considered, relative to the probability of the most likely token (default: 0.05).")
    n_keep = st.slider("N_keep:", -1, 512, 0, help="Specify the number of tokens from the prompt to retain when the context size is exceeded and tokens need to be discarded (default: 0). Use -1 to retain all tokens from the prompt.")
    typical_p = st.slider("Typical_p:", 0.0, 1.0, 1.0, help="Enable locally typical sampling with parameter p (default: 1.0).")
    repeat_penalty = st.slider("Repeat_penalty:", 0.0, 2.0, 1.1, help="Control the repetition of token sequences in the generated text (default: 1.1).")
    repeat_last_n = st.slider("Repeat_last_n:", -1, 512, 64, help="Last n tokens to consider for penalizing repetition (default: 64, 0 = disabled, -1 = ctx-size).")

    st.divider()

    st.markdown("### Penalty Settings")
    penalize_nl = st.toggle("Penalize newlines:", value=True, help="Penalize newline tokens when applying the repeat penalty (default: true).")
    presence_penalty = st.slider("Presence_penalty:", 0.0, 2.0, 0.0, help="Repeat alpha presence penalty (default: 0.0, 0.0 = disabled).")
    frequency_penalty = st.slider("Frequency_penalty:", 0.0, 2.0, 0.0, help="Repeat alpha frequency penalty (default: 0.0, 0.0 = disabled).")
    penalty_prompt = st.text_input("Penalty prompt:", "", help="This will replace the prompt for the purpose of the penalty evaluation. Can be either null, a string or an array of numbers representing tokens (default: null = use the original prompt).")

    st.divider()

    st.markdown("### Advanced Settings")
    mirostat = st.slider("Mirostat:", 0, 2, 0, help="Enable Mirostat sampling, controlling perplexity during text generation (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0).")
    mirostat_tau = st.slider("Mirostat_tau:", 0.0, 10.0, 5.0, help="Set the Mirostat target entropy, parameter tau (default: 5.0).")
    mirostat_eta = st.slider("Mirostat_eta:", 0.0, 1.0, 0.1, help="Set the Mirostat learning rate, parameter eta (default: 0.1).")
    seed = st.number_input("RNG seed:", value=-1, help="Set the random number generator (RNG) seed (default: -1, -1 = random seed).")
    ignore_eos = st.toggle("Ignore end of stream token:", value=False, help="Ignore end of stream token and continue generating (default: false).")
    n_probs = st.slider("N_probs:", 0, 100, 0, help="If greater than 0, the response also contains the probabilities of top N tokens for each generated token (default: 0).")
    slot_id = st.number_input("Slot_id:", value=-1, help="Assign the completion task to a specific slot. If -1, the task will be assigned to an idle slot (default: -1).")
    cache_prompt = st.toggle("Cache prompt:", value=False, help="Save the prompt and generation to avoid reprocessing the entire prompt if a part of it isn't changed (default: false).")

settings = {
    "temperature": temperature,
    "top_k": top_k,
    "top_p": top_p,
    "min_p": min_p,
    "n_predict": n_predict,
    "n_keep": n_keep,
    "typical_p": typical_p,
    "repeat_penalty": repeat_penalty,
    "repeat_last_n": repeat_last_n,
    "penalize_nl": penalize_nl,
    "presence_penalty": presence_penalty,
    "frequency_penalty": frequency_penalty,
    "penalty_prompt": penalty_prompt if penalty_prompt else None,
    "mirostat": mirostat,
    "mirostat_tau": mirostat_tau,
    "mirostat_eta": mirostat_eta,
    "seed": seed,
    "ignore_eos": ignore_eos,
    "n_probs": n_probs,
    "slot_id": slot_id,
    "cache_prompt": cache_prompt,
}

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st.session_state.partial_answer = ""
        with st.spinner("Writing..."):
            response = asyncio.run(chat_completion(prompt, st.session_state.messages, settings, debug_mode, display_timing, include_datetime))

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})
