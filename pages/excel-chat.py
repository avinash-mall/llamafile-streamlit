# pages/excel_chat.py

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv
from utils import setup_authentication_menu, clear_chat_history

load_dotenv()

st.set_page_config(page_icon="📊", page_title="Tabular Chat", layout="wide")
st.title("💡 Chat with Your Tabular Data")

# — Authentication —
name, config = setup_authentication_menu()
if not st.session_state.get("authentication_status"):
    st.markdown(":red[Please Login or Register]")
    st.stop()

# — Sidebar Settings —
st.sidebar.title("Settings")
if st.sidebar.button("Clear Chat History"):
    st.session_state.history = []
    clear_chat_history()
    st.toast("Chat history cleared.", icon="✅")

DEBUG = st.sidebar.toggle("Enable Debug Output", value=False)
if DEBUG:
    st.sidebar.write("Debug mode is on")

# — OpenAI client —
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL")
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

def local_llm_response(messages, stream_t=False):
    if stream_t:
        stream = client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, stream=True
        )
        return st.write_stream(stream)
    else:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, stream=False
        )
        return "".join(choice.message.content for choice in resp.choices)

def execute_code(code_str: str, mode: str, df: pd.DataFrame):
    env = {"df": df, "pd": pd, "plt": plt, "st": st}
    try:
        if DEBUG:
            st.markdown("```python\n" + code_str + "\n```")
        exec(code_str, env)
        if mode == "query":
            return True, env.get("result", None)
        fig = env.get("fig", None)
        if hasattr(fig, "get_figure"):
            fig = fig.get_figure()
        return True, fig
    except KeyError as e:
        cols = list(df.columns)
        return False, f"Column '{e.args[0]}' not found. Available: {cols}"
    except Exception as e:
        return False, str(e)

def run_query(query: str, df: pd.DataFrame):
    # Build schema & sample preview
    schema  = "\n".join(f"{c}: {t}" for c, t in df.dtypes.items())
    preview = df.head(5).to_csv(index=False)

    is_plot = any(k in query.lower() for k in ("plot", "chart", "graph"))

    system_plot = (
        "You are an expert Python data analyst. You have Pandas DataFrame `df` and `matplotlib.pyplot` as `plt` available."
        " Write a Python code snippet using `df` to generate a plot that answers the user's request."
        " Follow these steps:"
        " 1. Perform any necessary data aggregation or manipulation using Pandas on `df`."
        "    - When aggregating (e.g. using `.groupby()`, `.value_counts()`), ensure the result is suitable for plotting (e.g., use `.reset_index()` if needed)."
        # Specific guidance based on past errors:
        "    - For counting occurrences of categories in a column 'Col', prefer using `df['Col'].value_counts()`."
        "    - For getting the number of rows in a DataFrame or a filtered DataFrame `filtered_df`, use `len(filtered_df)` or `filtered_df.shape[0]`."
        "    - Do NOT use the `.size` attribute for simple row counts."
        " 2. Create a Matplotlib Figure and Axes using: `fig, ax = plt.subplots()`."
        " 3. Generate the plot using the `ax` object (e.g., using `result_df.plot(kind='...', ax=ax)`, `ax.scatter(...)`, `ax.hist(...)`, etc.)."
        " 4. Choose an appropriate plot type (e.g., 'bar', 'line', 'scatter', 'hist', 'pie') if the user doesn't specify one, based on the data."
        " 5. Set a meaningful title for the plot using `ax.set_title('...')`."
        # Crucial output instruction:
        " 6. The final Matplotlib Figure object MUST be assigned to a variable named `fig` (e.g., `fig = ax.figure`). Do NOT assign the Axes (`ax`) to `fig`."
        # Constraints:
        " Do NOT include `import` statements or `plt.show()`."
        " Return ONLY the raw Python code snippet for execution, without any markdown formatting (like ```python) or explanations."
    )

    system_query = (
        "You are an expert Python data analyst. You have a Pandas DataFrame named `df` available."
        " Write a Python code snippet using `df` to answer the user's question."
        " Use standard Pandas operations."
        # Specific guidance based on past errors:
        " For counting occurrences of categories in a column 'Col', prefer using `df['Col'].value_counts()`."
        " For getting the number of rows in a DataFrame or a filtered DataFrame `filtered_df`, use `len(filtered_df)` or `filtered_df.shape[0]`."
        " Do NOT use the `.size` attribute for simple row counts, as it counts all elements (rows * columns)."
        # Crucial output instruction:
        " The final result (e.g., a DataFrame, Series, string, number, or list) MUST be assigned to a variable named `result`."
        # Constraints:
        " Do NOT include import statements (like `import pandas as pd`) or code to load data."
        " Return ONLY the raw Python code snippet for execution, without any markdown formatting (like ```python) or explanations."
    )

    if is_plot:
        system = system_plot  # Use the refined plot prompt
        mode = "plot"
    else:
        system = system_query  # Use the refined query prompt
        mode = "query"

    user_msg = (
        f"Schema:\n{schema}\n\n"
        f"Sample (5 rows):\n{preview}\n\n"
        f"User Request: {query}\n"
        "Return only the code snippet."
    )

    # Build the full messages list, including history
    messages = [{"role": "system", "content": system}]
    for turn in st.session_state.history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_msg})

    raw = local_llm_response(messages, stream_t=False)

    # strip markdown fences
    lines = raw.strip().splitlines()
    if lines and lines[0].startswith("```"): lines = lines[1:]
    if lines and lines[-1].startswith("```"): lines = lines[:-1]
    code = "\n".join(lines).strip()

    return mode, *execute_code(code, mode, df)

# — Initialize in‑page history —
if "history" not in st.session_state:
    st.session_state.history = []

# — File upload —
df = None
uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx","csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
        df.columns = df.columns.str.strip()
        st.subheader("Data Preview")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")
        if DEBUG: st.write(e)
        st.stop()

# — Replay conversation history —
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        # For user: content is the question. For assistant: content is text or "<plot>"
        st.markdown(turn["content"])

# — Chat input (same style as other pages) —
if df is not None:
    if prompt := st.chat_input("Ask a question or request a plot:"):
        # record user turn
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # run the query + code exec
        mode, ok, out = run_query(prompt, df)

        # prepare assistant content for display & for next prompt
        if not ok:
            content = f"Error: {out}"
        elif mode == "query":
            if isinstance(out, pd.DataFrame):
                content = out.head(5).to_csv(index=False)
            else:
                content = str(out)
        else:  # plot
            content = "<plot>"

        st.session_state.history.append({
            "role": "assistant",
            "content": content
        })

        # display assistant turn
        with st.chat_message("assistant"):
            if not ok:
                st.error(out)
            elif mode == "query":
                if isinstance(out, pd.DataFrame):
                    st.dataframe(out)
                else:
                    st.write(out)
            else:
                st.pyplot(out)
