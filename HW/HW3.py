import streamlit as st
from openai import OpenAI
from anthropic import Anthropic, AuthenticationError
import requests
from bs4 import BeautifulSoup

# System prompt to guide bot behavior
SYSTEM_PROMPT = """You are a helpful Q&A chatbot. Follow these rules STRICTLY:
1. When answering a NEW QUESTION, provide a clear, concise answer that a 10-year-old can understand
2. Use simple words and avoid technical terms. Explain complex ideas with everyday examples.
3. ALWAYS end your answer with: "Do you want more info?"
4. If the user says "Yes" or "yes", provide additional detailed information and ALWAYS end with: "Do you want more info?"
5. If the user says "No" or "no", respond with: "How can I help you with something else?"
Keep responses focused, helpful, and easy to understand."""

def keep_last_n_user_messages(messages, n=3):
    """Keep only the last n user messages and their responses, while preserving system prompt"""
    # Find user message indices (skip system prompt at index 0)
    user_message_indices = [i for i, msg in enumerate(messages) if msg["role"] == "user"]
    
    if len(user_message_indices) <= n:
        # Keep system prompt + all user messages and responses
        return messages
    
    # Find the index of the (n)th most recent user message
    start_index = user_message_indices[-n]
    
    # Return system prompt (index 0) plus messages from that point onward
    return [messages[0]] + messages[start_index:]

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None

#Show title and description
st.title("MY HW3 question answering chatbot")
st.write("Ask a question to the chatbot and optionally add URLs if you would like it to use them for context or reference. The chatbot will answer your question based on its own knowledge and the provided URLs. The chatbot has a 6-message memory buffer, so it will only remember your last 3 questions and its last 3 answers. Therefore, if you asked more than 3 questions and want to refer to something you asked before, please add it again in your current question to remind the chatbot.")

# Client dictionary to store API clients for reuse in session state
if 'clients' not in st.session_state:
    st.session_state.clients = {}

# LLM options with premium model
llm = st.sidebar.selectbox("Select LLM", ("OpenAI", "Claude"))
if llm == "OpenAI":
    model = "gpt-5.2"
else:
   model = "claude-opus-4-5-20251101"

if llm == "OpenAI":
    if "openai" not in st.session_state.clients:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.session_state.clients["openai"] = OpenAI(api_key=api_key)

    client = st.session_state.clients["openai"]
    client.models.list() # Validates the key by asking for the models that the key is compatible with

elif llm == "Claude":
    if "claude" not in st.session_state.clients:
        api_key = st.secrets["CLAUDE_API_KEY"]
        st.session_state.clients["claude"] = Anthropic(api_key=api_key)

    client = st.session_state.clients["claude"]

#Validates claude api key
def validate_claude_key():
    try:
        client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
        return True
    except AuthenticationError:
        return False
    except Exception:
        return True

if "client" not in st.session_state:
    st.session_state.client = validate_claude_key()

if not st.session_state.client:
    st.error("Invalid Claude API key")

# Initialize messages with system prompt (so that it doesn't get removed)
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "How can I help you?"}
    ]

# Display chat history (skip system prompt)
for msg in st.session_state.messages[1:]:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

# URL inputs
st.subheader("Attach URLs (optional)")
url1 = st.text_input("URL 1", key="url1")
add_second = st.sidebar.radio("Do you want to add a second URL?", ("No", "Yes"), index=0)
if add_second == "Yes":
    url2 = st.text_input("URL 2", key="url2")

# Get user input
if prompt := st.chat_input("What do you need help with?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Apply buffer while also using system prompt
    messages_to_send = keep_last_n_user_messages(st.session_state.messages, n=3)

    # Read URL content inline (summary-style approach)
    url_text = ""

    if url1:
        content = read_url_content(url1)
        if content:
            url_text += "\n\n" + content # Add URL content to the message history for context

    if add_second == "Yes" and url2:
        content = read_url_content(url2)
        if content:
            url_text += "\n\n" + content # Add second URL content to the message history for context

    if url_text:
        messages_to_send = [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + "Use the following web content to answer the user's question. "
                    "If the answer is not in the content, say you don't know."
                    + url_text
                ),
            }
        ] + [m for m in messages_to_send if m["role"] != "system"]

    if llm == "OpenAI":
        client = st.session_state.clients["openai"]
        stream = client.chat.completions.create(
            model=model,
            messages=messages_to_send,
            stream=True
        )
        
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    else:  # Claude
        system_text = "".join(m["content"] for m in messages_to_send if m["role"] == "system")
        user_messages = [m for m in messages_to_send if m["role"] == "user"]

        client = st.session_state.clients["claude"]
        stream = client.messages.create(
            model=model, # Use selected model
            max_tokens=1024,
            messages=user_messages,
            system=system_text,
                )
        #Extract and display text response from Claude
        st.write(stream.content[0].text)