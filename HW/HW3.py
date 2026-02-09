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

def keep_last_n_user_messages(messages, n=2):
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
st.title("MY Lab3 question answering chatbot")

# Client dictionary to store API clients for reuse in session state
if 'clients' not in st.session_state:
    st.session_state.clients = {}

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
        claude_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
        return True
    except AuthenticationError:
        return False
    except Exception:
        return True

if "claude_client" not in st.session_state:
    st.session_state.claude_client = validate_claude_key()

if not st.session_state.claude_client:
    st.error("Invalid Claude API key")

# Initialize messages with system prompt (protected from removal)
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
    messages_to_send = keep_last_n_user_messages(st.session_state.messages, n=2)

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
        client = st.session_state.clients["claude"]
        # Anthropic uses a different message format; convert from OpenAI format
        # Remove system message and pass it separately
        user_messages = [msg for msg in messages_to_send if msg["role"] != "system"]
        system_msg = next((msg["content"] for msg in messages_to_send if msg["role"] == "system"), "")
        
        with st.chat_message("assistant"):
            response = ""
            with client.messages.stream(
                model=model,
                max_tokens=1024,
                system=system_msg,
                messages=user_messages,
            ) as stream:
                for text in stream.text_stream:
                    response += text
                    st.write(text, end="")
        st.session_state.messages.append({"role": "assistant", "content": response})