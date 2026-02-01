import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from anthropic import Anthropic, AuthenticationError

# Show title and description.
st.title("HW 2 - MY URL question answering")
st.write("Provide a URL and ask a question about it – GPT will answer!")

# Store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets['OPENAI_API_KEY']

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)
client.models.list() # Validates the key by asking for the models that the key is compatible with

# Create a Claude client
claude_client = Anthropic(api_key=st.secrets['CLAUDE_API_KEY'])

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

# Let the user provide a URL.
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None

# Read content from URL
url = st.text_input("Enter a URL: ")

# Define model options for each LLM (OpenAI and Claude)
llm_models = {
    'OpenAI': ['gpt-5-mini', 'gpt-5-nano', 'gpt-4.1', 'gpt-5.2'],
    'Claude': ['claude-sonnet-4-5-20250929', 'claude-haiku-4-5-20251001', 'claude-opus-4-5-20251101']
}

# Radio button to select LLM
selected_llm = st.sidebar.radio("Choose an LLM:", list(llm_models.keys()))

# Provide user with dropdown to select model based on LLM
model_option = st.sidebar.selectbox("Choose a model:", llm_models[selected_llm])

#Add a checkbox to allow user to select the most advanced model available
if st.sidebar.checkbox("Use advanced model"):
    model_option = llm_models[selected_llm][-1]  # Select the last model in the list as the most advanced

#Allow the user to choose the language of the output
output_language = st.sidebar.selectbox("Choose output language:", [
    'English',
    'French',
    'Spanish',
    'German'
], disabled = not url) # Disabled until a URL is provided

# Provide user with three options for generating a summary
summary_option = st.sidebar.radio("Choose a summary type:", [
    'Summarize the document in 100 words',
    'Summarize the document in 2 connecting paragraphs',
    'Summarize the document in 5 bullet points'
], disabled=not url) # Disabled until a URL is available

if st.sidebar.button("Generate Summary", disabled=not url):
    if url:
        url_content = read_url_content(url)
        if url_content:
            messages = [
                {
                    "role": "user",
                    "content": f"Here's a document: {url_content} \n\n---\n\n {summary_option} in {output_language}",
                }
            ] # API call takes summary option and output language into account
            
            # Generate an answer using the OpenAI API.
            stream = client.chat.completions.create(
                model=model_option, # Use selected model
                messages=messages,
                stream=True,
            )
            
            # Stream the response to the app using `st.write_stream`.
            st.write_stream(stream)
        else:
            st.error(f"Could not read the URL. Please check that it's valid.")