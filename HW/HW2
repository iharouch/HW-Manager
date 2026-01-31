import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

# Show title and description.
st.title("HW 2 - MY Document question answering")
st.write(
    "Provide a URL and ask a question about it – GPT will answer!"
)

# Store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets.OPENAI_API_KEY

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)
client.models.list() # Validates the key by asking for the models that the key is compatible with

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
url_content = read_url_content(url)

# Provide user with options for model selection
use_advanced_model = st.sidebar.checkbox("Use advanced model (gpt-5.2)", value=False) # Checkbox that uses most advanced model if checked

if use_advanced_model:
    model_option = 'gpt-5.2'
else:
    model_option = st.sidebar.selectbox("Choose a model:", [
        'gpt-5-mini',
        'gpt-5-nano',
        'gpt-4.1',
        'gpt-5.2'
    ], index=0) # Selectbox for model choice if not automatically using advanced model as default

# Provide user with three options for generating a summary
summary_option = st.sidebar.radio("Choose a summary type:", [
    'Summarize the document in 100 words',
    'Summarize the document in 2 connecting paragraphs',
    'Summarize the document in 5 bullet points'
], disabled=not url_content) # Disabled until a document is available

if st.sidebar.button("Generate Summary", disabled=not url_content):
    if url_content:
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {url_content} \n\n---\n\n {summary_option}",
            }
        ] # API call takes summary option into account

        # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model=model_option, # Use selected model
            messages=messages,
            stream=True,
        )

        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)