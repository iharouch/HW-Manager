import streamlit as st
from openai import OpenAI
import chromadb
from pathlib import Path
import sys

# A fix for working with ChromaDB on Streamlit
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#Create an OpenAI client
if 'client' not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.client = OpenAI(api_key=api_key)

# System prompt
SYSTEM_PROMPT = """You are a helpful news chatbot. 
You answer questions about the given news articles. 
Use clear and concise language and provide context from the articles. 
Always end with 'Do you want more info?'"""

#Initialize messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Hi! I can help you find news about companies or interesting topics."}
        ]

# Load ChromaDB collection
BASE_DIR = Path(__file__).parent
db_path = BASE_DIR / "ChromaDB_News"
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection(name="HW7_NewsCollection")

# Function to get relevant articles from ChromaDB
def get_rag_context(query, collection, top_k=3):
    # Embed the query
    client = st.session_state.client
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )

    # Query collection
    query_embedding = response.data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

     # Build RAG context
    rag_context = "\n\n".join(
        f"Source: {doc_id}\n{doc}" 
        for doc, doc_id in zip(results["documents"][0], results["ids"][0])
    )
    return rag_context

#Functions to answer interesting news questions
def find_most_interesting_news(collection, top_k=5):
    query = "Find the most interesting news"
    return get_rag_context(query, collection, top_k=top_k)

def find_news_about(query, collection, top_k=5):
    # Search articles mentioning the topic/company
    return get_rag_context(query, collection, top_k=top_k)

# Chatbot interface
st.title("News Chatbot")

# Display chat history
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Get user input
if prompt := st.chat_input("What do you want to know? Want me to find the most interesting news or find news about a certain topic?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get relevant rag context
    if prompt.lower() == "find the most interesting news":
        rag_context = find_most_interesting_news(collection)
    elif prompt.lower().startswith("find news about"):
        topic = prompt[len("find news about "):].strip()
        rag_context = find_news_about(topic, collection)
    else:
        rag_context = get_rag_context(prompt, collection)

    # Update message for model
    chat_messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f"Use the following articles to answer the questions: \n{rag_context}"}
    ] + st.session_state.messages[1:]

    # Call OpenAI chat API
    response_stream = st.session_state.client.chat.completions.create(
        model="gpt-5-mini",
        messages=chat_messages,
        stream=True
    )

    # Display response as they're written
    with st.chat_message("assistant"):
        response = st.write_stream(response_stream)
    st.session_state.messages.append({"role": "assistant", "content": response})