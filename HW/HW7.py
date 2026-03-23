import streamlit as st
from openai import OpenAI
import chromadb
from pathlib import Path
import sys
import pandas as pd

# A fix for working with ChromaDB on Streamlit
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Initialize OpenAI client in session state
if 'client' not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Paths
BASE_DIR = Path(__file__).parent
db_path = BASE_DIR / "ChromaDB_News"
db_path.mkdir(exist_ok=True)
csv_path = BASE_DIR / "HW-7-Data" / "news.csv"

# Load ChromaDB collection
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection(name="HW7_NewsCollection")

# Load CSV
csv_path = BASE_DIR / "HW-7-Data" / "news.csv"

# Build RAG DB only if empty
if "db_built" not in st.session_state:
    st.session_state.db_built = False

if not st.session_state.db_built:
    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
        text = str(row.get('Document', '')).replace('\n', ' ')[:5000]
        metadata = {
            "company_name": row.get('company_name', ''),
            "date": row.get('Date', ''),
            "url": row.get('URL', '')
        }
        response = st.session_state.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        collection.add(
            documents=[text],
            ids=[str(idx)],
            embeddings=[embedding],
            metadatas=[metadata]
        )
        print(f"Processed article {idx+1}/{len(df)}")

    st.session_state.db_built = True

# System prompt
SYSTEM_PROMPT = """You are a helpful news chatbot. 
You answer questions about the given news articles. 
Use clear and concise language and provide context from the articles. 
If the user asks "Find the most interesting news", return a ranked-order list of articles and context.
If the user asks "Find news about X", return news on specific topic/company.
Always end with 'Do you want more info?'"""

#Initialize messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Hi! I can help you find news about companies or interesting topics."}
        ]

# Function to get RAG context
def get_rag_context(query, top_k=3):
    response = st.session_state.client.embeddings.create(input=query, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    return "\n\n".join(f"Source: {doc_id}\n{doc}" for doc, doc_id in zip(results["documents"][0], results["ids"][0]))


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

    # Get rag context
    rag_context = get_rag_context(prompt)
    chat_messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\nUse these articles:\n{rag_context}"}
    ] + st.session_state.messages[1:]

    response_stream = st.session_state.client.chat.completions.create(
        model="gpt-5-mini",
        messages=chat_messages,
        stream=True
    )
    with st.chat_message("assistant"):
        response = st.write_stream(response_stream)
    st.session_state.messages.append({"role": "assistant", "content": response})