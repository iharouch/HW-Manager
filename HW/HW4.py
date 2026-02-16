import streamlit as st
from openai import OpenAI
import sys
import chromadb
from pathlib import Path
from bs4 import BeautifulSoup

# A fix for working with ChromaDB on Streamlit
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#Create ChromaDB client and collection
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = chromadb.PersistentClient(path="./ChromaDB_for_Lab")
if 'HW4_VectorDB' not in st.session_state:
    st.session_state.HW4_VectorDB = st.session_state.chroma_client.get_or_create_collection(name="HW4Collection")

collection = st.session_state.HW4_VectorDB 

### Using Chroma DB with OpenAI Embeddings ###
#Create an OpenAI client
openAI_model = "gpt-5-mini"

if 'client' not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.client = OpenAI(api_key=api_key)

### Extract text from HTML and populate collection ###
def extract_text_from_html(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style"]):
        s.extract()
    return soup.get_text(" ", strip=True)

def chunk_text(text, size=800, overlap=150):
    words = text.split()
    step = size - overlap
    return [
        " ".join(words[i:i+size])
        for i in range(0, len(words), step)
    ]

def load_htmls_to_collection(folder_path, collection):
    for file in Path(folder_path).glob("*.html"):
        text = extract_text_from_html(file)
        chunks = chunk_text(text)

        docs = []
        ids = []

        for i, chunk in enumerate(chunks, start=1):
            if chunk.strip():
                docs.append(chunk)
                file_name = f"{file.stem}_chunk{i}" # Unique ID for each chunk
                ids.append(file_name)

        if not docs:
            continue

        # batch embedding
        response = st.session_state.client.embeddings.create(
            model="text-embedding-3-small",
            input=docs
        )

        embeddings = [d.embedding for d in response.data]

        # add to collection
        collection.add(
            documents=docs,
            embeddings=embeddings,
            ids=ids
        )

# Check if collection is empty and load HTML files
if st.session_state.HW4_VectorDB.count() == 0:
    load_htmls_to_collection("./HW/HW-4-Data/", st.session_state.HW4_VectorDB)

# System prompt to guide bot behavior
SYSTEM_PROMPT = """You are a helpful Q&A chatbot. Follow these rules STRICTLY:
1. When answering a NEW QUESTION, provide a clear, concise answer that a 10-year-old can understand
2. Use simple words and avoid technical terms. Explain complex ideas with everyday examples.
3. ALWAYS end your answer with: "Do you want more info?"
4. If the user says "Yes" or "yes", provide additional detailed information and ALWAYS end with: "Do you want more info?"
5. If the user says "No" or "no", respond with: "How can I help you with something else?"
Keep responses focused, helpful, and easy to understand."""

### Main App ###
st.title("HW 4: Chatbot using RAG")

# Initialize messages with system prompt (protected from removal)
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "How can I help you?"}
    ]

# Get user input
if prompt := st.chat_input("What do you need help with?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = st.session_state.client
    response = client.embeddings.create(
        input = prompt,
        model = "text-embedding-3-small"
    )

    #Get the embedding vector
    query_embedding = response.data[0].embedding

    #Get the text related to this question (this prompt)
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results=3 #The number of closest documents to return
    )

    #Build RAG context
    rag_context = "\n\n".join(
    f"Source: {doc_id}\n{doc}" # Include source in context to know which document it is
    for doc, doc_id in zip(results["documents"][0], results["ids"][0])) #Zip to get both the document and its ID
    rag_prompt = f"""Use the following context from the documents to answer the question if it helps. If you use it, clearly say so in the answer.
    Document Information: {rag_context}, user question: {prompt}"""

    stream = client.chat.completions.create(
        model=openAI_model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rag_prompt}],
        stream=True
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})