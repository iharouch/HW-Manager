import streamlit as st
from openai import OpenAI
import sys
import chromadb
from pathlib import Path
import fitz

# A fix for working with ChromaDB on Streamlit
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#Create ChromaDB client and collection
if 'HW5_VectorDB' not in st.session_state:
    chroma_client = chromadb.PersistentClient(path="./ChromaDB_for_Lab")
    st.session_state.HW5_VectorDB = chroma_client.get_or_create_collection(name="HW5Collection")

collection = st.session_state.HW5_VectorDB

### Using Chroma DB with OpenAI Embeddings ###
#Create an OpenAI client
openAI_model = "gpt-5-mini"

if 'client' not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.client = OpenAI(api_key=api_key)

#A function that will add documents to collection
def add_to_collection(collection, text, file_name):
    """
    Collection = collection, already defined
    text = extarcted text from PDF
    file_name = name of the PDF file
    Embeddings inserted into the collection from OpenAI
    """
    #Create an embedding
    client = st.session_state.client
    response = client.embeddings.create(
        input = text,
        model = "text-embedding-3-small"
    )

    #Get the embedding vector
    embedding = response.data[0].embedding

    #Add embedding and document to ChromaDB
    collection.add(
        documents = [text],
        ids = file_name,
        embeddings = [embedding]
    )

### Extract text from PDF ###
def extract_text_from_pdf(file_path):
    """
    file_path = path to PDF file
    returns extracted text from PDF
    """
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

### Populate collection with PDFs ###
def load_pdfs_to_collection(folder_path, collection):
    """
    folder_path = path to folder containing PDFs
    collection = ChromaDB collection
    """
    pdf_files = Path(folder_path).glob("*.pdf")
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        add_to_collection(collection, text, pdf_file.stem)

# Function that takes a query input from the LLM and returns relevant information from ChromaDB collection
def relevant_course_info(query):
    """
    Takes a query and returns relevant documents from collection
    """
    client = st.session_state.client

    #Create embedding for query
    response = client.embeddings.create(
        input = query,
        model = "text-embedding-3-small"
    )

    query_embedding = response.data[0].embedding

    #Get the text related to this question (this prompt)
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results=3 #The number of closest documents to return
    )

    # Build RAG context
    rag_context = "\n\n".join(
        f"Source: {doc_id}\n{doc}" # Include source in context to know which PDF it is
        for doc, doc_id in zip(results["documents"][0], results["ids"][0])) #Zip to get both the document and its ID

    return rag_context

#Check if collection is empty and load PDFs
if st.session_state.HW5_VectorDB.count() == 0:
    loaded = load_pdfs_to_collection("./HW/HW-5-Data/", st.session_state.HW5_VectorDB)

# System prompt to guide bot behavior
SYSTEM_PROMPT = """You are a helpful Q&A chatbot. Follow these rules STRICTLY:
1. When answering a NEW QUESTION, provide a clear, concise answer that a 10-year-old can understand
2. Use simple words and avoid technical terms. Explain complex ideas with everyday examples.
3. ALWAYS end your answer with: "Do you want more info?"
4. If the user says "Yes" or "yes", provide additional detailed information and ALWAYS end with: "Do you want more info?"
5. If the user says "No" or "no", respond with: "How can I help you with something else?"
6. If you use the rag context, make sure to indicate that you did so.
Keep responses focused, helpful, and easy to understand."""

### Main App ###
st.title("HW 5: Short-term Memory Chatbot")

# Initialize messages with system prompt (protected from removal)
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "How can I help you?"}
    ]

# Get user input
if prompt := st.chat_input("What do you need help with?"):
    st.session_state.messages.append({"role": "user", "content": prompt}) #Store message in memory

    with st.chat_message("user"):
        st.markdown(prompt)

    #Call relevant_course_info function
    rag_context = relevant_course_info(prompt)
    
    # Add rag context to message prompt
    st.session_state.messages.append({
        "role": "system",
        "content": f"Relevant PDF information: {rag_context}"})

    # Call OpenAI API
    stream = st.session_state.client.chat.completions.create(
        model=openAI_model,
        messages=st.session_state.messages,
        stream=True
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response}) #Save response to memory