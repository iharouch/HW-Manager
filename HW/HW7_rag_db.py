import pandas as pd
from openai import OpenAI
import sys
import chromadb
import streamlit as st

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="./ChromaDB_News")
collection = chroma_client.get_or_create_collection(name="HW7_NewsCollection")

# Load CSV
df = pd.read_csv("./HW-7-Data/news.csv")

# Function to add articles
def add_article_to_collection(collection, text, metadata, article_id):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    collection.add(
        documents=[text],
        ids=[article_id],
        embeddings=[embedding],
        metadatas=[metadata]
    )

# Add data in database
for idx, row in df.iterrows():
    text = row['content']
    metadata = {
        "title": row.get('title', ''),
        "date": row.get('date', ''),
        "topic": row.get('topic', '')
    }
    add_article_to_collection(collection, text, metadata, str(idx))

print("RAG DB completed successfully")