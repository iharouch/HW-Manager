import pandas as pd
from openai import OpenAI
import sys
import chromadb
import streamlit as st
from pathlib import Path

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize ChromaDB client and collection
BASE_DIR = Path(__file__).parent
db_path = BASE_DIR / "ChromaDB_News"
db_path.mkdir(exist_ok=True)

chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection(name="HW7_NewsCollection")

# Load CSV
csv_path = BASE_DIR / "HW-7-Data" / "news.csv"

df = pd.read_csv(csv_path)

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
    text = row['Document']
    metadata = {
        "company_name": row.get('company_name', ''),
        "date": row.get('Date', ''),
        "url": row.get('URL', '')
    }
    add_article_to_collection(collection, text, metadata, str(idx))

    print(f"Processed article {idx+1}/{len(df)}")

print("RAG DB completed successfully")