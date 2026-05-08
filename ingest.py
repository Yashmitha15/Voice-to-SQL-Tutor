import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

def create_vector_db():
    # 1. Load the schema file
    if not os.path.exists("database_schema.txt"):
        print("❌ ERROR: database_schema.txt not found!")
        return

    loader = TextLoader("database_schema.txt")
    documents = loader.load()

    # 2. Split the text into small chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # 3. Create the "Embeddings" (Using the stable 2026 standard)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Create and save the FAISS index
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")
    print("✅ SUCCESS: faiss_index created!")

if __name__ == "__main__":
    create_vector_db()