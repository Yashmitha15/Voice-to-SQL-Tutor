import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

def create_vector_db(schema_file: str = "database_schema.txt", index_dir: str = "faiss_index"):
    """
    Build a FAISS vector database from a schema text file.
    Args:
        schema_file: Path to the schema text file.
        index_dir: Directory to save the FAISS index.
    """
    # 1. Load the schema file
    if not os.path.exists(schema_file):
        print(f" ERROR: {schema_file} not found!")
        return

    print(f"📂 Loading schema from: {schema_file}")
    loader = TextLoader(schema_file)
    documents = loader.load()

    # 2. Split the text into small chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print(f" Split into {len(docs)} chunks.")

    # 3. Create the embeddings
    print(" Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Create and save the FAISS index
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(index_dir)
    print(f" SUCCESS: FAISS index created at '{index_dir}'")

if __name__ == "__main__":
    create_vector_db()
