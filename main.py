from fastapi import FastAPI
from dotenv import load_dotenv
import os
import sys
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Ensure UTF-8 output on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --- 1. ENV + FASTAPI INIT ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

app = FastAPI(title="Voice-to-SQL Tutor Backend API")

# --- 2. LLM BACKEND INITIALIZATION ---
llm = None
if google_api_key:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            temperature=0
        )
        print(" Using Google Gemini backend.")
    except Exception as e:
        print(f" Failed to initialize Google Gemini: {e}")

if llm is None and groq_api_key:
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            groq_api_key=groq_api_key,
            temperature=0
        )
        print(" Using Groq backend.")
    except Exception as e:
        print(f" Failed to initialize Groq: {e}")

# --- 3. VECTOR DB SETUP ---
print(" Initializing Vector Database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if not os.path.exists("faiss_index"):
    try:
        from ingest import create_vector_db
        create_vector_db()
    except Exception as e:
        print(f" Could not build FAISS index: {e}")

if os.path.exists("faiss_index"):
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print(" System Ready.")
else:
    vector_db = None
    print(" ERROR: faiss_index not found.")

# --- 4. ROUTE: SQL GENERATION ---
@app.get("/generate-sql")
def generate(question: str):
    try:
        # Security filter: block destructive queries
        forbidden_words = ["drop", "delete", "truncate", "update", "alter"]
        if any(word in question.lower() for word in forbidden_words):
            return {"answer": "🚫 **Security Block**: Read-Only access only."}

        if llm is None:
            return {"answer": "Error: No LLM API key configured (set GOOGLE_API_KEY or GROQ_API_KEY in .env)."}

        if vector_db is None:
            return {"answer": "Error: Vector database not loaded."}

        # Retrieve schema context
        search_results = vector_db.similarity_search(question, k=1)
        retrieved_context = search_results[0].page_content if search_results else "No schema context found."

        # Prompt template
        template = """
        You are an expert SQL Tutor.
        Context Schema: {context}
        User Question: {question}

        Format exactly:
        ### 📌 SQL Query
        ```sql
        [SQL]
        ```
        ### 💡 Explanation
        [1-2 sentences]
        ### 📊 Sample Result Table
        | Col | Col |
        | :--- | :--- |
        | Data | Data |
        """

        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm

        response = chain.invoke({"context": retrieved_context, "question": question})

        return {"answer": response.content}

    except Exception as e:
        return {"answer": f"Backend Error: {str(e)}"}
