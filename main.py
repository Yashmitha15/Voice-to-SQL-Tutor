from fastapi import FastAPI
from dotenv import load_dotenv
import os

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

app = FastAPI()

# 2. Setup the AI (Gemini 3.1 Flash Lite)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", 
    google_api_key=api_key
)

# --- PERFORMANCE UPDATE: Pre-loading for Speed ---
print("⏳ Initializing Vector Database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists("faiss_index"):
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("✅ System Ready.")
else:
    vector_db = None
    print("❌ ERROR: faiss_index not found. Run ingest.py first!")

@app.get("/generate-sql")
def generate(question: str):
    try:
        # --- SECURITY LAYER ---
        forbidden_words = ["drop", "delete", "truncate", "update", "alter"]
        if any(word in question.lower() for word in forbidden_words):
            return {"answer": "🚫 **Security Block**: For safety, I only allow 'SELECT' (Read-Only) queries."}
        
        if vector_db is None:
            return {"answer": "Error: Vector database not loaded."}
        
        # --- RETRIEVAL (RAG) ---
        # k=1 is faster and prevents the LLM from getting confused by too much context
        search_results = vector_db.similarity_search(question, k=1)
        retrieved_context = search_results[0].page_content
        
        # --- SMART TUTOR PROMPT ---
        template = """
        You are an expert SQL Tutor. 
        Context Schema: {context}
        
        User Question: {question}
        
        Follow this EXACT Markdown format for the UI to display correctly:
        
        ### 🔍 SQL Query
        ```sql
        [Write the SQL here]
        ```
        
        ### 💡 Explanation
        [Write a clear 1-2 sentence explanation]
        
        ### 📊 Sample Result Table
        | Column | Column |
        | :--- | :--- |
        | Data | Data |
        
        ---
        *Tutor Tip: Don't forget the semicolon (;)*
        """
        
        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm
        response = chain.invoke({"context": retrieved_context, "question": question})
        
        return {"answer": response.content}

    except Exception as e:
        print(f"Backend Error: {e}")
        return {"answer": f"Backend Error: {str(e)}"}