from fastapi import FastAPI
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. ENV + FASTAPI INIT ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

app = FastAPI()

# --- 2. LLM BACKEND (Gemini 1.5 Flash) ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0
)

# --- VECTOR DB SETUP ---
print(" Initializing Vector Database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
            return {"answer": " **Security Block**: Read-Only access only."}

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
        ###  SQL Query
        ```sql
        [SQL]
        ```
        ###  Explanation
        [1-2 sentences]
        ###  Sample Result Table
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
