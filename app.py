import streamlit as st
import os
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from io import BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="SQL Tutor", page_icon="🎙️", layout="wide")

# Custom CSS for the Sidebar and Chat Bubbles
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #1e2124; }
    .stChatMessage { border-radius: 15px; padding: 15px; margin-bottom: 10px; }
    .stButton>button { border-radius: 10px; border: 1px solid #4e4e4e; }
    code { color: #50fa7b; } 
    </style>
    """, unsafe_allow_html=True)

# --- 2. AI BACKEND SETUP ---
# Ensure your GOOGLE_API_KEY is in Streamlit Secrets
api_key = st.secrets["GOOGLE_API_KEY"]
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return None

vector_db = load_db()

# --- 3. AUDIO PROCESSING ---
def process_audio(audio_bytes):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(BytesIO(audio_bytes)) as source:
            r.adjust_for_ambient_noise(source)
            audio = r.record(source)
        return r.recognize_google(audio)
    except: return None

# --- 4. SIDEBAR (MATCHING YOUR SCREENSHOT) ---
with st.sidebar:
    st.markdown("## ✨ AI Features")
    st.markdown("### 🎙️ Voice Query")
    st.info("Tip: Click, speak clearly, then click again to stop.")
    
    audio_bytes = audio_recorder(text="Click to speak", icon_size="2x", neutral_color="#6aa36f")
    
    voice_prompt = None
    if audio_bytes:
        voice_prompt = process_audio(audio_bytes)
        if voice_prompt:
            st.success(f"Heard: {voice_prompt}")

    st.write("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("### 📊 Share")
    if st.button("📄 Download Chat Log"):
        st.write("Log Prepared!")

# --- 5. CHAT DISPLAY & LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. INPUT HANDLING ---
text_prompt = st.chat_input("Ask about your database...")
final_prompt = voice_prompt if (voice_prompt and voice_prompt != "None") else text_prompt

if final_prompt:
    st.session_state.messages.append({"role": "user", "content": final_prompt})
    with st.chat_message("user"):
        st.markdown(final_prompt)

    with st.chat_message("assistant"):
        if vector_db:
            # RAG: Get schema context
            docs = vector_db.similarity_search(final_prompt, k=1)
            context = docs[0].page_content
            
            # THE PROMPT (Generates the specific UI sections in your screenshot)
            template = """
            You are an expert SQL Tutor. 
            Context: {context}
            
            Question: {question}
            
            Provide your response in this EXACT format:
            ### 🔍 SQL Query
            ```sql
            [Your Query Here]
            ```
            
            ### 💡 Explanation
            [1-2 sentences explaining how the query works]
            
            ### 📊 Sample Result Table
            | name | department | salary |
            | :--- | :--- | :--- |
            | [Example Name] | [Example Dept] | [Example Value] |
            """
            
            prompt = PromptTemplate.from_template(template)
            chain = prompt | llm
            response = chain.invoke({"context": context, "question": final_prompt})
            
            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
        else:
            st.error("Database schema not found. Please run ingest.py.")