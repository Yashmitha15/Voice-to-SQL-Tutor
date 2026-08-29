import streamlit as st
import os
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from io import BytesIO
from langchain_groq import ChatGroq 
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="SQL Tutor", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #1e2124; }
    .stChatMessage { border-radius: 15px; padding: 15px; margin-bottom: 10px; }
    .stButton>button { border-radius: 10px; border: 1px solid #4e4e4e; }
    code { color: #50fa7b; } 
    </style>
    """, unsafe_allow_html=True)

# --- 2. AI BACKEND SETUP (GROQ) ---
groq_key = st.secrets.get("GROQ_API_KEY", None)
if not groq_key:
    st.error(" Missing GROQ_API_KEY in Streamlit Secrets. Please add it in Settings → Secrets.")
    st.stop()

# ✅ Updated Line 21:
llm = ChatGroq(
    model="llama-3.1-70b-versatile", 
    groq_api_key=groq_key,
    temperature=0
)

# --- 3. FAISS INDEX LOADER/BUILDER ---
# @st.cache_resource
# def load_db():
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     if os.path.exists("faiss_index"):
#         return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     else:
#         # Auto-build FAISS if missing
#         try:
#             from ingest import create_vector_db   # make sure ingest.py has this function
#             create_vector_db()
#             return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#         except Exception as e:
#             st.error(f" Could not build FAISS index: {str(e)}")
#             return None

# vector_db = load_db()
@st.cache_resource
def get_embeddings():
    # Caches the model separately so it doesn't freeze the DB load
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_db():
    try:
        embeddings = get_embeddings()
        if os.path.exists("faiss_index"):
            return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        else:
            # Auto-build FAISS if missing
            from ingest import create_vector_db   
            create_vector_db()
            return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f" Could not build or load FAISS index: {str(e)}")
        return None

vector_db = load_db()

# --- 4. AUDIO PROCESSING ---
def process_audio(audio_bytes):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(BytesIO(audio_bytes)) as source:
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.record(source)
        return r.recognize_google(audio)
    except Exception:
        return None

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("##  AI Features")
    st.markdown("###  Voice Query")
    st.info("Tip: Click, speak, then click again to stop.")
    
    audio_bytes = audio_recorder(text="Click to speak", icon_size="2x", neutral_color="#6aa36f")
    
    voice_prompt = None
    if audio_bytes:
        with st.spinner("Transcribing..."):
            voice_prompt = process_audio(audio_bytes)
            if voice_prompt:
                st.success(f"Heard: {voice_prompt}")

    st.write("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("###  Share")
    if st.button("📄 Download Chat Log"):
        st.write("Log Prepared!")

# --- 6. CHAT DISPLAY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 7. INPUT HANDLING ---
text_prompt = st.chat_input("Ask about your database...")
final_prompt = voice_prompt if (voice_prompt and voice_prompt != "None") else text_prompt

if final_prompt:
    st.session_state.messages.append({"role": "user", "content": final_prompt})
    with st.chat_message("user"):
        st.markdown(final_prompt)

    with st.chat_message("assistant"):
        if vector_db:
            docs = vector_db.similarity_search(final_prompt, k=1)
            context = docs[0].page_content if docs else "No schema context found."

            template = """
            You are an expert SQL Tutor. 
            Context Schema: {context}
            
            User Question: {question}
            
            Format exactly:
            ###  SQL Query
            ```sql
            [SQL Here]
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

            try:
                response = chain.invoke({"context": context, "question": final_prompt})
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as e:
                st.error(f"AI Connection Error: {str(e)}")
        else:
            st.error(" Database schema (faiss_index) missing! Please build or load your FAISS index.")
