import streamlit as st
import os
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from io import BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. SETUP AI BRAIN (Inside Streamlit) ---
api_key = st.secrets["GOOGLE_API_KEY"] # Pulls from Streamlit "Secrets"
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=api_key)

@st.cache_resource # Keeps the DB in memory so it's fast
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return None

vector_db = load_vector_db()

# --- 2. UI & VOICE SETUP ---
st.set_page_config(page_title="SQL Tutor", page_icon="🎙️")
st.title("🎙️ Voice-to-SQL Tutor")

if "messages" not in st.session_state:
    st.session_state.messages = []

def process_audio(audio_bytes):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(BytesIO(audio_bytes)) as source:
            r.adjust_for_ambient_noise(source)
            audio = r.record(source)
        return r.recognize_google(audio)
    except: return None

# --- 3. SIDEBAR ---
with st.sidebar:
    audio_bytes = audio_recorder(text="Click to speak", icon_size="2x")
    voice_prompt = process_audio(audio_bytes) if audio_bytes else None
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- 4. CHAT LOGIC ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

text_prompt = st.chat_input("Ask about your database...")
final_prompt = voice_prompt if (voice_prompt and voice_prompt != "None") else text_prompt

if final_prompt:
    st.session_state.messages.append({"role": "user", "content": final_prompt})
    with st.chat_message("user"): st.markdown(final_prompt)

    with st.chat_message("assistant"):
        if vector_db:
            context = vector_db.similarity_search(final_prompt, k=1)[0].page_content
            template = "You are a SQL Tutor. Schema: {context}\nQuestion: {question}\nGive SQL and explanation."
            prompt = PromptTemplate.from_template(template)
            chain = prompt | llm
            response = chain.invoke({"context": context, "question": final_prompt})
            
            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
        else:
            st.error("Vector database (faiss_index) missing!")