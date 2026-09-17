import os
import sys
from io import BytesIO
from dotenv import load_dotenv
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from langchain_groq import ChatGroq 
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

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="Voice-to-SQL Tutor", page_icon="🎙️", layout="wide")
load_dotenv()

st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #1e2124; }
    .stChatMessage { border-radius: 15px; padding: 15px; margin-bottom: 10px; }
    .stButton>button { border-radius: 10px; border: 1px solid #4e4e4e; }
    code { color: #50fa7b; } 
    </style>
    """, unsafe_allow_html=True)

# --- 2. API KEY SETUP ---
env_groq_key = os.getenv("GROQ_API_KEY", "")
if not env_groq_key and hasattr(st, "secrets"):
    try:
        env_groq_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        env_groq_key = ""

# --- 3. FAISS INDEX LOADER/BUILDER ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_db():
    try:
        embeddings = get_embeddings()
        if not os.path.exists("faiss_index"):
            from ingest import create_vector_db   
            create_vector_db()
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"❌ Could not build or load FAISS index: {str(e)}")
        return None

vector_db = load_db()

# --- 4. AUDIO PROCESSING ---
def process_audio(audio_bytes):
    if not audio_bytes:
        return None
    r = sr.Recognizer()
    # Try direct reading as WAV
    try:
        with sr.AudioFile(BytesIO(audio_bytes)) as source:
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.record(source)
        return r.recognize_google(audio)
    except Exception:
        # Fallback with pydub conversion to standard WAV format
        try:
            from pydub import AudioSegment
            sound = AudioSegment.from_file(BytesIO(audio_bytes))
            wav_io = BytesIO()
            sound.export(wav_io, format="wav")
            wav_io.seek(0)
            with sr.AudioFile(wav_io) as source:
                r.adjust_for_ambient_noise(source, duration=0.2)
                audio = r.record(source)
            return r.recognize_google(audio)
        except Exception:
            return None

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    groq_api_key_input = st.text_input(
        "Groq API Key",
        value=env_groq_key,
        type="password",
        placeholder="gsk_...",
        help="Get your free API key at https://console.groq.com/keys"
    )
    
    if not groq_api_key_input:
        st.info("💡 [Get a free Groq API Key](https://console.groq.com/keys)")

    st.markdown("---")
    st.markdown("## 🎙️ AI Features")
    st.markdown("### Voice Query")
    st.info("Tip: Click, speak, then click again to stop.")
    
    audio_bytes = audio_recorder(text="Click to speak", icon_size="2x", neutral_color="#6aa36f")
    
    voice_prompt = None
    if audio_bytes:
        # Avoid repeated transcription of the same audio on subsequent reruns
        if "last_recorded_audio" not in st.session_state or st.session_state["last_recorded_audio"] != audio_bytes:
            st.session_state["last_recorded_audio"] = audio_bytes
            with st.spinner("Transcribing voice input..."):
                voice_prompt = process_audio(audio_bytes)
                if voice_prompt:
                    st.success(f"Heard: {voice_prompt}")
                else:
                    st.warning("Could not recognize speech. Please try speaking again or type your question.")

    st.write("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        if "last_recorded_audio" in st.session_state:
            del st.session_state["last_recorded_audio"]
        st.rerun()
    
    st.markdown("### 📄 Export")
    if "messages" in st.session_state and st.session_state.messages:
        chat_log = "\n\n".join([f"[{msg['role'].upper()}]:\n{msg['content']}" for msg in st.session_state.messages])
        st.download_button(
            label="📥 Download Chat Log",
            data=chat_log,
            file_name="sql_tutor_chat_log.txt",
            mime="text/plain"
        )

# Check active API Key
active_groq_key = groq_api_key_input.strip() if groq_api_key_input else None

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
        if not active_groq_key:
            error_msg = "⚠️ **Missing or Invalid API Key**: Please enter a valid Groq API Key in the sidebar or `.env` / Streamlit Secrets. You can create one for free at [console.groq.com/keys](https://console.groq.com/keys)."
            st.warning(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        elif not vector_db:
            st.error("❌ Database schema (faiss_index) missing! Please build or load your FAISS index.")
        else:
            with st.spinner("Thinking..."):
                try:
                    llm = ChatGroq(
                        model="openai/gpt-oss-20b",
                        groq_api_key=active_groq_key,
                        temperature=0
                    )

                    docs = vector_db.similarity_search(final_prompt, k=1)
                    context = docs[0].page_content if docs else "No schema context found."

                    template = """You are an expert SQL Tutor.
Context Schema:
{context}

User Question: {question}

Format your response exactly as follows:
### 📌 SQL Query
```sql
[SQL Here]
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

                    response = chain.invoke({"context": context, "question": final_prompt})
                    st.markdown(response.content)
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                except Exception as e:
                    err_str = str(e)
                    if "401" in err_str or "invalid_api_key" in err_str.lower() or "Invalid API Key" in err_str:
                        error_msg = "❌ **Invalid API Key (Error 401)**: Your Groq API key is invalid or has expired/been revoked. Please create a new key at [console.groq.com/keys](https://console.groq.com/keys) and enter it in the sidebar or update your `.env` / Streamlit Secrets."
                    else:
                        error_msg = f"AI Error: {err_str}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
