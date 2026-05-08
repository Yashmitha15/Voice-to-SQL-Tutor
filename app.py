import streamlit as st
import os
import requests
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from io import BytesIO

st.set_page_config(page_title="SQL Tutor", page_icon="🎙️", layout="wide")

# Custom Gemini-style styling
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .stButton>button { border-radius: 20px; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎙️ Voice-to-SQL Tutor")

# --- INITIALIZE CHAT MEMORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- VOICE PROCESSING (ENHANCED) ---
def process_audio(audio_bytes):
    r = sr.Recognizer()
    r.dynamic_energy_threshold = True 
    r.pause_threshold = 2.0  # More time for the user to finish sentences
    
    try:
        with sr.AudioFile(BytesIO(audio_bytes)) as source:
            # Calibrate for room noise
            r.adjust_for_ambient_noise(source, duration=1.0)
            audio = r.record(source)
        
        # Convert speech to text
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except Exception:
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("✨ AI Features")
    st.subheader("🎙️ Voice Query")
    
    st.info("Tip: Click, speak clearly, then click again to stop.")
    
    audio_bytes = audio_recorder(
        text="Click to speak",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_size="2x"
    )
    
    voice_prompt = None
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with st.spinner("Processing Voice..."):
            voice_prompt = process_audio(audio_bytes)
            if voice_prompt:
                st.success(f"Heard: {voice_prompt}")

    st.write("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # SHARE FEATURE
    if st.session_state.messages:
        st.subheader("📤 Share")
        chat_log = "--- SQL TUTOR SESSION ---\n\n"
        chat_log += "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
        st.download_button("📄 Download Chat Log", chat_log, file_name="sql_tutor_share.txt")

# --- DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- INPUT LOGIC (Text or Voice) ---
text_prompt = st.chat_input("Ask about your database...")

# Priorities voice if it exists
final_prompt = None
if voice_prompt and voice_prompt not in ["Could not understand audio"]:
    final_prompt = voice_prompt
elif text_prompt:
    final_prompt = text_prompt

if final_prompt:
    # 1. Save and display user message
    st.session_state.messages.append({"role": "user", "content": final_prompt})
    with st.chat_message("user"):
        st.markdown(final_prompt)
    
    # 2. Get AI Response
    with st.chat_message("assistant"):
        with st.spinner("Tutor is thinking..."):
            try:
                url = "http://127.0.0.1:8000/generate-sql"
                # Timeout increased to 120s for stability
                response = requests.get(url, params={"question": final_prompt}, timeout=120)
                
                if response.status_code == 200:
                    raw_answer = response.json()["answer"]
                    
                    # Extraction logic for metadata cleaning
                    if isinstance(raw_answer, list) and len(raw_answer) > 0:
                        clean_text = raw_answer[0].get('text', str(raw_answer))
                    else:
                        clean_text = raw_answer

                    st.markdown(clean_text)
                    
                    # Visualization Trigger
                    if any(word in final_prompt.lower() for word in ["salary", "department", "compare", "sales"]):
                        st.bar_chart({"Dept A": 40, "Dept B": 75, "Dept C": 25, "Dept D": 80})
                    
                    # 3. Save to memory
                    st.session_state.messages.append({"role": "assistant", "content": clean_text})
                else:
                    st.error("Backend Error: Check Terminal 1")
            except Exception as e:
                st.error(f"⚠️ Connection Error: {e}")