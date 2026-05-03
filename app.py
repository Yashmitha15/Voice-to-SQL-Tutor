import streamlit as st
import os

st.set_page_config(page_title="SQL Tutor", page_icon="🎙️")

st.title("🎙️ Voice-to-SQL Tutor")

# --- WEEK 1: KNOWLEDGE BASE SECTION ---
st.sidebar.header("Data Knowledge Base")

# Check if the schema file exists
if os.path.exists("database_schema.txt"):
    with open("database_schema.txt", "r") as f:
        schema = f.read()
    st.sidebar.success("✅ Database Schema Loaded")
    with st.sidebar.expander("View Schema Details"):
        st.text(schema)
else:
    st.sidebar.error("❌ Schema file not found!")

# --- WEEK 1: CHAT INTERFACE ---
st.info("Welcome! I am your GenAI SQL Tutor. Ask me to write a query for you.")

if prompt := st.chat_input("Ex: Show me all employees in Sales"):
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        st.write("I have access to your schema. In Week 2, I will generate the SQL for this!")