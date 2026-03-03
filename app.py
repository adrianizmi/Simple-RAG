import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from rag_functions import load_and_vectorize, search_best_context

# 1. AI Configuration
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 2. Web visualization configuration
st.set_page_config(page_title="My personal RAG", page_icon="🧠")
st.title("🧠 My Document Assistant (RAG)")
st.subheader("Ask anything you want about your PDFs and TXT files")

@st.cache_resource
def prepare_data():
    return load_and_vectorize()

with st.spinner("Reading documents and generating vectors..."):
    texts, embeddings = prepare_data()

# 4. Chat System
if "messages" not in st.session_state:
    st.session_state.messages = []

# Draw previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What says my PDF about...?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Response generation
    with st.chat_message("assistant"):
        context = search_best_context(prompt, texts, embeddings)
        
        complete_promp = f"""
        Use the following context to answer the question.
        CONTEXT: {context}
        QUESTION: {prompt}
        """
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": complete_promp}],
            model="llama-3.3-70b-versatile"
        )
        
        response = response.choices[0].message.content
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})