import streamlit as st
import numpy as np
import os
import time
from google import genai
from openai import OpenAI

# ────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────
GAME_NAME = "ISO Chatbot CF"

# RAG & Embedding (Powered by Google Gemini Free Tier)
EMBEDDING_MODEL = "gemini-embedding-001"
CHUNK_SIZE_WORDS = 150
CHUNK_OVERLAP_WORDS = 30
TOP_K_CHUNKS = 5

# Chat Models (Powered by OpenRouter Free Tier)
PRIMARY_MODEL = "openrouter/free" 
FALLBACK_MODEL = "openrouter/free"

st.set_page_config(page_title=f"{GAME_NAME} Expert", layout="wide")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stAppDeployButton {display:none;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title(f"🕹️ {GAME_NAME} Assistant")
st.caption("Ask anything — powered by OpenRouter & Semantic RAG")

# ────────────────────────────────────────
# RAG: CHUNKING & EMBEDDING (GEMINI)
# ────────────────────────────────────────
def get_chunks(text, chunk_size, overlap):
    """Splits text into overlapping word chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
    return chunks

@st.cache_resource(show_spinner="Forging Knowledge Base...")
def load_and_embed_knowledge():
    try:
        with open("knowledge.txt", "r", encoding="utf-8") as f:
            raw_text = f.read().strip()
    except FileNotFoundError:
        return [], None

    if not raw_text:
        return [], None

    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    chunks = get_chunks(raw_text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
    
    BATCH_SIZE = 100
    all_embeddings = []
    
    progress_text = "Embedding Knowledge Base..."
    my_bar = st.progress(0, text=progress_text)
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch
            )
            all_embeddings.extend([e.values for e in response.embeddings])
            
            progress = min((i + BATCH_SIZE) / len(chunks), 1.0)
            my_bar.progress(progress, text=f"{progress_text} ({int(progress*100)}%)")
            time.sleep(1.5) 
            
        except Exception as e:
            st.error(f"Error during embedding batch {i} to {i+BATCH_SIZE}: {str(e)}")
            st.stop()
            
    my_bar.empty()
    embeddings = np.array(all_embeddings)
    return chunks, embeddings

chunks, knowledge_embeddings = load_and_embed_knowledge()

if not chunks or knowledge_embeddings is None:
    st.error("knowledge.txt not found or empty! Please add it to the repo root.")
    st.stop()

def retrieve_context(query, client, top_k=TOP_K_CHUNKS):
    """Finds the top_k most relevant chunks using cosine similarity."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
    )
    query_embedding = np.array(response.embeddings[0].values)
    
    norms = np.linalg.norm(knowledge_embeddings, axis=1) * np.linalg.norm(query_embedding)
    similarities = np.dot(knowledge_embeddings, query_embedding) / norms
    
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

# ────────────────────────────────────────
# CHAT INTERFACE (OPENROUTER WITH FALLBACK)
# ────────────────────────────────────────
or_client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=st.secrets["OPENROUTER_API_KEY"],
)
gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(f"Ask about {GAME_NAME}..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # 1. Retrieve the best lore snippets
        retrieved_chunks = retrieve_context(prompt, gemini_client)
        context_str = "\n\n".join([f"[Source {i+1}]: {c}" for i, c in enumerate(retrieved_chunks)])
        
        DYNAMIC_SYSTEM_PROMPT = f"""You are the ultimate expert and lore master for {GAME_NAME}.
You ONLY use knowledge from the context snippets below. 
Never make up facts, never say you don't know — redirect politely to in-game info if needed.
Be immersive: use game-style language, nicknames, lore flavor. Keep answers concise.

=== RELEVANT LORE SNIPPETS ===
{context_str}
==============================
"""
        
        # 2. Format history for OpenRouter
        formatted_messages = [{"role": "system", "content": DYNAMIC_SYSTEM_PROMPT}]
        for msg in st.session_state.messages:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Helper function to generate stream
        def generate_response(model_name):
            return or_client.chat.completions.create(
                model=model_name,
                messages=formatted_messages,
                temperature=0.75,
                max_tokens=600,
                stream=True,
            )

        full_response = ""
        try:
            # Try Primary Model First (Llama 3.1)
            stream = generate_response(PRIMARY_MODEL)
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
                    
        except Exception as e:
            # If Primary fails, warn user and use Fallback Model (Gemma 2)
            st.warning(f"Network busy. Rerouting request...")
            full_response = ""
            try:
                stream = generate_response(FALLBACK_MODEL)
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
            except Exception as fallback_error:
                st.error(f"Both API models failed. Error: {str(fallback_error)}")
                st.stop()

        message_placeholder.markdown(full_response)
        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})