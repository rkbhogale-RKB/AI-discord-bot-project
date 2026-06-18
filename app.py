import streamlit as st
import numpy as np
import os
import time
from google import genai
from google.genai import types

# ────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────
GAME_NAME = "ISO Chatbot CF" 
# Use Gemini 3.5 Flash for rapid text generation
MODEL = "gemini-3.5-flash"
# Use Google's standard embedding model
EMBEDDING_MODEL = "text-embedding-004"

# RAG Configuration
CHUNK_SIZE_WORDS = 150
CHUNK_OVERLAP_WORDS = 30
TOP_K_CHUNKS = 3

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
st.caption("Ask anything — powered by Gemini & Semantic RAG")

# ────────────────────────────────────────
# RAG: CHUNKING & EMBEDDING
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
    
    # 1. Chunk the text
    chunks = get_chunks(raw_text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
    
    # 2. Batch Embed to avoid ClientError (Payload Too Large)
    BATCH_SIZE = 100 # Gemini API limit for embedding arrays
    all_embeddings = []
    
    # Create a progress bar in the UI
    progress_text = "Embedding Knowledge Base..."
    my_bar = st.progress(0, text=progress_text)
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch
            )
            # Extract and store the vectors
            all_embeddings.extend([e.values for e in response.embeddings])
            
            # Update progress bar
            progress = min((i + BATCH_SIZE) / len(chunks), 1.0)
            my_bar.progress(progress, text=f"{progress_text} ({int(progress*100)}%)")
            
            # Sleep briefly to respect free-tier rate limits (RPM)
            time.sleep(1.5) 
            
        except Exception as e:
            st.error(f"Error during embedding batch {i} to {i+BATCH_SIZE}: {str(e)}")
            st.stop()
            
    my_bar.empty() # Remove progress bar when done
    
    # 3. Convert to NumPy array for fast math
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
# CHAT INTERFACE
# ────────────────────────────────────────
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

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
        
        # 1. Retrieve the best lore snippets based on the prompt
        retrieved_chunks = retrieve_context(prompt, client)
        context_str = "\n\n".join([f"[Source {i+1}]: {c}" for i, c in enumerate(retrieved_chunks)])
        
        # 2. Inject ONLY the relevant snippets into the System Instruction
        DYNAMIC_SYSTEM_PROMPT = f"""You are the ultimate expert and lore master for {GAME_NAME}.
You ONLY use knowledge from the context snippets below. 
Never make up facts, never say you don't know — redirect politely to in-game info if needed.
Be immersive: use game-style language, nicknames, lore flavor. Keep answers concise.

=== RELEVANT LORE SNIPPETS ===
{context_str}
==============================
Current date is irrelevant — answer as if the game world is eternal.
"""
        
        try:
            # 3. Format history for Gemini (Requires 'user' and 'model' roles)
            formatted_messages = []
            for msg in st.session_state.messages:
                role = "model" if msg["role"] == "assistant" else "user"
                formatted_messages.append({"role": role, "parts": [{"text": msg["content"]}]})
            
            # 4. Stream generation
            config = types.GenerateContentConfig(
                system_instruction=DYNAMIC_SYSTEM_PROMPT,
                temperature=0.75,
                max_output_tokens=600,
            )
            
            stream = client.models.generate_content_stream(
                model=MODEL,
                contents=formatted_messages,
                config=config,
            )

            full_response = ""
            for chunk in stream:
                if chunk.text is not None:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error: {str(e)}\nCheck your GEMINI_API_KEY and Streamlit logs.")