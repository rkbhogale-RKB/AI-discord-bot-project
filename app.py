# ────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────
GAME_NAME = "CounterForce GPS RTS"           # Change this
MODEL = "llama-3.1-8b-instant"              # or whichever you prefer

# Load knowledge from external file (relative path from repo root)
try:
    with open("knowledge.txt", "r", encoding="utf-8") as f:
        knowledge_base = f.read().strip()
except FileNotFoundError:
    knowledge_base = "No knowledge base found. Please add knowledge.txt to the repo."
    st.error("knowledge.txt not found in repo root!")

SYSTEM_PROMPT = f"""You are the ultimate expert and lore master for {GAME_NAME}.
You ONLY use knowledge from the official game sources in the knowledge base below. 
Never make up facts, never say you don't know — redirect politely to in-game info if needed.
Be immersive: use game-style language, nicknames, lore flavor. Keep answers helpful, concise but detailed when asked.

=== FULL GAME KNOWLEDGE BASE ===
{knowledge_base}

Current date is irrelevant — answer as if the game world is eternal.
"""

# ────────────────────────────────────────
# APP
# ────────────────────────────────────────
st.set_page_config(page_title=f"{GAME_NAME} Expert", layout="wide")

st.title(f"🕹️ {GAME_NAME} Lore & Gameplay Expert")
st.caption("Ask anything — quests, builds, lore, bosses, secrets...")

# Initialize Groq client from secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Chat history (persists in session)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input(f"Ask about {GAME_NAME}..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            stream = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *st.session_state.messages
                ],
                model=MODEL,
                temperature=0.75,
                max_tokens=600,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error: {str(e)}\nCheck if your GROQ_API_KEY is valid and has credits.")
