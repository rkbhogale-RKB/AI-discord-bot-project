# main.py
import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
from groq import Groq
from flask import Flask
import threading

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

client = Groq(api_key=GROQ_API_KEY)

# Load knowledge base once at start
with open("knowledge.txt", "r", encoding="utf-8") as f:
    GAME_KNOWLEDGE = f.read().strip()

print(f"Loaded knowledge base: {len(GAME_KNOWLEDGE)} characters (~{len(GAME_KNOWLEDGE.split())} words)")

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print("Bot is ready!")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if bot.user.mentioned_in(message) or message.channel.name == "bot-chat":  # or use your channel
        # Get user question
        question = message.content.replace(f"<@{bot.user.id}>", "").strip()

        # Build prompt with knowledge
        prompt = f"""You are an expert game master for [Game Name]. 
Use ONLY this knowledge base — do NOT make up facts:

{GAME_KNOWLEDGE}

Answer the player's question helpfully, in character, short but complete.
Player asks: {question}

Your answer:"""

        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",   # or "llama-3.1-8b-instant" if rate limit issues
                temperature=0.7,
                max_tokens=800,          # limit reply length
                stream=False
            )

            reply = chat_completion.choices[0].message.content
            await message.reply(reply[:2000])  # Discord message limit

        except Exception as e:
            await message.reply(f"Oops... error: {str(e)[:100]}")

    await bot.process_commands(message)

# Optional simple command
@bot.command()
async def ping(ctx):
    await ctx.send("Pong!")

bot.run(DISCORD_TOKEN)

app = Flask(__name__)

@app.route('/')
def home():
    return f"ISO Intelligent Bot is alive! Knowledge size: {len(GAME_KNOWLEDGE)} chars"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# Start Flask in background thread
threading.Thread(target=run_flask, daemon=True).start()