"""
WinGo ProSignal — Telegram Userbot
====================================
Yeh script Telethon library use karke ek Telegram channel ko silently
listen karti hai. Jab bhi koi naya prediction message aata hai, yeh
usse parse karke Supabase ki `expert_predictions` table mein save kar
deti hai.

Setup:
    1. Telegram se API ID aur API Hash lena:
       → https://my.telegram.org → 'API development tools'
    2. .env file mein yeh environment variables set karo:
       TELEGRAM_API_ID=your_api_id
       TELEGRAM_API_HASH=your_api_hash
       TELEGRAM_CHANNEL=kingcolormaster
       SUPABASE_URL=...
       SUPABASE_KEY=...
    3. Pehli baar run karne par phone number aur OTP enter karna hoga.
       Session file ban jayegi taaki dobara login nahi karna pade.

Usage:
    python telegram_bot.py
"""

import os
import re
import asyncio
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client, Client
from telethon import TelegramClient, events

# ── Load environment variables ────────────────────────────────────
load_dotenv()

TELEGRAM_API_ID   = int(os.environ.get("TELEGRAM_API_ID", "0"))
TELEGRAM_API_HASH = os.environ.get("TELEGRAM_API_HASH", "")
TELEGRAM_CHANNEL  = os.environ.get("TELEGRAM_CHANNEL", "kingcolormaster")
SUPABASE_URL      = os.environ.get("SUPABASE_URL", "https://npbpjsdxisdutcruwkgr.supabase.co")
SUPABASE_KEY      = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5wYnBqc2R4aXNkdXRjcnV3a2dyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzk0MzMxOTEsImV4cCI6MjA5NTAwOTE5MX0.Td38AsexT9B7C6LSuFBml3QVFaaMn-m-rcJgXtQ_uIU")

# Session file naam — login state yahin save hota hai
SESSION_FILE = "prosignal_session"

# ── Logging setup ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ProSignal")

# ── Supabase client ───────────────────────────────────────────────
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Telegram client ───────────────────────────────────────────────
client = TelegramClient(SESSION_FILE, TELEGRAM_API_ID, TELEGRAM_API_HASH)


# ── Message Parser ────────────────────────────────────────────────
def parse_wingo_message(text: str) -> dict | None:
    """
    Channel message format:
        ✨ WINGO 1 MIN ✨
        📍 Period: 20260523100011088
        🔮 Prediction: SMALL
        🎲 Number: 1
        🎨 Color: 🟢
        🏁 Level: 1
    Returns parsed dict or None if message doesn't match.
    """
    # Only process WinGo 1 Min messages
    if "WINGO 1 MIN" not in text.upper():
        return None

    # Extract fields using regex (flexible — handles emojis)
    period     = re.search(r"Period[:\s]+(\d+)",     text, re.IGNORECASE)
    prediction = re.search(r"Prediction[:\s]+(\w+)", text, re.IGNORECASE)
    number     = re.search(r"Number[:\s]+(\d+)",     text, re.IGNORECASE)
    color_line = re.search(r"Color[:\s]+(.+)",       text, re.IGNORECASE)
    level      = re.search(r"Level[:\s]+(\d+)",      text, re.IGNORECASE)

    if not period or not prediction:
        return None  # Required fields not found

    # Normalize prediction to BIG / SMALL
    raw_pred = prediction.group(1).upper().strip()
    if raw_pred in ("BIG", "LARGE"):
        normalized_pred = "BIG"
    elif raw_pred in ("SMALL", "LITTLE"):
        normalized_pred = "SMALL"
    else:
        normalized_pred = raw_pred  # Keep as-is (e.g., RED, GREEN)

    # Detect colour from emoji or text
    raw_colour = color_line.group(1).strip() if color_line else ""
    if "🟢" in raw_colour or "GREEN" in raw_colour.upper():
        colour = "Green"
    elif "🔴" in raw_colour or "RED" in raw_colour.upper():
        colour = "Red"
    elif "🟣" in raw_colour or "VIOLET" in raw_colour.upper():
        colour = "Violet"
    else:
        colour = raw_colour  # Keep raw if unknown

    return {
        "period":     period.group(1).strip(),
        "game_code":  "WinGo_1Min",           # Channel only does 1Min
        "prediction": normalized_pred,
        "number":     int(number.group(1)) if number else None,
        "colour":     colour,
        "level":      int(level.group(1)) if level else 1,
    }


# ── Supabase Save ─────────────────────────────────────────────────
def save_to_supabase(data: dict) -> bool:
    """
    Save parsed prediction to expert_predictions table.
    Returns True on success, False on failure.
    """
    try:
        # Check if period already exists (avoid duplicates)
        existing = supabase.table("expert_predictions")\
            .select("id")\
            .eq("period", data["period"])\
            .eq("game_code", data["game_code"])\
            .execute()

        if existing.data:
            log.info(f"[SKIP] Period {data['period']} already saved.")
            return False

        # Insert new prediction
        supabase.table("expert_predictions").insert({
            "period":     data["period"],
            "game_code":  data["game_code"],
            "prediction": data["prediction"],
            "number":     data.get("number"),
            "colour":     data.get("colour"),
            "level":      data.get("level", 1),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()

        log.info(
            f"[SAVED] Period={data['period']} | "
            f"Pred={data['prediction']} | "
            f"Num={data.get('number')} | "
            f"Colour={data.get('colour')}"
        )
        return True

    except Exception as e:
        log.error(f"[DB ERROR] {e}")
        return False


# ── Telegram Event Handler ─────────────────────────────────────────
@client.on(events.NewMessage(chats=TELEGRAM_CHANNEL))
async def on_new_message(event):
    """Triggered every time a new message arrives in the channel."""
    text = event.message.text or ""
    if not text.strip():
        return

    log.info(f"[NEW MSG] {text[:80].replace(chr(10), ' ')}...")

    parsed = parse_wingo_message(text)
    if parsed:
        save_to_supabase(parsed)
    else:
        log.debug("[SKIP] Message is not a WinGo 1Min prediction.")


# ── Main ──────────────────────────────────────────────────────────
async def main():
    log.info("=" * 60)
    log.info("🔮 ProSignal Userbot Starting...")
    log.info(f"   Channel  : @{TELEGRAM_CHANNEL}")
    log.info(f"   Supabase : {SUPABASE_URL[:40]}...")
    log.info("=" * 60)

    await client.start()
    log.info("✅ Logged in to Telegram successfully.")
    log.info("👂 Listening for new messages... (Press Ctrl+C to stop)")

    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
