"""
Bot WhatsApp IA : WasenderAPI (webhook) + Gemini (Google AI Studio).

Flux : un client écrit sur WhatsApp -> WasenderAPI appelle /webhook
       -> on demande une réponse à l'IA -> on renvoie la réponse via /api/send-message.

Lancer :  uvicorn bot:app --host 0.0.0.0 --port 8000
"""
import os
import hmac
from collections import defaultdict, deque

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request, Response
from openai import AsyncOpenAI

load_dotenv()

WASENDER_API_KEY = os.environ["WASENDER_API_KEY"]          # clé de la session (onglet Credentials)
WASENDER_WEBHOOK_SECRET = os.getenv("WASENDER_WEBHOOK_SECRET", "")  # optionnel mais conseillé
AI_API_KEY = os.environ["AI_API_KEY"]                      # clé Google AI Studio (aistudio.google.com)
# Endpoint compatible OpenAI : on peut changer de fournisseur (ex. NVIDIA) sans toucher au code
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = os.getenv("AI_MODEL", "gemini-3.5-flash-lite")     # modèle rapide, palier gratuit
MAX_HISTORY = 20                                           # messages gardés par contact

# ---- Personnalité du bot : à adapter à ton activité ----
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """
Tu es l'assistant WhatsApp de [NOM DE L'ENTREPRISE] à Yaoundé, Cameroun.
Réponds en français, de façon courte, polie et claire (2 à 4 phrases maximum),
comme dans une conversation WhatsApp. Pas de mise en forme Markdown.
Si tu ne connais pas une information (prix, disponibilité, rendez-vous),
dis que tu transmets la question à un responsable qui répondra rapidement.
N'invente jamais de prix ni de promesses.
""").strip()

app = FastAPI()
ai = AsyncOpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL, timeout=60)
history: dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
seen_ids: deque = deque(maxlen=500)  # évite de répondre deux fois au même message


async def send_whatsapp(to: str, text: str) -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://www.wasenderapi.com/api/send-message",
            headers={"Authorization": f"Bearer {WASENDER_API_KEY}"},
            json={"to": to, "text": text},
        )
        if r.status_code >= 400:
            print(f"[ERREUR envoi] {r.status_code} {r.text}")


async def handle_message(sender: str, text: str) -> None:
    conv = history[sender]
    conv.append({"role": "user", "content": text})
    try:
        resp = await ai.chat.completions.create(
            model=MODEL,
            max_tokens=2000,                           # marge si le modèle raisonne avant de répondre
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *conv],
        )
        reply = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[ERREUR IA] {e}")
        conv.pop()
        return
    if not reply:                                      # réponse vide : ne rien envoyer
        conv.pop()
        return
    conv.append({"role": "assistant", "content": reply})
    await send_whatsapp(sender, reply)


@app.post("/webhook")
async def webhook(request: Request, background: BackgroundTasks):
    # Vérification du secret du webhook (si configuré)
    if WASENDER_WEBHOOK_SECRET:
        sig = request.headers.get("x-webhook-signature", "")
        if not hmac.compare_digest(sig, WASENDER_WEBHOOK_SECRET):
            return Response(status_code=401)

    payload = await request.json()
    if payload.get("event") not in (
        "messages.received", "messages-personal.received", "messages.personal.received", "messages.upsert",
    ):
        return {"ok": True}

    msg = (payload.get("data") or {}).get("messages") or {}
    key = msg.get("key") or {}

    if key.get("fromMe"):                              # ignorer nos propres messages
        return {"ok": True}
    if str(key.get("remoteJid", "")).endswith("@g.us"):  # ignorer les groupes
        return {"ok": True}

    msg_id = key.get("id")
    if msg_id in seen_ids:
        return {"ok": True}
    seen_ids.append(msg_id)

    sender = key.get("cleanedSenderPn")
    text = (msg.get("messageBody") or "").strip()
    if not sender or not text:                        # médias sans texte : ignorés pour l'instant
        return {"ok": True}

    # Répondre 200 tout de suite, traiter en arrière-plan
    background.add_task(handle_message, sender, text)
    return {"ok": True}


@app.get("/")
async def health():
    return {"status": "bot en ligne"}
