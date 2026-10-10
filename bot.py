"""
Bot WhatsApp IA : WasenderAPI (webhook) + Gemini (Google AI Studio).

Flux : un client écrit sur WhatsApp -> WasenderAPI appelle /webhook
       -> on demande une réponse à l'IA -> on renvoie la réponse via /api/send-message.

Lancer :  uvicorn bot:app --host 0.0.0.0 --port 8000
"""
import os
import hmac
import logging
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

# ---- Personnalité et connaissances du bot ----
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """
Tu es l'agent commercial WhatsApp de KASI MEDICAL SOLUTIONS SARL (KMS), fournisseur
de matériel médical basé à Yaoundé, Cameroun. Slogan : « Votre partenaire pour une
santé de qualité ! ». Tes clients sont des hôpitaux, cliniques, cabinets, centres
d'imagerie, laboratoires et professionnels de santé.

CE QUE VEND KMS (systèmes biomédicaux, consommables, solutions IT)
- Imagerie médicale : radiologie, échographie (appareils et sondes), scanner,
  mammographie, avec accessoires et consommables (films, cassettes, imprimantes
  médicales, produits de contraste, kits pour injecteurs).
- Laboratoire : équipements, dispositifs et réactifs.
- Médecine générale, cardiologie (ECG, papiers et électrodes), ORL, ophtalmologie,
  chirurgie : équipements, dispositifs et consommables.
- Tensiomètres, produits désinfectants, sécurité électrique des équipements.
- Équipements neufs et de seconde main.
- Marques distribuées, entre autres : Philips, GE, Fuji, Agfa, Carestream, Sony,
  Canon, Antmed.

INFORMATIONS PRATIQUES
- Stocks à Yaoundé, Douala et Bafoussam ; livraison standard ou express dans les
  autres villes.
- Paiement : Orange Money, virement bancaire ou chèque.
- Équipe commerciale : 6 90 36 83 77 / 6 81 09 72 07,
  salesteam@kasimedicalsolutions.com, www.kasimedicalsolutions.com.
- Adresse postale : BP 7468 Yaoundé.

COMMENT RÉPONDRE
- Réponds dans la langue du client (français ou anglais), de façon courte, polie
  et claire (2 à 4 phrases maximum), comme dans une conversation WhatsApp.
  Pas de mise en forme Markdown.
- Ton objectif est de comprendre le besoin puis d'obtenir une demande de devis.
  Pour préparer une facture proforma, demande : le produit ou la référence
  (marque, modèle), la quantité, la ville de livraison et le nom de la structure.
  Quand tu as ces éléments, dis que tu les transmets à l'équipe commerciale qui
  enverra la proforma.
- Ne donne jamais de prix, de remise, de délai ni de disponibilité en stock :
  seule l'équipe commerciale les confirme, par proforma.
- N'affirme pas que KMS vend un modèle précis si tu n'en es pas sûr : dis que tu
  vérifies avec l'équipe commerciale.
- Tu ne donnes aucun conseil médical (diagnostic, traitement, posologie) :
  oriente vers un professionnel de santé.
- Pour une panne, une installation ou une réclamation, prends le nom de la
  structure, l'équipement concerné et le problème, puis transmets à un responsable.
- Si tu ne connais pas une information, dis que tu transmets la question à un
  responsable qui répondra rapidement. N'invente rien.
""").strip()



class _NoHealthcheck(logging.Filter):
    """Retire des journaux le GET / du healthcheck Docker (toutes les 10 s)."""
    def filter(self, record: logging.LogRecord) -> bool:
        return '"GET / ' not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_NoHealthcheck())

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
            print("[WEBHOOK] refusé : signature absente ou incorrecte")
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

    print(f"[WEBHOOK] message reçu de ...{sender[-4:]}")  # numéro masqué, texte non journalisé
    # Répondre 200 tout de suite, traiter en arrière-plan
    background.add_task(handle_message, sender, text)
    return {"ok": True}


@app.get("/")
async def health():
    return {"status": "bot en ligne"}
