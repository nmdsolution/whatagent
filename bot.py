"""
Bot WhatsApp IA : WasenderAPI (webhook) + Gemini (Google AI Studio).

Flux : un client écrit sur WhatsApp -> WasenderAPI appelle /webhook
       -> on demande une réponse à l'IA -> on renvoie la réponse via /api/send-message.

Lancer :  uvicorn bot:app --host 0.0.0.0 --port 8000
"""
import os
import re
import hmac
import asyncio
import logging
import unicodedata
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
AI_RETRY_DELAY = 20                                        # secondes avant la 2e tentative IA
SEND_ATTEMPTS = 5                                          # essais d'envoi quand Wasender limite le débit
# Numéro WhatsApp du conseiller prévenu à chaque demande (format international, sans +). Vide = aucune alerte.
ADVISOR_NUMBER = "".join(c for c in os.getenv("ADVISOR_NUMBER", "") if c.isdigit())

# ---- Message d'accueil de la hotline (envoyé tel quel, sans passer par l'IA) ----
WELCOME = """👋 Bienvenue sur la Hotline *KASI MEDICAL SOLUTIONS (KMS)*

Pour un traitement rapide de votre demande, merci de nous communiquer :

👤 Nom et prénom :
🏥 Structure / Établissement :
📍 Ville :

📋 Objet de votre demande :

1️⃣ 🚨 Urgence technique
2️⃣ 🛠️ Assistance technique / SAV
3️⃣ 🛒 Produits / Prix / Devis
4️⃣ 📦 Commandes / Livraisons
5️⃣ 🔧 Maintenance / Installation
6️⃣ 💻 Informatique médicale
7️⃣ 💰 Facturation / Paiements
8️⃣ 🤝 Autres demandes

✍️ Décrivez brièvement votre besoin en précisant le numéro correspondant.

✅ Notre équipe vous répondra dans les meilleurs délais.

🇬🇧 English welcome.

*KMS – Votre partenaire en solutions médicales.*"""

SERVICES = {
    "1": "🚨 Urgence technique",
    "2": "🛠️ Assistance technique / SAV",
    "3": "🛒 Produits / Prix / Devis",
    "4": "📦 Commandes / Livraisons",
    "5": "🔧 Maintenance / Installation",
    "6": "💻 Informatique médicale",
    "7": "💰 Facturation / Paiements",
    "8": "🤝 Autres demandes",
}

# ---- Personnalité et connaissances du bot (suite de la conversation, gérée par l'IA) ----
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """
Tu es l'assistant de la Hotline WhatsApp de KASI MEDICAL SOLUTIONS SARL (KMS),
fournisseur de matériel médical basé à Yaoundé, Cameroun. Tes clients sont des
hôpitaux, cliniques, cabinets, centres d'imagerie, laboratoires et professionnels
de santé. Ton rôle : recueillir la demande du client en un minimum de messages,
puis la transmettre à l'équipe KMS.

CE QUE FAIT KMS (équipements, solutions, maintenance, service après-vente)
- Imagerie médicale : radiologie, échographie (appareils et sondes), scanner,
  mammographie, avec accessoires et consommables (films, cassettes, imprimantes
  médicales, produits de contraste, kits pour injecteurs).
- Laboratoire : équipements, dispositifs et réactifs.
- Médecine générale, cardiologie (ECG, papiers et électrodes), ORL, ophtalmologie,
  chirurgie : équipements, dispositifs et consommables.
- Tensiomètres, produits désinfectants, sécurité électrique des équipements.
- Équipements neufs et de seconde main.
- Assistance technique, maintenance, installation, formation, informatique médicale.
- Marques distribuées, entre autres : Philips, GE, Fuji, Agfa, Carestream, Sony,
  Canon, Antmed.

INFORMATIONS PRATIQUES
- Stocks à Yaoundé, Douala et Bafoussam ; livraison standard ou express dans les
  autres villes.
- Paiement : Orange Money, virement bancaire ou chèque.
- Conseillers : 6 90 36 83 77 / 6 81 09 72 07,
  salesteam@kasimedicalsolutions.com, www.kasimedicalsolutions.com.
- Adresse postale : BP 7468 Yaoundé.

LE MESSAGE D'ACCUEIL QUE LE CLIENT A REÇU
""" + WELCOME + """

LA DEMANDE À RECUEILLIR (5 éléments)
👤 Nom et prénom
🏥 Structure / Établissement
📍 Ville
📋 Objet (un des 8 ci-dessus)
✍️ Besoin (ce que le client veut, en quelques mots)

DÉROULEMENT
- Le client répond en général à l'accueil en un seul message, souvent en vrac et
  sans libellés. Retrouve toi-même les 5 éléments dans ce qu'il écrit. S'il ne
  donne pas de numéro, déduis l'objet de son besoin. Une panne qui met un
  équipement à l'arrêt ou bloque l'activité relève de l'objet 1 (urgence).
- Si les éléments essentiels sont là (le besoin, plus la structure ou le nom),
  envoie tout de suite le récapitulatif et transmets : ne pose pas de question
  de plus. Les précisions utiles (marque, modèle, quantité, numéro de facture ou
  de commande) seront demandées par le conseiller.
- Sinon, demande en UN seul message court ce qui manque, en reprenant les lignes
  avec leur emoji. Si le client n'a envoyé qu'un numéro, demande-lui son nom, sa
  structure, sa ville et une courte description du besoin.
- Ne redemande jamais une information déjà donnée. Ne relance qu'une fois : si le
  client ne complète pas, transmets avec ce que tu as.
- Le récapitulatif est court : une ligne par élément, avec l'objet écrit comme
  dans le menu, puis une phrase annonçant que l'équipe KMS le recontacte dans les
  meilleurs délais (pour l'objet 1 : que la demande est signalée comme urgente).
- Après la transmission, réponds simplement aux messages suivants. Si le client
  ajoute une information importante ou une nouvelle demande, renvoie un
  récapitulatif mis à jour et transmets à nouveau.
- Un message entre crochets comme [Photo reçue] signale un fichier envoyé par le
  client : tu ne peux pas l'ouvrir. Pour une photo, une vidéo ou un document,
  remercie et dis qu'il est joint à la demande pour l'équipe KMS. Un message vocal
  n'est pas joint : explique que tu ne peux pas l'écouter et demande poliment
  d'écrire la demande. N'écris jamais toi-même de texte entre crochets.

TRANSMISSION AU CONSEILLER KMS
- Chaque fois que tu envoies un récapitulatif, termine ton message par une
  dernière ligne, seule, exactement de cette forme : #TRANSMETTRE 3 NORMAL
  (le numéro de l'objet de 1 à 8, puis URGENT pour l'objet 1, sinon NORMAL).
  Cette ligne n'est pas montrée au client : elle envoie ton message au conseiller
  KMS. Sans elle, personne n'est prévenu.
- N'ajoute jamais cette ligne à un message qui pose une question au client.

LIBELLÉS EN ANGLAIS (à utiliser si le client écrit en anglais)
👤 Full name:
🏥 Facility / Company:
📍 City:
📋 Subject: 1 Technical emergency, 2 Technical support / After-sales,
3 Products / Prices / Quotes, 4 Orders / Deliveries, 5 Maintenance / Installation,
6 Medical IT, 7 Invoicing / Payments, 8 Other requests
✍️ Your need:

RÈGLES
- Réponds dans la langue du client (français ou anglais), sur un ton chaleureux et
  professionnel. Si le client écrit en anglais, tout ton message est en anglais.
  Messages courts, comme sur WhatsApp ; quelques emojis, sans excès.
  Pour le gras, utilise *une seule étoile* de chaque côté ; pas de titres, pas de
  tableaux, pas de liens formatés.
- Ne donne jamais de prix, de remise, de délai ni de disponibilité en stock : seul
  un conseiller KMS les confirme, par proforma.
- N'affirme pas que KMS vend un modèle précis si tu n'en es pas sûr : dis que
  l'équipe vérifie.
- Ne donne aucun numéro de ticket ou de dossier, et ne promets pas d'heure de rappel.
- Tu ne donnes aucun conseil médical (diagnostic, traitement, posologie) :
  oriente vers un professionnel de santé. Pour une panne, ne propose aucune
  manipulation technique de l'appareil : l'équipe technique s'en charge.
- Si tu ne connais pas une information, dis que tu transmets la question à un
  conseiller KMS. N'invente rien.
- Ne te présente pas à nouveau et ne renvoie pas le menu toi-même.
""").strip()

# Envoyé si l'IA ne répond pas : le client ne reste jamais sans réponse.
FALLBACK = """Merci pour votre message, il est bien reçu. 🙏

Notre assistant est momentanément très sollicité. Un conseiller KMS vous répondra dès que possible.
Pour une urgence, appelez le 6 90 36 83 77 ou le 6 81 09 72 07."""

GREETING = re.compile(
    r"(bonjour|bonsoir|bjr|bsr|salut|slt|coucou|cc|allo|hello|hi|hey|good (morning|afternoon|evening)|menu)"
    r"( (kms|monsieur|madame|docteur|svp|s il vous plait|please))?"
)


def _plain(text: str) -> str:
    """Minuscules, sans accents ni ponctuation, pour reconnaître salutations et numéros."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c if c.isalnum() else " " for c in text if unicodedata.category(c) != "Mn")
    return " ".join(text.split())


def fixed_reply(conv: deque, text: str) -> str | None:
    """Message d'accueil fixe quand il s'applique (premier bonjour, ou « menu »), sinon None."""
    plain = _plain(text)
    if plain == "menu" or (not conv and (not plain or GREETING.fullmatch(plain))):
        return WELCOME
    return None


class _NoHealthcheck(logging.Filter):
    """Retire des journaux le GET / du healthcheck Docker (toutes les 10 s)."""
    def filter(self, record: logging.LogRecord) -> bool:
        return '"GET / ' not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_NoHealthcheck())

app = FastAPI()
ai = AsyncOpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL, timeout=60)
history: dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
seen_ids: deque = deque(maxlen=500)  # évite de répondre deux fois au même message
send_lock = asyncio.Lock()
sender_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)  # messages d'un client traités dans l'ordre


async def send_whatsapp(to: str, text: str) -> bool:
    """Envoie un message ; en cas de limite de débit Wasender (429), attend puis réessaie. True si envoyé."""
    async with send_lock:                              # un envoi à la fois : respecte la limite du compte
        for attempt in range(1, SEND_ATTEMPTS + 1):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post(
                        "https://www.wasenderapi.com/api/send-message",
                        headers={"Authorization": f"Bearer {WASENDER_API_KEY}"},
                        json={"to": to, "text": text},
                    )
            except httpx.HTTPError as e:
                print(f"[ERREUR envoi] {type(e).__name__} {e}")
                return False
            if r.status_code < 400:
                return True
            if r.status_code != 429 or attempt == SEND_ATTEMPTS:
                print(f"[ERREUR envoi] abandon vers ...{to[-4:]} : {r.status_code} {r.text[:200]}")
                return False
            try:
                wait = int(r.json().get("retry_after", 60))
            except ValueError:
                wait = 60
            print(f"[ENVOI] limite Wasender atteinte, nouvel essai dans {wait + 1} s (tentative {attempt}/{SEND_ATTEMPTS})")
            await asyncio.sleep(min(wait, 120) + 1)
    return False


TRANSMIT = re.compile(r"\n?[ \t]*#TRANSMETTRE\b[ \t]*(\d)?[ \t]*(URGENT|NORMAL)?[ \t]*", re.I)


def split_transmit(reply: str) -> tuple[str, str | None, bool]:
    """Sépare le texte destiné au client de la ligne #TRANSMETTRE : (texte, service ou None, urgent)."""
    m = TRANSMIT.search(reply)
    if not m:
        return reply, None, False
    return TRANSMIT.sub("", reply).strip(), m.group(1) or "8", (m.group(2) or "").upper() == "URGENT"


async def notify_advisor(sender: str, title: str, body: str) -> None:
    if not ADVISOR_NUMBER:
        return
    ok = await send_whatsapp(ADVISOR_NUMBER, f"{title}\n📱 Client : +{sender}\n👉 https://wa.me/{sender}\n\n{body}")
    print(f"[ALERTE] conseiller {'prévenu' if ok else 'NON PRÉVENU (envoi échoué)'} pour ...{sender[-4:]}")


EN_WORDS = set("the i you we is are do does have need want for and of to please thanks thank hello what how can my our your with from".split())
FR_WORDS = set("le la les je vous nous est sont avez besoin veux pour et de du des svp merci bonjour bonsoir quoi comment mon notre votre avec un une".split())


def language_note(conv: deque) -> str:
    """Consigne de langue d'après le dernier vrai message du client (les marqueurs de fichiers ne comptent pas)."""
    for m in reversed(conv):
        if m["role"] != "user":
            continue
        words = _plain(m["content"].split("]", 1)[-1]).split()
        en, fr = sum(w in EN_WORDS for w in words), sum(w in FR_WORDS for w in words)
        if en > fr:
            return "\n\nThe customer writes in English: write your whole reply in English, with the English labels."
        if fr > en:
            return ""
    return ""


async def ai_reply(conv: deque) -> str:
    """Réponse de l'IA, ou chaîne vide si elle reste indisponible après une seconde tentative."""
    for attempt in (1, 2):
        try:
            resp = await ai.chat.completions.create(
                model=MODEL,
                max_tokens=2000,                       # marge si le modèle raisonne avant de répondre
                messages=[{"role": "system", "content": SYSTEM_PROMPT + language_note(conv)}, *conv],
            )
            reply = (resp.choices[0].message.content or "").strip()
            if reply:
                return reply
        except Exception as e:
            print(f"[ERREUR IA] tentative {attempt} : {str(e)[:300]}")
        if attempt == 1:
            await asyncio.sleep(AI_RETRY_DELAY)        # quota par minute dépassé : laisser passer
    return ""


async def handle_message(sender: str, text: str) -> None:
    async with sender_locks[sender]:
        await _handle_message(sender, text)


async def _handle_message(sender: str, text: str) -> None:
    conv = history[sender]
    reply = fixed_reply(conv, text)
    conv.append({"role": "user", "content": text})
    if reply is None:
        reply = await ai_reply(conv)
    if not reply:
        conv.append({"role": "assistant", "content": FALLBACK})
        await send_whatsapp(sender, FALLBACK)
        await notify_advisor(sender, "⚠️ *Hotline KMS : client à reprendre*",
                             f"L'assistant n'a pas pu répondre. Dernier message du client :\n{text[:600]}")
        return
    conv.append({"role": "assistant", "content": reply})  # avec la ligne #TRANSMETTRE : l'IA sait que c'est fait
    client_text, service, urgent = split_transmit(reply)
    await send_whatsapp(sender, client_text)
    if service:
        title = ("🚨 *URGENT : " if urgent else "🔔 *") + "Nouvelle demande Hotline KMS*"
        await notify_advisor(sender, title, f"🗂️ Service : {SERVICES.get(service, SERVICES['8'])}\n\n{client_text}")


MEDIA_NOTES = {
    "imageMessage": "[Photo reçue]",
    "videoMessage": "[Vidéo reçue]",
    "documentMessage": "[Document reçu]",
    "documentWithCaptionMessage": "[Document reçu]",
    "audioMessage": "[Message vocal reçu]",
}


def media_note(msg: dict) -> str:
    """Marqueur texte pour un fichier envoyé par le client (le bot ne lit pas les fichiers)."""
    content = msg.get("message")
    if not isinstance(content, dict):
        return ""
    return next((note for kind, note in MEDIA_NOTES.items() if kind in content), "")


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
    text = " ".join(filter(None, [media_note(msg), (msg.get("messageBody") or "").strip()]))
    if not sender or not text:                        # autres types (stickers, réactions...) : ignorés
        return {"ok": True}

    print(f"[WEBHOOK] message reçu de ...{sender[-4:]}")  # numéro masqué, texte non journalisé
    # Répondre 200 tout de suite, traiter en arrière-plan
    background.add_task(handle_message, sender, text)
    return {"ok": True}


@app.get("/")
async def health():
    return {"status": "bot en ligne"}
