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

# ---- Messages fixes de la hotline (envoyés tels quels, sans passer par l'IA) ----
WELCOME = """Bonjour 👋

Bienvenue sur la Hotline *KASI MEDICAL SOLUTIONS (KMS)*.
Merci de nous avoir contactés. Pour traiter rapidement votre demande, choisissez le service souhaité :

1️⃣ 🛠️ Assistance technique / Panne / SAV
2️⃣ 🛒 Demande commerciale / Équipement / Consommables
3️⃣ 📋 Maintenance / Installation / Formation
4️⃣ 💰 Demande de devis / Proforma / Prix
5️⃣ 📦 Commande / Livraison / Disponibilité produit
6️⃣ 💻 Informatique médicale / Logiciel / DICOM
7️⃣ 📞 Être mis en relation avec un conseiller KMS
8️⃣ Autre demande

👉 Répondez avec le numéro de votre choix, ou écrivez directement votre besoin.
🇬🇧 You can also write to us in English.

*KASI MEDICAL SOLUTIONS*
Équipements • Solutions • Maintenance • Service après-vente"""

_IDENTITE = """🏥 Établissement / Société :
📍 Ville :
👤 Nom et fonction :"""

FORMS = {
    "1": f"""Nous sommes désolés pour ce désagrément. 🛠️ Notre équipe technique va vous aider.

Merci de nous indiquer :

{_IDENTITE}
⚙️ Équipement concerné (marque, modèle, n° de série si possible) :
⚠️ Panne constatée ou message d'erreur :
📅 Depuis quand :
🚨 L'équipement est-il totalement à l'arrêt ? (oui / non)

Une photo ou une courte vidéo de la panne nous aide à diagnostiquer plus vite.

Un technicien KMS prendra votre demande en charge.""",
    "2": f"""Merci pour votre intérêt pour les solutions KMS. 🤝

Afin de mieux répondre à votre besoin, merci de nous indiquer :

{_IDENTITE}
🔎 Équipement ou produit recherché :
🔢 Quantité souhaitée :
📅 Délai souhaité :

Vous pouvez aussi nous envoyer une photo, une référence ou une fiche technique du produit recherché.

Un conseiller KMS prendra votre demande en charge.""",
    "3": f"""Merci de faire confiance à KMS pour vos équipements. 📋

Merci de nous indiquer :

{_IDENTITE}
🧰 Prestation souhaitée (maintenance préventive, réparation, installation, formation) :
⚙️ Équipement(s) concerné(s) (marque, modèle, nombre) :
📅 Période souhaitée :

Notre service technique vous proposera une date d'intervention.""",
    "4": f"""Avec plaisir, nous préparons votre proforma. 💰

Merci de nous indiquer :

{_IDENTITE}
🔎 Produit(s) ou référence(s) :
🔢 Quantité pour chaque produit :
🚚 Ville de livraison :
📅 Délai souhaité :

Une photo, une référence ou une ancienne facture nous permet d'aller plus vite.

Un conseiller KMS vous enverra la facture proforma.""",
    "5": f"""Nous vérifions cela pour vous. 📦

Merci de nous indiquer :

{_IDENTITE}
🧾 Numéro de proforma, de commande ou de facture (si vous en avez un) :
🔎 Produit concerné et quantité :
🚚 Ville de livraison :

Un conseiller KMS vous confirmera la disponibilité ou l'état de votre livraison.""",
    "6": f"""Notre équipe informatique médicale est à votre écoute. 💻

Merci de nous indiquer :

{_IDENTITE}
🧩 Votre besoin (PACS, RIS, DICOM, logiciel, réseau, impression de films...) :
⚙️ Équipements concernés (marque, modèle) :
📝 Problème rencontré ou projet envisagé :

Une capture d'écran du message d'erreur est la bienvenue.

Un spécialiste KMS prendra votre demande en charge.""",
    "7": f"""Un conseiller KMS va vous recontacter. 📞

Pour vous orienter vers la bonne personne, merci de nous indiquer :

{_IDENTITE}
📝 Objet de votre demande :
🕘 Meilleur moment pour vous joindre :

Vous pouvez aussi appeler directement le 6 90 36 83 77 ou le 6 81 09 72 07.""",
    "8": """Bien sûr. ✍️ Décrivez-nous votre demande en quelques mots, en précisant votre établissement et votre ville.

Notre équipe vous orientera vers le bon interlocuteur.""",
}

# ---- Personnalité et connaissances du bot (suite de la conversation, gérée par l'IA) ----
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """
Tu es l'assistant de la Hotline WhatsApp de KASI MEDICAL SOLUTIONS SARL (KMS),
fournisseur de matériel médical basé à Yaoundé, Cameroun. Slogan : « Votre partenaire
pour une santé de qualité ! ». Tes clients sont des hôpitaux, cliniques, cabinets,
centres d'imagerie, laboratoires et professionnels de santé.

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

LES 8 SERVICES DE LA HOTLINE ET LES INFORMATIONS À RECUEILLIR
""" + "\n\n".join(f"--- Service {n} ---\n{form}" for n, form in FORMS.items()) + """

LIBELLÉS DES FICHES EN ANGLAIS (à utiliser tels quels si le client écrit en anglais)
🏥 Facility / Company:
📍 City:
👤 Name and position:
🔎 Equipment or product needed:
🔢 Quantity:
📅 Desired timeframe:
🚚 Delivery city:
⚙️ Equipment concerned (brand, model, serial number if possible):
⚠️ Fault observed or error message:
🚨 Is the equipment completely down? (yes / no)
🧰 Service needed (preventive maintenance, repair, installation, training):
🧾 Proforma, order or invoice number (if any):
🧩 Your need (PACS, RIS, DICOM, software, network, film printing...):
📝 Subject of your request:
🕘 Best time to reach you:

DÉROULEMENT
- Le client reçoit d'abord le menu des 8 services. S'il répond par un numéro, il
  reçoit la fiche du service. S'il écrit directement son besoin, choisis toi-même
  le service qui correspond, sans lui demander de numéro.
- Ne redemande jamais une information que le client a déjà donnée. Reprends ce
  qu'il a écrit et demande seulement ce qui manque, en reprenant les lignes de la
  fiche (avec leur emoji), une par ligne. S'il manque une ou deux informations,
  pose simplement la question en une phrase.
- Si le client ne veut pas ou ne peut pas donner une information, n'insiste pas.
- Quand la fiche est complète, envoie un récapitulatif court (une ligne par
  information), puis annonce que l'équipe KMS concernée prend la demande en charge
  et le recontacte. Pour une panne avec équipement à l'arrêt, précise que la
  demande est signalée comme urgente.
- Si le client change de sujet, passe au service correspondant. S'il écrit
  « menu », le menu lui est renvoyé automatiquement.
- Un message entre crochets comme [Photo reçue] signale un fichier envoyé par le
  client : tu ne peux pas l'ouvrir. Pour une photo, une vidéo ou un document,
  remercie et dis qu'il est joint à la demande pour l'équipe KMS. Un message vocal
  n'est pas joint : explique que tu ne peux pas l'écouter et demande poliment
  d'écrire la demande. N'écris jamais toi-même de texte entre crochets.

RÈGLES
- Réponds dans la langue du client (français ou anglais), sur un ton chaleureux et
  professionnel. Si le client écrit en anglais, tout ton message est en anglais, y
  compris les lignes des fiches, et tu restes en anglais pour la suite, même après
  un message entre crochets. Messages courts, comme sur WhatsApp ; quelques emojis,
  sans excès.
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
- Ne cite pas le slogan et ne te présente pas à nouveau à chaque message.
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
    """Réponse fixe (menu ou fiche d'un service) quand elle s'applique, sinon None."""
    plain = _plain(text)
    if plain == "menu" or (not conv and (not plain or GREETING.fullmatch(plain))):
        return WELCOME
    last = conv[-1]["content"] if conv else ""
    if last == WELCOME and plain in FORMS:
        return FORMS[plain]
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


async def send_whatsapp(to: str, text: str) -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://www.wasenderapi.com/api/send-message",
            headers={"Authorization": f"Bearer {WASENDER_API_KEY}"},
            json={"to": to, "text": text},
        )
        if r.status_code >= 400:
            print(f"[ERREUR envoi] {r.status_code} {r.text}")


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
    conv = history[sender]
    reply = fixed_reply(conv, text)
    conv.append({"role": "user", "content": text})
    if reply is None:
        reply = await ai_reply(conv) or FALLBACK
    conv.append({"role": "assistant", "content": reply})
    await send_whatsapp(sender, reply)


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
