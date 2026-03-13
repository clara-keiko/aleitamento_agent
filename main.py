import os
import requests
from collections import deque
from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse
from typing import Annotated
from openai import OpenAI

app = FastAPI()

# =========================
# ENV
# =========================
#VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")

#if not VERIFY_TOKEN:
    #raise ValueError("VERIFY_TOKEN não configurado")
if not WHATSAPP_TOKEN:
    raise ValueError("WHATSAPP_TOKEN não configurado")
if not PHONE_NUMBER_ID:
    raise ValueError("PHONE_NUMBER_ID não configurado")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY não configurado")
if not VECTOR_STORE_ID:
    raise ValueError("VECTOR_STORE_ID não configurado")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# LOOP PROTECTION
# =========================
processed_message_ids = set()
processed_message_queue = deque(maxlen=1000)

def mark_processed(message_id: str) -> bool:
    if message_id in processed_message_ids:
        return False

    if len(processed_message_queue) == processed_message_queue.maxlen:
        old = processed_message_queue.popleft()
        processed_message_ids.discard(old)

    processed_message_queue.append(message_id)
    processed_message_ids.add(message_id)
    return True

# =========================
# PROMPT
# =========================
SAFE_SYSTEM_PROMPT = """
Você é um assistente educativo em puericultura e amamentação.

Seu tom deve ser:
- acolhedor
- empático
- cordial
- calmo
- claro
- respeitoso
- humano, sem soar robótico

Regras obrigatórias:
- Responda apenas com base no conteúdo recuperado dos arquivos.
- Não faça diagnóstico.
- Não prescreva medicamentos.
- Não substitua avaliação médica.
- Não invente informações fora dos arquivos.
- Se a base recuperada for insuficiente, diga isso claramente.
- Use português do Brasil.
- Priorize linguagem simples e reconfortante.
- Sempre valide brevemente a preocupação do cuidador antes de orientar.
- Evite soar fria, seca ou excessivamente técnica.
- Responda em no máximo 3 frases curtas.
- Evite explicações longas.
- Seja breve, mas gentil.
"""

# =========================
# RED FLAGS
# =========================
EMERGENCY_FLAGS = [
    "não respira",
    "nao respira",
    "dificuldade para respirar",
    "falta de ar",
    "convuls",
    "roxo",
    "arroxeado",
    "inconsciente",
]

REFER_MEDICAL_CARE_FLAGS = [
    "febre",
    "muito molinho",
    "muito sonolento",
    "não mama",
    "nao mama",
    "não quer mamar",
    "nao quer mamar",
    "desidrat",
    "sem xixi",
    "pouco xixi",
    "sangue nas fezes",
    "vomitando tudo",
    "vomita tudo",
    "pele muito amarela",
]

def classify_risk(text: str) -> str:
    t = text.lower()

    if any(flag in t for flag in EMERGENCY_FLAGS):
        return "EMERGENCY_NOW"

    if any(flag in t for flag in REFER_MEDICAL_CARE_FLAGS):
        return "REFER_MEDICAL_CARE"

    return "EDUCATIONAL_OK"

def emergency_message() -> str:
    return (
        "Entendo sua preocupação. Isso pode ser uma situação de urgência. "
        "Procure atendimento médico de emergência imediatamente."
    )

def medical_referral_message() -> str:
    return (
        "Entendo sua preocupação. Pode haver um sinal de alerta. "
        "O mais seguro é procurar avaliação médica o quanto antes."
    )

# =========================
# INTENT / SMALL TALK
# =========================
def classify_intent(text: str) -> str:
    prompt = f"""
Classifique a mensagem em UMA categoria:

greeting
thanks
smalltalk
medical_question

Mensagem:
{text}
"""
    r = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        max_output_tokens=20,
    )
    return r.output_text.strip().lower()

def greeting_reply() -> str:
    return "Oi! 😊 Posso ajudar com dúvidas sobre bebês, amamentação ou vacinas."

def thanks_reply() -> str:
    return "De nada! 😊 Se tiver outra dúvida, é só me chamar."

def smalltalk_reply() -> str:
    return "Pode me mandar sua dúvida sobre bebê, amamentação ou puericultura."

# =========================
# WHATSAPP
# =========================
def send_message(phone: str, text: str) -> None:
    url = f"https://graph.facebook.com/v25.0/{PHONE_NUMBER_ID}/messages"

    payload = {
        "messaging_product": "whatsapp",
        "to": phone,
        "type": "text",
        "text": {"body": text},
    }

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print("SEND STATUS:", response.status_code)
    print("SEND BODY:", response.text)

def typing_indicator(message_id: str) -> None:
    url = f"https://graph.facebook.com/v25.0/{PHONE_NUMBER_ID}/messages"

    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
        "typing_indicator": {"type": "text"},
    }

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print("TYPING STATUS:", response.status_code)
    print("TYPING BODY:", response.text)

# =========================
# ANSWER GENERATION
# =========================
def shorten_answer(answer: str) -> str:
    if len(answer) < 500:
        return answer

    prompt = f"""
Resuma a resposta abaixo em no máximo 3 frases curtas, mantendo um tom acolhedor e claro.

Resposta:
{answer}
"""

    r = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        max_output_tokens=120,
    )

    return r.output_text.strip()

def generate_answer(question: str) -> str:
    r = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": SAFE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [VECTOR_STORE_ID],
                "max_num_results": 6,
            }
        ],
        max_output_tokens=120,
    )

    answer = r.output_text.strip()
    return shorten_answer(answer)

# =========================
# WEBHOOK VERIFY
# =========================
@app.get("/webhook", response_class=PlainTextResponse)
def verify_webhook(
    hub_mode: Annotated[str | None, Query(alias="hub.mode")] = None,
    hub_verify_token: Annotated[str | None, Query(alias="hub.verify_token")] = None,
    hub_challenge: Annotated[str | None, Query(alias="hub.challenge")] = None,
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return hub_challenge

    return "Forbidden"

# =========================
# WEBHOOK MESSAGE
# =========================
@app.post("/webhook")
async def receive(request: Request):
    data = await request.json()

    try:
        value = data["entry"][0]["changes"][0]["value"]

        # 1) Ignore all status callbacks
        if "statuses" in value:
            return {"status": "ok"}

        # 2) Only handle real inbound user messages
        messages = value.get("messages")
        if not messages:
            return {"status": "ok"}

        msg = messages[0]

        # 3) Only text messages
        if msg.get("type") != "text":
            return {"status": "ok"}

        # 4) Deduplicate by message id
        message_id = msg.get("id")
        if not message_id:
            return {"status": "ok"}

        if not mark_processed(message_id):
            return {"status": "ok"}

        # 5) Ignore messages sent by your own business number
        business_display_phone = value.get("metadata", {}).get("display_phone_number")
        sender = msg.get("from")
        if sender == business_display_phone:
            return {"status": "ok"}

        phone = sender
        text = msg["text"]["body"]

        risk = classify_risk(text)

        if risk == "EMERGENCY_NOW":
            send_message(phone, emergency_message())
            return {"status": "ok"}

        if risk == "REFER_MEDICAL_CARE":
            send_message(phone, medical_referral_message())
            return {"status": "ok"}

        intent = classify_intent(text)

        if intent == "greeting":
            send_message(phone, greeting_reply())
            return {"status": "ok"}

        if intent == "thanks":
            send_message(phone, thanks_reply())
            return {"status": "ok"}

        if intent == "smalltalk":
            send_message(phone, smalltalk_reply())
            return {"status": "ok"}

        typing_indicator(message_id)
        answer = generate_answer(text)
        send_message(phone, answer)

    except Exception as e:
        print("WEBHOOK ERROR:", e)
        print("PAYLOAD:", data)

    return {"status": "ok"}
