from typing import Annotated
import os
import requests

from fastapi import FastAPI, Query, Request
from fastapi.responses import PlainTextResponse
from openai import OpenAI

app = FastAPI()

# =========================
# Variáveis de ambiente
# =========================
#VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID", "")

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
# Prompt seguro
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
- Seja breve, mas gentil.
"""

# =========================
# Guardrails clínicos
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
        "Entendo sua preocupação. Pelo que você descreveu, isso pode ser uma situação de urgência. "
        "Procure atendimento médico de emergência imediatamente."
    )

def medical_referral_message() -> str:
    return (
        "Entendo sua preocupação. Pelo que você descreveu, pode haver um sinal de alerta. "
        "O mais seguro é procurar avaliação médica o quanto antes. "
        "Se houver dificuldade para respirar, sonolência excessiva, febre, piora "
        "ou recusa para mamar, busque atendimento imediatamente."
    )

# =========================
# WhatsApp
# =========================
def send_whatsapp_text(to: str, body: str) -> None:
    url = f"https://graph.facebook.com/v25.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body},
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print("WHATSAPP SEND STATUS:", response.status_code)
    print("WHATSAPP SEND BODY:", response.text)

def send_typing_indicator(message_id: str) -> None:
    """
    Marca a mensagem como lida e exibe o indicador de 'digitando'
    para casos seguros, antes da consulta à base + IA.
    """
    url = f"https://graph.facebook.com/v25.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
        "typing_indicator": {
            "type": "text"
        }
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print("TYPING STATUS:", response.status_code)
    print("TYPING BODY:", response.text)

# =========================
# OpenAI + file_search
# =========================
def generate_safe_reply(user_text: str) -> str:
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": SAFE_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [VECTOR_STORE_ID],
                    "max_num_results": 3,
                }
            ],
        )

        answer = response.output_text.strip()

        # Pós-checagem simples
        lowered = answer.lower()
        blocked_terms = ["diagnóstico", "diagnostico", "prescrev", "medicamento"]

        if any(term in lowered for term in blocked_terms):
            return (
                "Quero te ajudar da forma mais segura possível. "
                "Posso oferecer apenas orientação educativa geral com base nos materiais aprovados. "
                "Se houver preocupação clínica, o mais seguro é procurar avaliação médica."
            )

        return answer

    except Exception as e:
        print("OPENAI ERROR:", str(e))
        return (
            "Entendo sua preocupação. No momento só consigo oferecer orientação educativa geral limitada. "
            "Se houver febre, dificuldade para respirar, piora, sonolência excessiva "
            "ou recusa para mamar, procure atendimento médico."
        )

# =========================
# Webhook GET (validação Meta)
# =========================
@app.get("/webhook", response_class=PlainTextResponse)
def verify_webhook(
    hub_mode: Annotated[str | None, Query(alias="hub.mode")] = None,
    hub_verify_token: Annotated[str | None, Query(alias="hub.verify_token")] = None,
    hub_challenge: Annotated[str | None, Query(alias="hub.challenge")] = None,
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN and hub_challenge:
        return hub_challenge

    return PlainTextResponse("Forbidden", status_code=403)

# =========================
# Webhook POST (mensagens)
# =========================
@app.post("/webhook")
async def receive_webhook(request: Request):
    data = await request.json()
    print("EVENTO:", data)

    try:
        entry = data.get("entry", [])
        if not entry:
            return {"status": "ok"}

        changes = entry[0].get("changes", [])
        if not changes:
            return {"status": "ok"}

        value = changes[0].get("value", {})

        # Ignora eventos de status
        if "statuses" in value:
            print("STATUS EVENT:", value["statuses"])
            return {"status": "ok"}

        if "messages" not in value:
            return {"status": "ok"}

        msg = value["messages"][0]

        # Trata apenas texto no MVP
        if msg.get("type") != "text":
            print("Mensagem ignorada. Tipo não suportado:", msg.get("type"))
            return {"status": "ok"}

        phone = msg["from"]
        text = msg["text"]["body"]
        message_id = msg["id"]

        print("PHONE:", phone)
        print("TEXT:", text)

        # Guardrail antes da IA
        risk = classify_risk(text)
        print("RISK:", risk)

        # Para red flags, responde imediatamente sem digitação
        if risk == "EMERGENCY_NOW":
            send_whatsapp_text(phone, emergency_message())
            return {"status": "ok"}

        if risk == "REFER_MEDICAL_CARE":
            send_whatsapp_text(phone, medical_referral_message())
            return {"status": "ok"}

        # Para casos seguros, mostra "digitando" antes de consultar a base
        send_typing_indicator(message_id)

        # Caso seguro: usa OpenAI + file_search
        reply = generate_safe_reply(text)
        send_whatsapp_text(phone, reply)

    except Exception as e:
        print("WEBHOOK ERROR:", str(e))

    return {"status": "ok"}
