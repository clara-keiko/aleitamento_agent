from typing import Annotated
import os
import requests

from fastapi import FastAPI, Query, Request
from fastapi.responses import PlainTextResponse
from openai import OpenAI

app = FastAPI()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "puericultura_token")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)

RED_FLAGS = [
    "febre",
    "não respira",
    "nao respira",
    "falta de ar",
    "muito molinho",
    "muito sonolento",
    "convuls",
    "não mama",
    "nao mama",
    "não quer mamar",
    "nao quer mamar",
    "roxo",
    "arroxeado",
    "desidrat",
    "sem xixi",
    "pouco xixi",
    "sangue nas fezes",
]

SAFE_SYSTEM_PROMPT = """
Você é um assistente educativo em puericultura e amamentação.

Regras obrigatórias:
- Não faça diagnóstico.
- Não prescreva medicamentos.
- Não substitua avaliação médica.
- Responda apenas com orientação educativa geral.
- Se houver qualquer sinal de alerta, diga para procurar atendimento médico.
- Seja breve, clara e conservadora.
"""

@app.get("/webhook", response_class=PlainTextResponse)
def verify_webhook(
    hub_mode: Annotated[str | None, Query(alias="hub.mode")] = None,
    hub_verify_token: Annotated[str | None, Query(alias="hub.verify_token")] = None,
    hub_challenge: Annotated[str | None, Query(alias="hub.challenge")] = None,
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN and hub_challenge:
        return hub_challenge
    return PlainTextResponse("Forbidden", status_code=403)

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
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    print("SEND STATUS:", r.status_code)
    print("SEND BODY:", r.text)

def detect_red_flags(text: str) -> list[str]:
    t = text.lower()
    hits = [flag for flag in RED_FLAGS if flag in t]
    return hits

def medical_referral_message() -> str:
    return (
        "Seu relato pode indicar um sinal de alerta. "
        "Procure avaliação médica o quanto antes. "
        "Se houver dificuldade para respirar, sonolência excessiva, febre, piora ou recusa para mamar, "
        "busque atendimento imediatamente."
    )

def generate_safe_reply(user_text: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": SAFE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Pergunta do cuidador: {user_text}\n\n"
                           f"Responda apenas com orientação educativa segura e conservadora."
            },
        ],
    )
    return response.output_text.strip()

@app.post("/webhook")
async def receive_webhook(request: Request):
    data = await request.json()
    print("EVENTO:", data)

    try:
        value = data["entry"][0]["changes"][0]["value"]

        # ignora updates de status
        if "statuses" in value:
            return {"status": "ok"}

        if "messages" not in value:
            return {"status": "ok"}

        msg = value["messages"][0]

        if msg.get("type") != "text":
            return {"status": "ok"}

        phone = msg["from"]
        text = msg["text"]["body"]

        # Guardrail 1: regras duras
        red_flags = detect_red_flags(text)
        if red_flags:
            send_whatsapp_text(phone, medical_referral_message())
            return {"status": "ok"}

        # Guardrail 2: IA só em caso liberado
        reply = generate_safe_reply(text)

        # Guardrail 3: pós-checagem simples
        if any(flag in reply.lower() for flag in ["diagnóstico", "prescrev", "medicamento"]):
            reply = "Posso ajudar apenas com orientação educativa geral. Se houver preocupação clínica, procure avaliação médica."

        send_whatsapp_text(phone, reply)

    except Exception as e:
        print("Erro:", str(e))

    return {"status": "ok"}
