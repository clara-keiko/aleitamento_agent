import os
import requests
from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse
from typing import Annotated
from collections import defaultdict, deque

from openai import OpenAI

app = FastAPI()

# =========================
# ENV
# =========================

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# MEMÓRIA CONVERSA
# =========================

conversation_memory = defaultdict(lambda: deque(maxlen=4))

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
- humano

Regras obrigatórias:

- Responda principalmente com base nos documentos recuperados.
- Pode usar conhecimento médico geral confiável se necessário.
- Não faça diagnóstico.
- Não prescreva medicamentos.
- Não substitua avaliação médica.
- Se houver sinais de alerta, oriente procurar atendimento médico.
- Use linguagem simples e reconfortante.
- Seja breve mas gentil.
"""

# =========================
# RED FLAGS
# =========================

EMERGENCY_FLAGS = [
    "não respira",
    "convuls",
    "inconsciente",
    "roxo",
    "arroxeado",
]

MEDICAL_FLAGS = [
    "febre",
    "muito sonolento",
    "não mama",
    "desidrat",
    "vomitando tudo",
]

def classify_risk(text: str):

    t = text.lower()

    if any(flag in t for flag in EMERGENCY_FLAGS):
        return "EMERGENCY"

    if any(flag in t for flag in MEDICAL_FLAGS):
        return "MEDICAL"

    return "SAFE"


# =========================
# WHATSAPP
# =========================

def send_whatsapp_text(phone, body):

    url = f"https://graph.facebook.com/v25.0/{PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": phone,
        "type": "text",
        "text": {"body": body},
    }

    requests.post(url, headers=headers, json=payload, timeout=30)


def send_typing_indicator(message_id):

    url = f"https://graph.facebook.com/v25.0/{PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
        "typing_indicator": {"type": "text"},
    }

    requests.post(url, headers=headers, json=payload, timeout=30)


# =========================
# QUERY REWRITE
# =========================

def rewrite_query(question):

    prompt = f"""
Reescreva a pergunta abaixo de forma mais completa para busca em documentos
sobre puericultura e amamentação.

Pergunta original:
{question}
"""

    r = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return r.output_text.strip()


# =========================
# VECTOR SEARCH
# =========================

def vector_search(query):

    r = client.vector_stores.search(
        vector_store_id=VECTOR_STORE_ID,
        query=query,
        max_num_results=8
    )

    docs = []

    for item in r.data:
        docs.append(item.content[0].text)

    return docs


# =========================
# RERANK
# =========================

def rerank_docs(question, docs):

    joined = "\n\n".join(docs)

    prompt = f"""
Pergunta do usuário:
{question}

Trechos de documentos:

{joined}

Escolha os 5 trechos mais relevantes para responder a pergunta.
"""

    r = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return r.output_text


# =========================
# RESPOSTA FINAL
# =========================

def generate_answer(phone, question):

    query = rewrite_query(question)

    docs = vector_search(query)

    context = rerank_docs(query, docs)

    history = list(conversation_memory[phone])

    messages = [
        {"role": "system", "content": SAFE_SYSTEM_PROMPT}
    ]

    messages += history

    messages.append({
        "role": "user",
        "content": f"""
Documentos relevantes:

{context}

Pergunta:
{question}
"""
    })

    r = client.responses.create(
        model="gpt-5-mini",
        input=messages
    )

    answer = r.output_text.strip()

    conversation_memory[phone].append({
        "role": "user",
        "content": question
    })

    conversation_memory[phone].append({
        "role": "assistant",
        "content": answer
    })

    return answer


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
# WEBHOOK MESSAGES
# =========================

@app.post("/webhook")
async def receive_webhook(request: Request):

    data = await request.json()

    try:

        entry = data["entry"][0]["changes"][0]["value"]

        if "messages" not in entry:
            return {"status": "ok"}

        msg = entry["messages"][0]

        if msg["type"] != "text":
            return {"status": "ok"}

        phone = msg["from"]
        text = msg["text"]["body"]
        message_id = msg["id"]

        risk = classify_risk(text)

        if risk == "EMERGENCY":

            send_whatsapp_text(
                phone,
                "Isso pode ser uma situação de emergência. Procure atendimento médico imediatamente."
            )

            return {"status": "ok"}

        if risk == "MEDICAL":

            send_whatsapp_text(
                phone,
                "Entendo sua preocupação. Pode ser importante procurar avaliação médica."
            )

            return {"status": "ok"}

        # typing indicator apenas para casos seguros
        send_typing_indicator(message_id)

        answer = generate_answer(phone, text)

        send_whatsapp_text(phone, answer)

    except Exception as e:

        print("ERRO:", e)

    return {"status": "ok"}
