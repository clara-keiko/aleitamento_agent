import os
import requests
from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse
from typing import Annotated
from collections import defaultdict, deque
from openai import OpenAI

app = FastAPI()

#VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")

client = OpenAI(api_key=OPENAI_API_KEY)

memory = defaultdict(lambda: deque(maxlen=4))


SAFE_SYSTEM_PROMPT = """
Você é um assistente educativo sobre bebês, amamentação e puericultura.

Tom:
acolhedor, empático e claro.

Regras:
- não faça diagnóstico
- não prescreva medicamentos
- não substitua avaliação médica
- seja breve, mas sempre gentil
"""


# -------------------------
# RED FLAG
# -------------------------

EMERGENCY = ["convuls", "não respira", "inconsciente"]

def classify_risk(text):

    t = text.lower()

    if any(w in t for w in EMERGENCY):
        return "emergency"

    return "safe"


# -------------------------
# INTENT CLASSIFIER
# -------------------------

def classify_intent(text):

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
        input=prompt
    )

    return r.output_text.strip().lower()


# -------------------------
# WHATSAPP
# -------------------------

def send_message(phone, text):

    url = f"https://graph.facebook.com/v25.0/{PHONE_NUMBER_ID}/messages"

    payload = {
        "messaging_product":"whatsapp",
        "to":phone,
        "type":"text",
        "text":{"body":text}
    }

    headers = {
        "Authorization":f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type":"application/json"
    }

    requests.post(url,headers=headers,json=payload)


def typing_indicator(message_id):

    url = f"https://graph.facebook.com/v25.0/{PHONE_NUMBER_ID}/messages"

    payload = {
        "messaging_product":"whatsapp",
        "status":"read",
        "message_id":message_id,
        "typing_indicator":{"type":"text"}
    }

    headers = {
        "Authorization":f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type":"application/json"
    }

    requests.post(url,headers=headers,json=payload)


# -------------------------
# RAG
# -------------------------

def generate_answer(question):

    r = client.responses.create(
        model="gpt-5",
        input=[
            {"role":"system","content":SAFE_SYSTEM_PROMPT},
            {"role":"user","content":question}
        ],
        tools=[
            {
                "type":"file_search",
                "vector_store_ids":[VECTOR_STORE_ID],
                "max_num_results":6
            }
        ]
    )

    return r.output_text.strip()


# -------------------------
# SHORT REPLIES
# -------------------------

def greeting_reply():

    return "Oi! 😊 Posso ajudar com dúvidas sobre bebês, amamentação ou vacinas."


def thanks_reply():

    return "De nada! 😊 Se tiver outra dúvida é só perguntar."


def smalltalk_reply():

    return "Se quiser, pode me perguntar algo sobre seu bebê."


# -------------------------
# WEBHOOK VERIFY
# -------------------------

@app.get("/webhook",response_class=PlainTextResponse)
def verify_webhook(
    hub_mode: Annotated[str|None,Query(alias="hub.mode")]=None,
    hub_verify_token: Annotated[str|None,Query(alias="hub.verify_token")]=None,
    hub_challenge: Annotated[str|None,Query(alias="hub.challenge")]=None
):

    if hub_mode=="subscribe" and hub_verify_token==VERIFY_TOKEN:
        return hub_challenge

    return "Forbidden"


# -------------------------
# WEBHOOK MESSAGE
# -------------------------

@app.post("/webhook")
async def receive(request:Request):

    data = await request.json()

    try:

        value=data["entry"][0]["changes"][0]["value"]

        if "messages" not in value:
            return {"status":"ok"}

        msg=value["messages"][0]

        if msg["type"]!="text":
            return {"status":"ok"}

        phone=msg["from"]
        text=msg["text"]["body"]
        message_id=msg["id"]

        risk=classify_risk(text)

        if risk=="emergency":

            send_message(
                phone,
                "Isso pode ser uma emergência. Procure atendimento médico imediatamente."
            )

            return {"status":"ok"}

        intent=classify_intent(text)

        if intent=="greeting":

            send_message(phone,greeting_reply())
            return {"status":"ok"}

        if intent=="thanks":

            send_message(phone,thanks_reply())
            return {"status":"ok"}

        if intent=="smalltalk":

            send_message(phone,smalltalk_reply())
            return {"status":"ok"}

        typing_indicator(message_id)

        answer=generate_answer(text)

        send_message(phone,answer)

    except Exception as e:

        print("ERRO:",e)

    return {"status":"ok"}
