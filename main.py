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
DEBUG_REFERENCES = os.getenv("DEBUG_REFERENCES", "false").lower() == "true"

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

Regras obrigatórias:
- Responda apenas com base no conteúdo recuperado dos arquivos.
- Não faça diagnóstico.
- Não prescreva medicamentos.
- Não substitua avaliação médica.
- Não invente informações fora dos arquivos.
- Se a base recuperada for insuficiente, diga isso claramente.
- Seja breve, clara, conservadora e educativa.
- Use português do Brasil.
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
        "Isso pode ser uma urgência. "
        "Procure atendimento médico de emergência imediatamente."
    )

def medical_referral_message() -> str:
    return (
        "Seu relato pode indicar um sinal de alerta. "
        "Procure avaliação médica o quanto antes. "
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

# =========================
# Debug de referências
# =========================
def debug_search_references(user_text: str) -> list[dict]:
    """
    Busca direta no vector store para inspecionar quais arquivos/chunks
    estão sendo recuperados para a pergunta.
    """
    try:
        results = client.vector_stores.search(
            vector_store_id=VECTOR_STORE_ID,
            query=user_text,
        )

        references = []

        # Dependendo da versão do SDK, results.data pode trazer objetos com
        # atributos diferentes. Por isso usamos getattr com fallback.
        for item in results.data:
            content_preview = None

            # tenta extrair conteúdo de forma defensiva
            if hasattr(item, "content") and item.content:
                content_preview = item.content
            elif isinstance(item, dict):
                content_preview = item.get("content")

            references.append({
                "file_id": getattr(item, "file_id", None) if not isinstance(item, dict) else item.get("file_id"),
                "filename": getattr(item, "filename", None) if not isinstance(item, dict) else item.get("filename"),
                "score": getattr(item, "score", None) if not isinstance(item, dict) else item.get("score"),
                "text": content_preview,
            })

        return references

    except Exception as e:
        print("DEBUG SEARCH ERROR:", str(e))
        return []

def print_references(refs: list[dict]) -> None:
    print("\n=== REFERÊNCIAS RECUPERADAS ===")
    if not refs:
        print("Nenhuma referência encontrada.")
    for i, ref in enumerate(refs, start=1):
        print(f"\n[{i}] arquivo: {ref.get('filename')}")
        print(f"score: {ref.get('score')}")
        print(f"file_id: {ref.get('file_id')}")
        print(f"trecho: {str(ref.get('text'))[:500]}")
    print("=== FIM REFERÊNCIAS ===\n")

def format_reference_list(refs: list[dict]) -> str:
    """
    Formata lista enxuta de arquivos consultados para anexar à resposta
    quando DEBUG_REFERENCES estiver ativo.
    """
    seen = []
    for ref in refs:
        filename = ref.get("filename")
        if filename and filename not in seen:
            seen.append(filename)

    if not seen:
        return ""

    return "\n\nFontes consultadas:\n- " + "\n- ".join(seen[:5])

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
                }
            ],
        )

        answer = response.output_text.strip()

        # Pós-checagem simples
        lowered = answer.lower()
        blocked_terms = ["diagnóstico", "diagnostico", "prescrev", "medicamento"]

        if any(term in lowered for term in blocked_terms):
            return (
                "Posso ajudar apenas com orientação educativa geral com base nos materiais aprovados. "
                "Se houver preocupação clínica, procure avaliação médica."
            )

        return answer

    except Exception as e:
        print("OPENAI ERROR:", str(e))
        return (
            "No momento só posso oferecer orientação educativa geral limitada. "
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

        # 1) status de mensagens enviadas pela empresa
        if "statuses" in value:
            print("STATUS EVENT:", value["statuses"])
            return {"status": "ok"}

        # 2) mensagens recebidas do usuário
        if "messages" not in value:
            return {"status": "ok"}

        msg = value["messages"][0]

        # trata só texto no MVP
        if msg.get("type") != "text":
            print("Mensagem ignorada. Tipo não suportado:", msg.get("type"))
            return {"status": "ok"}

        phone = msg["from"]
        text = msg["text"]["body"]

        print("PHONE:", phone)
        print("TEXT:", text)

        # Guardrail 1: risco clínico antes da IA
        risk = classify_risk(text)
        print("RISK:", risk)

        if risk == "EMERGENCY_NOW":
            send_whatsapp_text(phone, emergency_message())
            return {"status": "ok"}

        if risk == "REFER_MEDICAL_CARE":
            send_whatsapp_text(phone, medical_referral_message())
            return {"status": "ok"}

        # Debug opcional de referências
        refs = []
        if DEBUG_REFERENCES:
            refs = debug_search_references(text)
            print_references(refs)

        # Guardrail 2: só casos liberados chegam à IA
        reply = generate_safe_reply(text)

        # Se quiser ver as fontes também no WhatsApp em modo debug
        if DEBUG_REFERENCES:
            reply += format_reference_list(refs)

        send_whatsapp_text(phone, reply)

    except Exception as e:
        print("WEBHOOK ERROR:", str(e))

    return {"status": "ok"}
