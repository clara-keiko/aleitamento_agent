import logging
import time
from typing import Annotated, Optional
from collections import defaultdict
import os
import re
import requests
import threading

from fastapi import FastAPI, Query, Request, BackgroundTasks
from fastapi.responses import PlainTextResponse
from openai import OpenAI
from pydantic import BaseModel

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="WhatsApp Bot API", version="1.0.0")

# =========================
# Variáveis de ambiente
# =========================
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

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
# Loop Prevention
# =========================
class MessageDeduplicator:
    """Previne processamento duplicado de mensagens."""
    
    def __init__(self, ttl_seconds: int = 60):
        self.processed_ids: dict[str, float] = {}
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
    
    def is_duplicate(self, message_id: str) -> bool:
        """Verifica se mensagem já foi processada."""
        self._cleanup()
        
        with self._lock:
            if message_id in self.processed_ids:
                logger.warning(f"Duplicate message detected: {message_id}")
                return True
            
            self.processed_ids[message_id] = time.time()
            return False
    
    def _cleanup(self) -> None:
        """Remove mensagens expiradas."""
        now = time.time()
        with self._lock:
            expired = [
                msg_id for msg_id, timestamp in self.processed_ids.items()
                if now - timestamp > self.ttl
            ]
            for msg_id in expired:
                del self.processed_ids[msg_id]


deduplicator = MessageDeduplicator(ttl_seconds=60)

# =========================
# Short Memory (por usuário)
# =========================
class ConversationMemory:
    """Memória curta de conversação por usuário."""
    
    def __init__(self, max_messages: int = 10, ttl_seconds: int = 3600):
        self.conversations: dict[str, list[dict]] = defaultdict(list)
        self.timestamps: dict[str, float] = {}
        self.max_messages = max_messages
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
    
    def add_message(self, phone: str, role: str, content: str) -> None:
        """Adiciona mensagem à memória."""
        self._cleanup_user(phone)
        
        with self._lock:
            self.conversations[phone].append({
                "role": role,
                "content": content
            })
            
            # Manter apenas últimas N mensagens
            if len(self.conversations[phone]) > self.max_messages:
                self.conversations[phone] = self.conversations[phone][-self.max_messages:]
            
            self.timestamps[phone] = time.time()
    
    def get_history(self, phone: str) -> list[dict]:
        """Retorna histórico de mensagens do usuário."""
        self._cleanup_user(phone)
        
        with self._lock:
            return self.conversations[phone].copy()
    
    def clear(self, phone: str) -> None:
        """Limpa histórico do usuário."""
        with self._lock:
            self.conversations.pop(phone, None)
            self.timestamps.pop(phone, None)
    
    def _cleanup_user(self, phone: str) -> None:
        """Remove histórico expirado."""
        with self._lock:
            if phone in self.timestamps:
                if time.time() - self.timestamps[phone] > self.ttl:
                    self.conversations.pop(phone, None)
                    self.timestamps.pop(phone, None)


memory = ConversationMemory(max_messages=10, ttl_seconds=3600)

# =========================
# Models
# =========================
class WebhookResponse(BaseModel):
    status: str


class RiskLevel:
    EMERGENCY_NOW = "EMERGENCY_NOW"
    REFER_MEDICAL_CARE = "REFER_MEDICAL_CARE"
    SAFE = "SAFE"


# =========================
# Prompt seguro
# =========================
SAFE_SYSTEM_PROMPT = """
Você é um assistente educativo em puericultura e amamentação.
Responda de forma clara, empática e segura.
Nunca forneça diagnósticos médicos.
Em caso de emergência, oriente a buscar atendimento imediato.
Se não tiver certeza sobre uma informação, deixe isso explícito.
Prefira respostas completas, com passos práticos e objetivos.
"""

# =========================
# WhatsApp API Functions
# =========================
MAX_WHATSAPP_TEXT_LENGTH = 4096


def normalize_phone(phone: str) -> str:
    """Mantém apenas dígitos no número de telefone."""
    return re.sub(r"\D", "", phone or "")


def sanitize_whatsapp_text(text: str) -> str:
    """Garante texto válido para a API do WhatsApp."""
    clean_text = (text or "").strip()
    if not clean_text:
        return "Desculpe, ocorreu um erro ao gerar a resposta."

    if len(clean_text) > MAX_WHATSAPP_TEXT_LENGTH:
        return clean_text[: MAX_WHATSAPP_TEXT_LENGTH - 3] + "..."

    return clean_text


def send_whatsapp_text(phone: str, text: str) -> bool:
    """Envia mensagem de texto via WhatsApp."""
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    normalized_phone = normalize_phone(phone)
    message_text = sanitize_whatsapp_text(text)

    if not normalized_phone:
        logger.error("Failed to send message: invalid phone number")
        return False

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": normalized_phone,
        "type": "text",
        "text": {"body": message_text}
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(f"Message sent to {phone[-4:]}")
        return True
    except requests.HTTPError as e:
        response_body = e.response.text if e.response is not None else ""
        logger.error(f"Failed to send message: {e} | response: {response_body}")
        return False
    except requests.RequestException as e:
        logger.error(f"Failed to send message: {e}")
        return False


def send_typing_indicator(phone: str) -> bool:
    """Envia indicador de digitação (typing...)."""
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": phone,
        "recipient_type": "individual",
        "type": "reaction",
        "reaction": {
            "message_id": "",
            "emoji": ""
        }
    }
    
    # WhatsApp Business API usa "typing_on" action
    payload = {
        "messaging_product": "whatsapp",
        "to": phone,
        "type": "text",
        "text": {"body": "..."}  # Placeholder - substitua pelo método correto da sua API
    }
    
    # Método correto para typing indicator (se disponível na sua versão da API)
    try:
        # Nota: A API oficial do WhatsApp Business não tem typing indicator nativo
        # Esta é uma aproximação - você pode usar "read receipts" ou um método customizado
        logger.debug(f"Typing indicator sent to {phone[-4:]}")
        return True
    except Exception as e:
        logger.error(f"Failed to send typing indicator: {e}")
        return False


def mark_as_read(phone: str, message_id: str) -> bool:
    """Marca mensagem como lida."""
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        logger.error(f"Failed to mark as read: {e}")
        return False


# =========================
# Risk Classification
# =========================
EMERGENCY_KEYWORDS = [
    "engasgou", "engasgando", "não respira", "convulsão", 
    "desmaio", "sangramento", "queda", "bateu a cabeça",
    "febre alta", "roxo", "cianose", "parou de respirar"
]

MEDICAL_KEYWORDS = [
    "febre", "vômito", "diarreia", "manchas", "alergia",
    "não come", "chorando muito", "irritado", "moleira"
]


def classify_risk(text: str) -> str:
    """Classifica o risco da mensagem."""
    text_lower = text.lower()
    
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in text_lower:
            return RiskLevel.EMERGENCY_NOW
    
    for keyword in MEDICAL_KEYWORDS:
        if keyword in text_lower:
            return RiskLevel.REFER_MEDICAL_CARE
    
    return RiskLevel.SAFE


def emergency_message() -> str:
    return """🚨 EMERGÊNCIA DETECTADA

Por favor, procure IMEDIATAMENTE atendimento médico de emergência:
• SAMU: 192
• Bombeiros: 193
• Pronto-socorro mais próximo

Sua segurança é prioridade. Não espere!"""


def medical_referral_message() -> str:
    return """⚠️ ATENÇÃO

Os sintomas que você descreveu precisam de avaliação médica.
Por favor, consulte um pediatra ou procure uma unidade de saúde.

Posso ajudar com dúvidas educativas sobre amamentação e cuidados gerais."""


# =========================
# AI Response Generation
# =========================
def generate_safe_reply(phone: str, text: str) -> str:
    """Gera resposta segura usando OpenAI com memória."""
    try:
        # Adiciona mensagem do usuário à memória
        memory.add_message(phone, "user", text)
        
        # Monta histórico para contexto
        history = memory.get_history(phone)
        
        messages = [{"role": "system", "content": SAFE_SYSTEM_PROMPT}]
        messages.extend(history)
        
        response = None
        max_output_tokens = 280

        # Compatibilidade entre versões/modelos que aceitam
        # max_completion_tokens vs max_tokens.
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_completion_tokens=max_output_tokens
            )
        except Exception as first_error:
            logger.warning(
                "chat.completions com max_completion_tokens falhou, "
                "tentando fallback com max_tokens. Erro: %s",
                first_error
            )
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=max_output_tokens
            )

        first_choice = response.choices[0]
        choice = first_choice.message
        reply = ""

        # Compatibilidade com diferentes formatos do SDK
        if isinstance(choice.content, str):
            reply = choice.content
        elif isinstance(choice.content, list):
            text_parts = []
            for part in choice.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif hasattr(part, "type") and getattr(part, "type") == "text":
                    text_parts.append(getattr(part, "text", ""))
            reply = "".join(text_parts)

        reply = sanitize_whatsapp_text(reply)

        finish_reason = getattr(first_choice, "finish_reason", None)
        if finish_reason == "length":
            reply = sanitize_whatsapp_text(
                f"{reply}\n\nSe quiser, posso continuar a explicação em mais detalhes."
            )

        # Adiciona resposta à memória
        memory.add_message(phone, "assistant", reply)
        
        return reply
        
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return "Desculpe, não consegui processar sua mensagem. Tente novamente."


# =========================
# Message Handlers
# =========================
def handle_emergency(phone: str, message_id: str) -> None:
    """Handle emergency cases."""
    logger.warning(f"Emergency detected for phone: {phone[-4:]}")
    mark_as_read(phone, message_id)
    send_whatsapp_text(phone, emergency_message())


def handle_medical_referral(phone: str, message_id: str) -> None:
    """Handle medical referral cases."""
    logger.info(f"Medical referral for phone: {phone[-4:]}")
    mark_as_read(phone, message_id)
    send_whatsapp_text(phone, medical_referral_message())


def handle_safe_message(phone: str, text: str, message_id: str) -> None:
    """Handle safe messages with AI response."""
    logger.info(f"Processing safe message for phone: {phone[-4:]}")
    mark_as_read(phone, message_id)
    reply = generate_safe_reply(phone, text)
    send_whatsapp_text(phone, reply)


# =========================
# Webhook Helpers
# =========================
def extract_message(payload: dict) -> Optional[tuple[str, str, str]]:
    """Extract phone, text and message_id from webhook payload."""
    try:
        entry = payload.get("entry", [])
        if not entry:
            return None
        
        changes = entry[0].get("changes", [])
        if not changes:
            return None
        
        value = changes[0].get("value", {})
        
        # Ignorar status updates (loop prevention)
        if "statuses" in value:
            logger.debug("Ignoring status update")
            return None
        
        messages = value.get("messages", [])
        if not messages:
            return None
        
        msg = messages[0]
        
        # Tratar só texto no MVP
        if msg.get("type") != "text":
            logger.debug(f"Ignoring non-text message type: {msg.get('type')}")
            return None
        
        phone = msg.get("from")
        text = msg.get("text", {}).get("body")
        message_id = msg.get("id")
        
        if not phone or not text or not message_id:
            return None
        
        return phone, text, message_id
        
    except (KeyError, IndexError) as e:
        logger.error(f"Error extracting message: {e}")
        return None


def process_message(phone: str, text: str, message_id: str) -> None:
    """Process incoming message with guardrails."""
    try:
        # Guardrail 1: classify risk before AI
        risk = classify_risk(text)
        logger.info(f"Risk classification: {risk}")
        
        if risk == RiskLevel.EMERGENCY_NOW:
            handle_emergency(phone, message_id)
            return
        
        if risk == RiskLevel.REFER_MEDICAL_CARE:
            handle_medical_referral(phone, message_id)
            return
        
        # Guardrail 2: only safe cases reach AI
        handle_safe_message(phone, text, message_id)
        
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        send_whatsapp_text(phone, "Desculpe, ocorreu um erro. Tente novamente.")


# =========================
# Endpoints
# =========================
@app.get("/webhook")
async def verify_webhook(
    hub_mode: Annotated[str, Query(alias="hub.mode")] = "",
    hub_challenge: Annotated[str, Query(alias="hub.challenge")] = "",
    hub_verify_token: Annotated[str, Query(alias="hub.verify_token")] = ""
):
    """Webhook verification endpoint for WhatsApp."""
    verify_token = os.getenv("VERIFY_TOKEN", "")
    
    if hub_mode == "subscribe" and hub_verify_token == verify_token:
        logger.info("Webhook verified successfully")
        return PlainTextResponse(content=hub_challenge)
    
    logger.warning("Webhook verification failed")
    return PlainTextResponse(content="Forbidden", status_code=403)


@app.post("/webhook", response_model=WebhookResponse)
async def webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle incoming WhatsApp webhook."""
    try:
        payload = await request.json()
        logger.debug("Received webhook payload")
        
        # Extract message data
        message_data = extract_message(payload)
        if not message_data:
            return WebhookResponse(status="ok")
        
        phone, text, message_id = message_data
        
        # Loop prevention: check for duplicate messages
        if deduplicator.is_duplicate(message_id):
            return WebhookResponse(status="ok")
        
        logger.info(f"Message from: {phone[-4:]}, id: {message_id[:8]}...")
        
        # Process in background for faster response
        background_tasks.add_task(process_message, phone, text, message_id)
        
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}", exc_info=True)
    
    return WebhookResponse(status="ok")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_conversations": len(memory.conversations),
        "processed_messages": len(deduplicator.processed_ids)
    }


@app.post("/clear-memory/{phone}")
async def clear_user_memory(phone: str):
    """Clear conversation memory for a user."""
    memory.clear(phone)
    return {"status": "cleared", "phone": phone[-4:]}
