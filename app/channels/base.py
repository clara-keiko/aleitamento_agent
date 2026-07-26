"""Contrato comum entre provedores de WhatsApp.

O gargalo do projeto é o número, não a lógica. Manter o canal atrás de uma
interface permite trocar Meta Cloud API por Twilio (ou por outro BSP) mudando
uma variável de ambiente, sem tocar em guardrails nem no RAG.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class IncomingMessage:
    """Mensagem recebida, já normalizada entre provedores."""

    message_id: str
    sender: str
    kind: str  # "text" | "audio" | "unsupported"
    text: str = ""
    media_id: str = ""
    media_url: str = ""
    media_mime: str = ""


class Channel(Protocol):
    name: str

    def verify_signature(self, raw_body: bytes, headers: dict, url: str) -> bool:
        """Confirma que a requisição veio mesmo do provedor."""

    def parse_webhook(self, payload: dict) -> list[IncomingMessage]:
        """Extrai mensagens do usuário; ignora eventos de status e ecos."""

    def send_text(self, to: str, body: str) -> bool:
        """Envia texto. Retorna True em caso de aceite pelo provedor."""

    def fetch_media(self, message: IncomingMessage) -> bytes | None:
        """Baixa o áudio de uma mensagem de voz, ou None se indisponível."""


# WhatsApp corta textos longos; deixamos margem para o sufixo de disclaimer.
MAX_MESSAGE_CHARS = 3800


def split_message(body: str, limit: int = MAX_MESSAGE_CHARS) -> list[str]:
    """Divide uma resposta longa em partes que o WhatsApp aceita."""
    body = (body or "").strip()
    if not body:
        return []
    if len(body) <= limit:
        return [body]

    parts: list[str] = []
    remaining = body
    while len(remaining) > limit:
        # Prefere quebrar em parágrafo, depois em frase, e só então no limite.
        window = remaining[:limit]
        cut = window.rfind("\n\n")
        if cut < limit // 2:
            cut = window.rfind(". ")
            if cut != -1:
                cut += 1
        if cut < limit // 2:
            cut = limit
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    if remaining:
        parts.append(remaining)
    return parts
