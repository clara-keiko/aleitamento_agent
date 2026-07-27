"""Canal web: protótipo no navegador, mesmo cérebro do WhatsApp.

O ponto deste canal é ser uma *pré-visualização fiel*, não uma demo paralela.
Ele passa exatamente pelo mesmo pipeline — triagem clínica, RAG, checagem de
fundamentação, memória, rate limit. Isso significa que o que a consultora de
amamentação aprovar aqui é literalmente o que a mãe vai receber no WhatsApp,
sem uma segunda implementação para divergir.

Inclusive a quebra de mensagens longas é mantida: se a resposta vira três
balões no WhatsApp, vira três balões aqui.

Diferença de mecânica: WhatsApp é assíncrono (respondemos por outra conexão),
o navegador espera a resposta na mesma requisição. Por isso `send_text`
acumula num buffer por sessão, que o endpoint drena e devolve.
"""

import threading

from app.channels.base import Channel, IncomingMessage, split_message
from app.logging_utils import get_logger

log = get_logger(__name__)


class WebChannel(Channel):
    name = "web"

    def __init__(self):
        self._lock = threading.Lock()
        self._buffers: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Contrato do Channel
    # ------------------------------------------------------------------
    def verify_signature(self, raw_body: bytes, headers: dict, url: str) -> bool:
        # Não há assinatura de provedor aqui. A proteção do endpoint é o
        # código de acesso, verificado no main antes de chegar ao pipeline.
        return True

    def parse_webhook(self, payload: dict) -> list[IncomingMessage]:
        session = (payload.get("session") or "").strip()
        text = (payload.get("text") or "").strip()
        message_id = (payload.get("message_id") or "").strip()

        if not session or not text:
            return []

        return [
            IncomingMessage(
                message_id=message_id, sender=session, kind="text", text=text
            )
        ]

    def send_text(self, to: str, body: str) -> bool:
        partes = split_message(body)
        with self._lock:
            self._buffers.setdefault(to, []).extend(partes)
        return True

    def fetch_media(self, message: IncomingMessage) -> bytes | None:
        # Áudio no protótipo web fica para depois; no WhatsApp já funciona.
        return None

    # ------------------------------------------------------------------
    # Específico do canal
    # ------------------------------------------------------------------
    def drain(self, session: str) -> list[str]:
        """Retira e devolve as mensagens acumuladas para uma sessão."""
        with self._lock:
            return self._buffers.pop(session, [])

    def discard(self, session: str) -> None:
        with self._lock:
            self._buffers.pop(session, None)
