"""Orquestração de uma mensagem recebida até a resposta enviada.

Ordem deliberada: deduplicação e rate limit primeiro (baratos), triagem
clínica antes do modelo (uma emergência nunca deve depender de a OpenAI estar
no ar) e checagem de fundamentação depois.
"""

import time

from app import guardrails
from app.channels.base import Channel, IncomingMessage
from app.config import Settings
from app.llm import AssistantEngine, fallback_message
from app.logging_utils import get_logger, pseudonymize
from app.memory import ConversationStore, KnownUsers, MessageDeduplicator, RateLimiter

log = get_logger(__name__)


class MessagePipeline:
    def __init__(
        self,
        settings: Settings,
        channel: Channel,
        engine: AssistantEngine,
        conversations: ConversationStore | None = None,
        deduplicator: MessageDeduplicator | None = None,
        rate_limiter: RateLimiter | None = None,
        known_users: KnownUsers | None = None,
    ):
        self.settings = settings
        self.channel = channel
        self.engine = engine
        self.conversations = conversations or ConversationStore(
            max_turns=settings.history_turns,
            ttl_seconds=settings.history_ttl_seconds,
        )
        self.deduplicator = deduplicator or MessageDeduplicator()
        self.rate_limiter = rate_limiter or RateLimiter(
            max_messages=settings.rate_limit_messages,
            window_seconds=settings.rate_limit_window_seconds,
        )
        self.known_users = known_users or KnownUsers()

    def handle(self, message: IncomingMessage) -> str:
        """Processa a mensagem e devolve o desfecho.

        O desfecho é o mesmo valor que vai para o log — e é o que o protótipo
        web mostra no modo de inspeção, para dar para ver *qual* camada agiu.
        """
        started = time.monotonic()
        user = pseudonymize(message.sender)
        outcome = "unknown"

        try:
            outcome = self._handle(message, user)
            return outcome
        finally:
            # Uma linha por mensagem, sem dado pessoal: é o que permite medir
            # taxa de fora-de-escopo, disparo de guardrail e latência.
            log.info(
                "processado user=%s outcome=%s kind=%s duracao_ms=%d",
                user,
                outcome,
                message.kind,
                int((time.monotonic() - started) * 1000),
            )

    def _handle(self, message: IncomingMessage, user: str) -> str:
        if not self.deduplicator.check_and_mark(message.message_id):
            return "duplicada"

        if not self.rate_limiter.allow(message.sender):
            self.channel.send_text(message.sender, guardrails.rate_limited_message())
            return "rate_limited"

        text = self._resolve_text(message)
        if text is None:
            self.channel.send_text(message.sender, guardrails.unsupported_media_message())
            return "nao_suportado"

        if guardrails.is_opt_out(text):
            self._forget(message.sender)
            self.channel.send_text(message.sender, guardrails.opt_out_message())
            return "opt_out"

        # Primeiro contato: escopo, aviso de automação e canal de emergência.
        if self.known_users.is_first_contact(message.sender):
            self.channel.send_text(message.sender, guardrails.welcome_message())

        # Triagem clínica antes de tudo: "oi, meu bebê não respira" é
        # emergência, não saudação.
        risk = guardrails.classify_risk(text)

        if risk.level == guardrails.EMERGENCY_NOW:
            self.channel.send_text(message.sender, guardrails.emergency_message())
            return "emergencia"

        if risk.level == guardrails.REFER_MEDICAL_CARE:
            self.channel.send_text(message.sender, guardrails.medical_referral_message())
            return "encaminhamento"

        # Saudação ou agradecimento não vai ao modelo: gastaria token e cairia
        # na checagem de fundamentação, devolvendo "não encontrei no material".
        if guardrails.is_small_talk(text):
            self.channel.send_text(message.sender, guardrails.small_talk_message())
            return "social"

        history = self.conversations.history(message.sender)
        answer = self.engine.answer(text, history=history)

        if answer.error:
            self.channel.send_text(message.sender, fallback_message())
            return "erro_modelo"

        if not answer.grounded:
            self.channel.send_text(message.sender, guardrails.out_of_scope_message())
            return "fora_de_escopo"

        reply = answer.text
        if risk.needs_safety_note:
            reply += guardrails.safety_note()

        # Só guardamos no histórico o que foi efetivamente entregue.
        self.conversations.append(message.sender, "user", text)
        self.conversations.append(message.sender, "assistant", answer.text)

        self.channel.send_text(message.sender, reply)
        return "respondido_com_nota" if risk.needs_safety_note else "respondido"

    def _forget(self, sender: str) -> None:
        """Apaga tudo que guardamos de um usuário (LGPD, direito de eliminação)."""
        self.conversations.forget(sender)
        self.known_users.forget(sender)
        self.rate_limiter.forget(sender)

    def _resolve_text(self, message: IncomingMessage) -> str | None:
        """Texto da mensagem, transcrevendo áudio quando necessário."""
        if message.kind == "text":
            return (message.text or "").strip() or None

        if message.kind == "audio":
            if not self.settings.enable_audio:
                return None
            audio = self.channel.fetch_media(message)
            if not audio:
                return None
            return self.engine.transcribe(audio, message.media_mime) or None

        return None
