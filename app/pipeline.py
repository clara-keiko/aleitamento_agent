"""Orquestração de uma mensagem recebida até a resposta enviada.

Ordem deliberada: deduplicação e rate limit primeiro (baratos), triagem
clínica antes do modelo (uma emergência nunca deve depender de a OpenAI estar
no ar) e checagem de fundamentação depois.
"""

from app import guardrails
from app.channels.base import Channel, IncomingMessage
from app.config import Settings
from app.llm import AssistantEngine, fallback_message
from app.logging_utils import get_logger, pseudonymize
from app.memory import ConversationStore, MessageDeduplicator, RateLimiter

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

    def handle(self, message: IncomingMessage) -> None:
        user = pseudonymize(message.sender)

        if not self.deduplicator.check_and_mark(message.message_id):
            log.info("mensagem repetida ignorada (%s)", user)
            return

        if not self.rate_limiter.allow(message.sender):
            log.warning("rate limit atingido (%s)", user)
            self.channel.send_text(message.sender, guardrails.rate_limited_message())
            return

        text = self._resolve_text(message)
        if text is None:
            self.channel.send_text(message.sender, guardrails.unsupported_media_message())
            return

        if guardrails.is_opt_out(text):
            self.conversations.forget(message.sender)
            self.channel.send_text(message.sender, guardrails.opt_out_message())
            log.info("opt-out processado (%s)", user)
            return

        # Primeiro contato: escopo, aviso de automação e canal de emergência.
        if self.conversations.mark_greeted(message.sender):
            self.channel.send_text(message.sender, guardrails.welcome_message())

        risk = guardrails.classify_risk(text)
        log.info("mensagem processada (%s) risco=%s", user, risk.level)

        if risk.level == guardrails.EMERGENCY_NOW:
            self.channel.send_text(message.sender, guardrails.emergency_message())
            return

        if risk.level == guardrails.REFER_MEDICAL_CARE:
            self.channel.send_text(message.sender, guardrails.medical_referral_message())
            return

        history = self.conversations.history(message.sender)
        answer = self.engine.answer(text, history=history)

        if answer.error:
            self.channel.send_text(message.sender, fallback_message())
            return

        if not answer.grounded:
            log.info("resposta sem citação da base; devolvendo fora de escopo (%s)", user)
            self.channel.send_text(message.sender, guardrails.out_of_scope_message())
            return

        reply = answer.text
        if risk.needs_safety_note:
            reply += guardrails.safety_note()

        # Só guardamos no histórico o que foi efetivamente entregue.
        self.conversations.append(message.sender, "user", text)
        self.conversations.append(message.sender, "assistant", answer.text)

        self.channel.send_text(message.sender, reply)

    def _resolve_text(self, message: IncomingMessage) -> str | None:
        """Texto da mensagem, transcrevendo áudio quando necessário."""
        if message.kind == "text":
            text = (message.text or "").strip()
            return text or None

        if message.kind == "audio":
            if not self.settings.enable_audio:
                return None
            audio = self.channel.fetch_media(message)
            if not audio:
                return None
            transcript = self.engine.transcribe(audio, message.media_mime)
            return transcript or None

        return None
