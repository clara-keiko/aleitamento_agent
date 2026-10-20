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
from app.piloto import (
    AUTORIZADO,
    ESGOTADO,
    RegistroDoPiloto,
    custo_da_resposta,
    encontrar_codigo,
)

log = get_logger(__name__)


def variantes_do_numero(numero: str) -> set[str]:
    """Formas equivalentes de um mesmo telefone.

    Celular brasileiro tem um 9 depois do DDD, mas o WhatsApp entrega boa
    parte das contas antigas sem ele: 5521981814170 chega como 552181814170.
    Comparar só a forma literal faria a lista de permitidos recusar
    justamente o número que a pessoa digitou certo.
    """
    digitos = "".join(c for c in numero if c.isdigit())
    if not digitos:
        return set()

    formas = {digitos}
    if digitos.startswith("55"):
        if len(digitos) == 13 and digitos[4] == "9":
            formas.add(digitos[:4] + digitos[5:])
        elif len(digitos) == 12:
            formas.add(digitos[:4] + "9" + digitos[4:])
    return formas


def ler_lista_de_permitidos(bruto: str) -> set[str]:
    permitidos: set[str] = set()
    for item in (bruto or "").split(","):
        permitidos |= variantes_do_numero(item)
    return permitidos


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
        piloto: RegistroDoPiloto | None = None,
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
        self.permitidos = ler_lista_de_permitidos(settings.allowed_senders)
        self.piloto = piloto

    def _sender_allowed(self, sender: str) -> bool:
        # O protótipo web usa id de sessão como remetente, não telefone; a
        # lista não faz sentido lá e bloquearia toda conversa.
        if not self.permitidos or self.channel.name == "web":
            return True
        return bool(variantes_do_numero(sender) & self.permitidos)

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
        # Antes de tudo, inclusive do dedup: remetente fora da lista não deixa
        # rastro nenhum — nem memória, nem contagem, nem resposta.
        if not self._sender_allowed(message.sender):
            return "fora_da_lista"

        if not self.deduplicator.check_and_mark(message.message_id):
            return "duplicada"

        portao = self._portao_do_convite(message)
        if portao:
            return portao

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

        # Limites só aqui, depois da triagem: emergência, encaminhamento e
        # cumprimento não gastam modelo, e nenhum teto pode impedir uma mãe
        # de receber a orientação de ligar para o 192.
        bloqueio = self._checar_limites(message.sender)
        if bloqueio:
            return bloqueio

        history = self.conversations.history(message.sender)
        answer = self.engine.answer(text, history=history)
        self._registrar_uso(message.sender, answer)

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

    # ------------------------------------------------------------------
    # Fase de testes
    # ------------------------------------------------------------------
    def _portao_do_convite(self, message: IncomingMessage) -> str | None:
        """Decide se a mensagem segue. None = segue; texto = desfecho.

        Não vale no protótipo web (o remetente é sessão do navegador, e ele
        já tem código de acesso próprio) nem com MODO_ACESSO=aberto.
        """
        if (
            self.piloto is None
            or self.settings.modo_acesso != "convite"
            or self.channel.name == "web"
        ):
            return None

        sender = message.sender
        texto = (message.text or "").strip() if message.kind == "text" else ""
        codigo = encontrar_codigo(texto, self.piloto.codigos)

        if self.piloto.autorizado(sender) or (
            self.permitidos and variantes_do_numero(sender) & self.permitidos
        ):
            if codigo:
                # Tocou no link de novo. Sem isto, a frase do convite iria
                # ao modelo e voltaria como "não encontrei no material".
                self.channel.send_text(sender, guardrails.pilot_already_member_message())
                return "convite_repetido"
            return None

        if codigo:
            resultado = self.piloto.usar_convite(sender, codigo)
            if resultado == AUTORIZADO:
                # Marca como conhecida para as boas-vindas gerais não se
                # repetirem na primeira pergunta.
                self.known_users.is_first_contact(sender)
                self.channel.send_text(sender, guardrails.welcome_message())
                self.channel.send_text(
                    sender,
                    guardrails.pilot_welcome_message(self.settings.limite_diario_por_pessoa),
                )
                return "convite_aceito"
            if resultado == ESGOTADO:
                self.channel.send_text(sender, guardrails.pilot_invite_full_message())
                return "convite_esgotado"

        # Fora do piloto, mas em emergência: a triagem não gasta modelo, e
        # a orientação de ligar para o 192 não depende de convite.
        if texto and guardrails.classify_risk(texto).level == guardrails.EMERGENCY_NOW:
            self.channel.send_text(sender, guardrails.emergency_message())
            return "emergencia_nao_convidado"

        if self.piloto.deve_avisar_recusa(sender):
            self.channel.send_text(sender, guardrails.pilot_closed_message())
            return "nao_convidado"
        return "nao_convidado_silencio"

    def _checar_limites(self, sender: str) -> str | None:
        if self.piloto is None:
            return None

        orcamento = self.settings.orcamento_mensal_usd
        if orcamento > 0 and self.piloto.gasto_do_mes() >= orcamento:
            self.channel.send_text(sender, guardrails.budget_exhausted_message())
            return "orcamento_esgotado"

        limite = self.settings.limite_diario_por_pessoa
        if (
            limite > 0
            and self.channel.name != "web"
            and self.piloto.perguntas_hoje(sender) >= limite
        ):
            self.channel.send_text(sender, guardrails.daily_limit_message(limite))
            return "limite_diario"
        return None

    def _registrar_uso(self, sender: str, answer) -> None:
        """Toda chamada ao modelo conta, inclusive a que não foi entregue.

        Resposta recusada por falta de fundamentação também gastou token.
        """
        if self.piloto is None:
            return
        custo = custo_da_resposta(
            answer.input_tokens,
            answer.output_tokens,
            self.settings.preco_entrada_usd_1m,
            self.settings.preco_saida_usd_1m,
        )
        # Sem token, a chamada falhou antes de buscar na base.
        if answer.input_tokens:
            custo += self.settings.custo_busca_usd
        self.piloto.registrar_pergunta(None if self.channel.name == "web" else sender, custo)

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
