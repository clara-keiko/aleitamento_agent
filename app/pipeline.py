"""Orquestração de uma mensagem recebida até a resposta enviada.

Ordem deliberada: deduplicação e rate limit primeiro (baratos), triagem
clínica antes do modelo (uma emergência nunca deve depender de a OpenAI estar
no ar) e checagem de fundamentação depois.
"""

import time

from app import caderneta, guardrails, planos
from app.channels.base import Channel, IncomingMessage
from app.config import Settings
from app.lembretes import LEMBRETES, VERSAO_LEMBRETES
from app.llm import AssistantEngine, fallback_message
from app.logging_utils import get_logger, pseudonymize
from app.memoria_persistente import MemoriaPersistente
from app.memory import ConversationStore, KnownUsers, MessageDeduplicator, RateLimiter
from app.piloto import (
    AUTORIZADO,
    ESGOTADO,
    RegistroDoPiloto,
    custo_da_resposta,
    encontrar_codigo,
)
from app.questionario import (
    Questionario,
    descrever_atualizacao,
    descrever_perfil,
    ler_informacao_avulsa,
)

log = get_logger(__name__)

# Por quanto tempo um pedido de exclusão espera o CONFIRMAR.
CONFIRMACAO_VALE_S = 15 * 60


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
        memoria: MemoriaPersistente | None = None,
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
        self.memoria = memoria
        # Quem já recebeu a pergunta de memória neste processo e ainda não
        # respondeu. Evita perguntar a cada mensagem; depois de um reinício,
        # pergunta de novo uma vez, o que é aceitável.
        self._perguntados: set[str] = set()
        self.questionario = Questionario(memoria) if memoria is not None else None
        # Leitura da caderneta esperando "1" para ser gravada:
        # número -> (bebê, vacinas lidas, quando).
        self._cadernetas_pendentes: dict[str, tuple] = {}
        # Situação do plano medida antes da pergunta, para o aviso de
        # "restam N" no fim da resposta.
        self._situacoes: dict = {}
        # Quem recebeu a pergunta de lembretes e ainda não respondeu.
        self._lembretes_pendentes: set[str] = set()
        # Dado solto esperando "1": número -> (tipo, valor, bebê, quando).
        self._atualizacoes_pendentes: dict[str, tuple] = {}
        # Pedidos de exclusão esperando CONFIRMAR: número -> quando pediu.
        self._exclusoes_pendentes: dict[str, float] = {}

    @property
    def _lembretes_ligados(self) -> bool:
        return self._memoria_ligada and self.settings.lembretes_ativos

    @property
    def _memoria_ligada(self) -> bool:
        return (
            self.memoria is not None
            and self.settings.memoria_ativa
            and self.channel.name != "web"
        )

    def _na_memoria(self, metodo: str, *args, padrao=None):
        """Banco fora do ar nunca pode impedir uma resposta."""
        try:
            return getattr(self.memoria, metodo)(*args)
        except Exception as exc:  # noqa: BLE001
            log.error("memória persistente indisponível (%s): %s", metodo, exc)
            return padrao

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

        if message.kind == "image":
            return self._foto(message)

        text = self._resolve_text(message)
        if text is None:
            self.channel.send_text(message.sender, guardrails.unsupported_media_message())
            return "nao_suportado"

        direito = self._direitos_da_titular(message, text)
        if direito:
            return direito

        # True/False pela decisão de memória; None se não perguntado ou
        # se o banco estiver fora (aí segue com a memória curta).
        consentiu = self._na_memoria("decisao", message.sender) if self._memoria_ligada else None

        # Triagem clínica antes de tudo: "oi, meu bebê não respira" é
        # emergência, não saudação.
        risk = guardrails.classify_risk(text)

        # Primeiro contato: escopo, aviso de automação, canal de emergência e
        # a pergunta de guarda. Em emergência a pergunta fica para depois:
        # nada pode competir com o 192.
        if self.known_users.is_first_contact(message.sender) and not self._ja_conhece(
            message.sender, consentiu
        ):
            pergunta = (
                "" if risk.level == guardrails.EMERGENCY_NOW
                else self._pergunta_de_memoria(message.sender)
            )
            self.channel.send_text(message.sender, guardrails.welcome_message() + pergunta)

        if risk.level == guardrails.EMERGENCY_NOW:
            self.channel.send_text(message.sender, guardrails.emergency_message())
            return "emergencia"

        if risk.level == guardrails.REFER_MEDICAL_CARE:
            self.channel.send_text(message.sender, guardrails.medical_referral_message())
            return "encaminhamento"

        # Questionário inicial em andamento: a resposta curta é dele. Uma
        # dúvida no meio (texto longo, "?") passa e é respondida normalmente.
        if self.questionario and self.questionario.ativo(message.sender):
            seguinte = self._na_memoria_questionario(message.sender, text)
            if seguinte:
                if not self.questionario.ativo(message.sender):
                    seguinte += self._pergunta_de_lembretes(message.sender)
                self.channel.send_text(message.sender, seguinte)
                return "questionario"

        # "Ela pesou 5,2 kg", "tenho 33 anos": dado de perfil dito de
        # passagem. O modelo não grava nada (e chegou a fingir que gravou);
        # aqui o serviço oferece atualizar, com confirmação.
        if consentiu:
            oferta = self._oferecer_atualizacao(message.sender, text)
            if oferta:
                return oferta

        # Saudação ou agradecimento não vai ao modelo: gastaria token e cairia
        # na checagem de fundamentação, devolvendo "não encontrei no material".
        if guardrails.is_small_talk(text):
            self.channel.send_text(
                message.sender,
                guardrails.small_talk_message() + self._pergunta_de_memoria(message.sender),
            )
            return "social"

        # Limites só aqui, depois da triagem: emergência, encaminhamento e
        # cumprimento não gastam modelo, e nenhum teto pode impedir uma mãe
        # de receber a orientação de ligar para o 192.
        bloqueio = self._checar_limites(message.sender)
        if bloqueio:
            return bloqueio

        history = None
        if consentiu:
            history = self._na_memoria("historico", message.sender, self.settings.history_turns)
        if history is None:
            history = self.conversations.history(message.sender)
        if consentiu:
            perfil = descrever_perfil(
                self._na_memoria("perfil", message.sender), self.questionario._hoje()
            )
            if perfil:
                history = [{
                    "role": "system",
                    "content": "O que esta pessoa já contou (use para adaptar a "
                    "resposta à idade e à situação do bebê; não invente o que "
                    "não está aqui):\n" + perfil,
                }] + history
        sobre_vacina = guardrails.is_vaccine_question(text)
        if sobre_vacina:
            answer = self.engine.answer_vacina(text, history=history)
        else:
            answer = self.engine.answer(text, history=history)
        self._registrar_uso(message.sender, answer, "vacina" if sobre_vacina else "pergunta")

        if answer.error:
            self.channel.send_text(message.sender, fallback_message())
            return "erro_modelo"

        # Remédio na amamentação: a base pode não ter o medicamento, e o
        # e-Lactancia tem quase todos. Vai junto até da resposta "não achei".
        sobre_remedio = guardrails.is_medication_question(text)
        nota_remedio = guardrails.elactancia_note() if sobre_remedio else ""

        if not answer.grounded:
            if sobre_vacina:
                self.channel.send_text(message.sender, guardrails.vaccine_fallback_message())
                return "vacina_sem_fonte"
            self.channel.send_text(
                message.sender, guardrails.out_of_scope_message() + nota_remedio
            )
            return "fora_de_escopo"

        reply = answer.text
        if risk.needs_safety_note:
            reply += guardrails.safety_note()
        # Vacina já traz os links do gov.br e da SBIm no próprio texto.
        if not sobre_vacina:
            reply += guardrails.sources_footer(getattr(answer, "fontes", []))
        reply += nota_remedio
        if sobre_vacina and consentiu:
            reply += self._sugestao_de_foto(message.sender)
        situacao = self._situacoes.pop(message.sender, None)
        if situacao is not None:
            reply += planos.aviso_de_restantes(situacao)

        desfecho = "respondido_com_nota" if risk.needs_safety_note else "respondido"

        # Só guardamos no histórico o que foi efetivamente entregue.
        self.conversations.append(message.sender, "user", text)
        self.conversations.append(message.sender, "assistant", answer.text)
        if consentiu:
            self._na_memoria(
                "registrar_troca", message.sender, text, answer.text,
                "audio" if message.kind == "audio" else "texto", desfecho,
            )

        self.channel.send_text(message.sender, reply + self._pergunta_de_memoria(message.sender))
        return desfecho

    # ------------------------------------------------------------------
    # Memória persistente e direitos da titular
    # ------------------------------------------------------------------
    def _ja_conhece(self, sender: str, consentiu: bool | None) -> bool:
        """Quem já passou pelas boas-vindas antes de um reinício.

        A lista de conhecidos vive na memória do processo e zera a cada
        deploy; sem isto, toda testadora recebia a apresentação de novo no
        meio da conversa. Decisão de guarda (sim ou não) e convite aceito
        ficam no banco: são prova de que ela já foi apresentada.
        """
        if consentiu is not None:
            return True
        return bool(self.piloto and self.piloto.autorizado(sender))

    def _lembretes_da_titular(self, message: IncomingMessage, text: str) -> str | None:
        if not self._lembretes_ligados:
            return None
        sender = message.sender

        if guardrails.is_reminder_stop(text):
            self._lembretes_pendentes.discard(sender)
            self._na_memoria(
                "decidir", sender, False, message.message_id, LEMBRETES, VERSAO_LEMBRETES
            )
            self._na_memoria("apagar_telefone", sender)
            self.channel.send_text(sender, guardrails.reminders_stopped_message())
            return "lembretes_parados"

        if sender in self._lembretes_pendentes:
            resposta = guardrails.consent_answer(text)
            if resposta is not None:
                self._lembretes_pendentes.discard(sender)
                self._na_memoria(
                    "decidir", sender, resposta, message.message_id, LEMBRETES, VERSAO_LEMBRETES
                )
                if resposta:
                    self._na_memoria("salvar_telefone_para_lembretes", sender)
                    self.channel.send_text(sender, guardrails.reminders_yes_message())
                    return "lembretes_aceitos"
                self.channel.send_text(sender, guardrails.reminders_no_message())
                return "lembretes_recusados"

        if guardrails.is_reminder_start(text):
            if not self._na_memoria("decisao", sender):
                self.channel.send_text(sender, guardrails.reminders_need_consent_message())
                return "lembretes_sem_autorizacao"
            if not (self._na_memoria("perfil", sender) or {}).get("bebes"):
                self.channel.send_text(sender, guardrails.reminders_need_profile_message())
                return "lembretes_sem_bebe"
            self._lembretes_pendentes.add(sender)
            self.channel.send_text(sender, guardrails.reminders_question())
            return "pergunta_de_lembretes"
        return None

    def _pergunta_de_lembretes(self, sender: str) -> str:
        """Depois do questionário com bebê: oferecer os lembretes, uma vez."""
        if not self._lembretes_ligados or sender in self._lembretes_pendentes:
            return ""
        if not (self._na_memoria("perfil", sender) or {}).get("bebes"):
            return ""
        if self._na_memoria("decisao", sender, LEMBRETES, padrao=False) is not None:
            return ""
        self._lembretes_pendentes.add(sender)
        return "\n\n" + guardrails.reminders_question()

    def _pergunta_de_memoria(self, sender: str) -> str:
        """A pergunta de consentimento, para ir no fim de uma mensagem.

        Vazia se a memória está desligada, se já houve decisão ou se ela já
        foi feita a esta pessoa neste processo.
        """
        if not self._memoria_ligada or sender in self._perguntados:
            return ""
        # Banco fora: sem decisão conhecida, melhor não perguntar agora.
        if self._na_memoria("decisao", sender, padrao=False) is not None:
            return ""
        self._perguntados.add(sender)
        return "\n\n" + guardrails.memory_consent_question(self.settings.url_politica)

    def _direitos_da_titular(self, message: IncomingMessage, text: str) -> str | None:
        sender = message.sender

        # Resposta à conferência da caderneta: nada é gravado sem o "1".
        pendente = self._cadernetas_pendentes.get(sender)
        if pendente and time.monotonic() - pendente[2] < CONFIRMACAO_VALE_S:
            resposta = guardrails.consent_answer(text)
            if resposta is not None:
                self._cadernetas_pendentes.pop(sender, None)
                if not resposta:
                    self.channel.send_text(sender, caderneta.MENSAGEM_NAO_ANOTADO)
                    return "vacinas_recusadas"
                bebe_id, vacinas, _ = pendente
                anotadas = self._na_memoria(
                    "registrar_vacinas", bebe_id, [(v.rotulo, v.data) for v in vacinas]
                )
                if anotadas is None:
                    self.channel.send_text(sender, fallback_message())
                    return "erro_memoria"
                self.channel.send_text(sender, caderneta.mensagem_anotadas(*anotadas))
                return "vacinas_anotadas"

        # Resposta à confirmação de um dado dito de passagem.
        atualizacao = self._atualizacoes_pendentes.get(sender)
        if atualizacao and time.monotonic() - atualizacao[3] < CONFIRMACAO_VALE_S:
            resposta = guardrails.consent_answer(text)
            if resposta is not None:
                self._atualizacoes_pendentes.pop(sender, None)
                if not resposta:
                    self.channel.send_text(sender, "Combinado, não alterei nada.")
                    return "atualizacao_recusada"
                tipo, valor, bebe_id, _ = atualizacao
                hoje = self.questionario._hoje()
                if tipo == "idade_mae":
                    feito = self._na_memoria(
                        "salvar_ano_nascimento_mae", sender, hoje.year - int(valor), padrao=False
                    )
                else:
                    feito = self._na_memoria(
                        "registrar_do_bebe", bebe_id, tipo, hoje, valor, None, "conversa",
                        padrao=False,
                    )
                if feito is False:
                    self.channel.send_text(sender, fallback_message())
                    return "erro_memoria"
                self.channel.send_text(sender, "Atualizado! ✅ Para ver tudo, *MEUS DADOS*.")
                return "perfil_atualizado"

        lembrete = self._lembretes_da_titular(message, text)
        if lembrete:
            return lembrete

        # "parar" no meio do questionário é parar o questionário, não sair do
        # serviço: precisa ser visto antes do SAIR/PARAR geral.
        if (
            self.questionario
            and self.questionario.ativo(sender)
            and self.questionario.e_parada(text)
        ):
            self.channel.send_text(sender, self.questionario.responder(sender, text))
            return "questionario_parado"

        # Apagar é irreversível: com dados guardados, só depois de CONFIRMAR.
        pediu_antes = time.monotonic() - self._exclusoes_pendentes.get(sender, -1e9)
        if pediu_antes < CONFIRMACAO_VALE_S and (
            guardrails.is_confirmation(text) or guardrails.is_delete_command(text)
        ):
            self._exclusoes_pendentes.pop(sender, None)
            self._na_memoria("apagar_tudo", sender)
            self._forget(sender)
            self.channel.send_text(sender, guardrails.deleted_message())
            return "apagou_tudo"

        apagar = guardrails.is_delete_command(text)
        if apagar or guardrails.is_opt_out(text):
            # Há o que perder se há conversa guardada ou autorização ativa:
            # quem só disse "pode guardar" também precisa confirmar, e a
            # exclusão tem de revogar essa autorização.
            if self._memoria_ligada and (
                self._na_memoria("tem_dados", sender, padrao=False)
                or self._na_memoria("decisao", sender)
            ):
                self._exclusoes_pendentes[sender] = time.monotonic()
                self.channel.send_text(sender, guardrails.delete_confirmation_request())
                return "pedido_de_exclusao"
            # Nada guardado nem autorizado: só a memória curta, que se perde de
            # qualquer jeito. Ainda assim passa pelo banco, que é idempotente:
            # se a leitura acima falhou, a exclusão não pode ficar pela metade.
            if self._memoria_ligada:
                self._na_memoria("apagar_tudo", sender)
            self._forget(sender)
            self.channel.send_text(
                sender, guardrails.deleted_message() if apagar else guardrails.opt_out_message()
            )
            return "apagou_tudo" if apagar else "opt_out"

        if guardrails.is_plan_command(text):
            situacao = self._situacao_do_plano(sender)
            if situacao is not None:
                self.channel.send_text(sender, planos.mensagem_meu_plano(situacao))
                return "meu_plano"

        if guardrails.is_my_data_command(text):
            dados = self._na_memoria("meus_dados", sender, padrao={}) if self._memoria_ligada else {}
            if dados:
                perfil = self._na_memoria("perfil", sender)
                dados["perfil_texto"] = descrever_perfil(perfil, self.questionario._hoje())
            self.channel.send_text(sender, guardrails.my_data_message(dados or {}))
            return "meus_dados"

        if self._memoria_ligada and guardrails.is_profile_command(text):
            if self._na_memoria("decisao", sender):
                self.channel.send_text(sender, self.questionario.iniciar(sender))
                return "questionario"
            # Sem autorização não há onde guardar: primeiro a pergunta.
            self._perguntados.add(sender)
            self.channel.send_text(
                sender, guardrails.memory_consent_question(self.settings.url_politica)
            )
            return "pergunta_de_memoria"

        if self._memoria_ligada and guardrails.is_store_command(text):
            if self._na_memoria("decisao", sender):
                self.channel.send_text(sender, guardrails.memory_already_stored_message())
                return "memoria_ja_aceita"
            # Aceite só vale respondendo à pergunta inteira: o que fica
            # guardado, por quanto tempo, como apagar e o link da política.
            self._perguntados.add(sender)
            self.channel.send_text(
                sender, guardrails.memory_consent_question(self.settings.url_politica)
            )
            return "pergunta_de_memoria"

        if self._memoria_ligada:
            resposta = guardrails.consent_answer(text)
            if resposta is not None and self._e_resposta_de_memoria(sender, text):
                self._na_memoria("decidir", sender, resposta, message.message_id)
                self._perguntados.discard(sender)
                if resposta:
                    self.channel.send_text(
                        sender,
                        guardrails.memory_consent_yes_message()
                        + "\n\n" + self.questionario.iniciar(sender),
                    )
                    return "memoria_aceita"
                self.channel.send_text(sender, guardrails.memory_consent_no_message())
                return "memoria_recusada"
        return None

    def _e_resposta_de_memoria(self, sender: str, text: str) -> bool:
        """"sim" só vale como consentimento se a pergunta estiver no ar.

        Depois de um reinício a lista de perguntados zera; "1" ou "2" de quem
        nunca decidiu ainda contam, porque não significam outra coisa aqui.
        """
        if sender in self._perguntados:
            return True
        return text.strip() in {"1", "2"} and self._na_memoria("decisao", sender, padrao=False) is None

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
                # Um balão só: cada mensagem a mais conta para o WhatsApp.
                self.channel.send_text(
                    sender,
                    guardrails.welcome_message()
                    + "\n\n"
                    + guardrails.pilot_welcome_message(self.settings.limite_diario_por_pessoa)
                    + self._pergunta_de_memoria(sender),
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

        # Um aviso por mês, não por dia: mensagem a quem não é testadora é
        # justamente o tipo de envio que mais pesa contra o número.
        if self.piloto.deve_avisar_recusa(sender, intervalo_horas=24 * 30):
            self.channel.send_text(sender, guardrails.pilot_closed_message())
            return "nao_convidado"
        return "nao_convidado_silencio"

    def boas_vindas_tardias(self, numero: str) -> None:
        """Mensagens de entrada para quem foi liberado pelo painel.

        A autorização já foi feita; aqui é só avisar a pessoa. Marca todas as
        formas do número como conhecidas: a resposta dela pode chegar sem o 9.
        """
        for forma in variantes_do_numero(numero):
            self.known_users.is_first_contact(forma)
        self.channel.send_text(
            numero,
            guardrails.pilot_late_welcome_message(self.settings.limite_diario_por_pessoa)
            + "\n\n"
            + guardrails.welcome_message()
            + self._pergunta_de_memoria(numero),
        )

    def _situacao_do_plano(self, sender: str):
        """Plano e uso no ciclo; None sem planos, no web ou com banco fora
        (banco fora não bloqueia ninguém: o teto global de gasto segue)."""
        if not self.settings.planos_ativos or self.memoria is None or self.channel.name == "web":
            return None
        try:
            return planos.situacao(self.memoria, self.settings, sender, self.questionario._agora())
        except Exception as exc:  # noqa: BLE001
            log.error("planos: não consegui consultar o uso: %s", exc)
            return None

    def _checar_limites(self, sender: str) -> str | None:
        if self.piloto is not None:
            orcamento = self.settings.orcamento_mensal_usd
            if orcamento > 0 and self.piloto.gasto_do_mes() >= orcamento:
                self.channel.send_text(sender, guardrails.budget_exhausted_message())
                return "orcamento_esgotado"

        situacao = self._situacao_do_plano(sender)
        if situacao is not None:
            if situacao.usados_dia >= situacao.plano.diario:
                self.channel.send_text(sender, planos.mensagem_limite_diario(situacao))
                return "limite_diario"
            if situacao.usados_mes >= situacao.plano.mensal:
                self.channel.send_text(sender, planos.mensagem_limite_mensal(situacao, self.settings))
                return "limite_mensal"
            self._situacoes[sender] = situacao
            return None

        if self.piloto is None:
            return None

        limite = self.settings.limite_diario_por_pessoa
        if (
            limite > 0
            and self.channel.name != "web"
            and self.piloto.perguntas_hoje(sender) >= limite
        ):
            self.channel.send_text(sender, guardrails.daily_limit_message(limite))
            return "limite_diario"
        return None

    def _registrar_uso(self, sender: str, answer, tipo: str = "pergunta") -> None:
        """Toda chamada ao modelo conta, inclusive a que não foi entregue.

        Resposta recusada por falta de fundamentação também gastou token.
        """
        custo = custo_da_resposta(
            answer.input_tokens,
            answer.output_tokens,
            self.settings.preco_entrada_usd_1m,
            self.settings.preco_saida_usd_1m,
        )
        buscas_web = getattr(answer, "buscas_web", 0)
        if buscas_web:
            custo += buscas_web * self.settings.custo_busca_web_usd
        # Sem token, a chamada falhou antes de buscar na base.
        elif answer.input_tokens:
            custo += self.settings.custo_busca_usd
        numero = None if self.channel.name == "web" else sender
        if self.piloto is not None:
            self.piloto.registrar_pergunta(numero, custo)
        if self.memoria is not None:
            # Custo real por chamada, para calibrar os limites por plano.
            self._na_memoria(
                "registrar_uso", numero, tipo, custo,
                answer.input_tokens, answer.output_tokens, buscas_web,
            )

    def _na_memoria_questionario(self, sender: str, text: str) -> str | None:
        """Resposta ao questionário; banco fora encerra sem travar a conversa."""
        try:
            return self.questionario.responder(sender, text)
        except Exception as exc:  # noqa: BLE001
            log.error("questionário interrompido: %s", exc)
            self.questionario.encerrar(sender)
            return None

    def _sugestao_de_foto(self, sender: str) -> str:
        """Depois de uma resposta sobre vacina: a foto da caderneta resolve
        o "quais ele já tomou?". Só para quem guarda e ainda não mandou."""
        bebes = (self._na_memoria("perfil", sender) or {}).get("bebes") or []
        if not bebes:
            return (
                "\n\n📷 Quer que eu anote as vacinas que o bebê já tomou? Mande "
                "*PERFIL* (para eu saber a data de nascimento) e depois uma foto "
                "da página de vacinas da caderneta."
            )
        if bebes[0].get("vacinas"):
            return ""
        return (
            "\n\n📷 Dica: mande uma *foto da página de vacinas da caderneta* que "
            "eu anoto as que o bebê já tomou."
        )

    def _oferecer_atualizacao(self, sender: str, text: str) -> str | None:
        informacao = ler_informacao_avulsa(text)
        if not informacao:
            return None
        tipo, valor = informacao
        perfil = self._na_memoria("perfil", sender) or {}
        bebe_id = None
        if tipo == "idade_mae":
            # "Tenho 33 anos" só é a idade da mãe se quem fala é a mãe.
            if perfil.get("relacao") != "mae":
                return None
        else:
            if perfil.get("relacao") not in ("mae", "pai", "responsavel_legal"):
                return None
            bebes = perfil.get("bebes") or []
            if not bebes:
                return None
            bebe_id = bebes[0]["id"]
        self._atualizacoes_pendentes[sender] = (tipo, valor, bebe_id, time.monotonic())
        descricao = descrever_atualizacao(tipo, valor, self.questionario._hoje())
        self.channel.send_text(
            sender,
            f"Quer que eu atualize o seu perfil?\n• {descricao}\n\n1 – Sim\n2 – Não",
        )
        return "oferta_de_atualizacao"

    def _foto(self, message: IncomingMessage) -> str:
        """Foto: só a página de vacinas da caderneta, e só para quem guarda."""
        sender = message.sender
        legenda = (message.text or "").strip()
        if legenda and guardrails.classify_risk(legenda).level == guardrails.EMERGENCY_NOW:
            self.channel.send_text(sender, guardrails.emergency_message())
            return "emergencia"

        if not self._memoria_ligada:
            self.channel.send_text(sender, guardrails.unsupported_media_message())
            return "nao_suportado"
        if not self._na_memoria("decisao", sender):
            self.channel.send_text(sender, caderneta.MENSAGEM_SEM_AUTORIZACAO)
            return "foto_sem_autorizacao"
        bebes = (self._na_memoria("perfil", sender) or {}).get("bebes") or []
        if not bebes:
            self.channel.send_text(sender, caderneta.MENSAGEM_SEM_BEBE)
            return "foto_sem_bebe"

        bloqueio = self._checar_limites(sender)
        if bloqueio:
            return bloqueio

        imagem = self.channel.fetch_media(message)
        if not imagem:
            self.channel.send_text(sender, caderneta.MENSAGEM_FOTO_FALHOU)
            return "foto_falhou"

        # A foto só existe nesta função: é lida e descartada aqui.
        leitura = self.engine.ler_caderneta(imagem, message.media_mime)
        del imagem
        self._registrar_uso(sender, leitura, "foto_caderneta")
        if leitura.erro:
            self.channel.send_text(sender, fallback_message())
            return "erro_modelo"
        if not leitura.e_caderneta:
            self.channel.send_text(sender, caderneta.MENSAGEM_NAO_E_CADERNETA)
            return "foto_nao_caderneta"

        bebe = bebes[0]  # o mais novo
        validas, duvidosas = caderneta.separar(
            leitura, bebe["nascimento"], self.questionario._hoje()
        )
        if not validas:
            self.channel.send_text(sender, caderneta.MENSAGEM_ILEGIVEL)
            return "foto_ilegivel"

        self._cadernetas_pendentes[sender] = (bebe["id"], validas, time.monotonic())
        self.channel.send_text(sender, caderneta.mensagem_para_conferir(validas, duvidosas))
        return "foto_caderneta_lida"

    def _forget(self, sender: str) -> None:
        """Apaga tudo que guardamos de um usuário (LGPD, direito de eliminação)."""
        self.conversations.forget(sender)
        self.known_users.forget(sender)
        self.rate_limiter.forget(sender)
        if self.questionario:
            self.questionario.encerrar(sender)
        self._cadernetas_pendentes.pop(sender, None)
        self._atualizacoes_pendentes.pop(sender, None)
        self._lembretes_pendentes.discard(sender)

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
