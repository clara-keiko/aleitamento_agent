"""Cobrança do LactAI Plus pelo Asaas.

Fluxo:
1. A pessoa manda ASSINAR. O LactAI cria (uma vez) um link de pagamento
   recorrente mensal só dela e manda no WhatsApp. No Asaas ela escolhe Pix,
   cartão ou boleto e preenche nome e CPF — o LactAI nunca vê o CPF.
2. Cada pagamento confirmado chega como aviso (webhook) com o id do link.
   O LactAI acha a pessoa pelo link e estende o Plus por um mês (mais 3 dias
   de tolerância), uma vez por pagamento.
3. CANCELAR ASSINATURA cancela a recorrência no Asaas; o que já foi pago
   continua valendo até o fim.

Docs: https://docs.asaas.com (links de pagamento, assinaturas, webhooks).
"""

import calendar
import datetime as dt
import hmac

import requests

from app.logging_utils import get_logger

log = get_logger(__name__)

BASES = {
    "sandbox": "https://api-sandbox.asaas.com/v3",
    "producao": "https://api.asaas.com/v3",
}
EVENTOS_DE_PAGAMENTO = {"PAYMENT_CONFIRMED", "PAYMENT_RECEIVED"}
TOLERANCIA = dt.timedelta(days=3)
TIMEOUT = 20


def mais_um_mes(dia: dt.date) -> dt.date:
    ano, mes = (dia.year + 1, 1) if dia.month == 12 else (dia.year, dia.month + 1)
    return dt.date(ano, mes, min(dia.day, calendar.monthrange(ano, mes)[1]))


class ErroDoAsaas(RuntimeError):
    pass


class Asaas:
    def __init__(self, settings, post=requests.post, get=requests.get, delete=requests.delete):
        self.settings = settings
        self.base = BASES.get(settings.asaas_ambiente, BASES["sandbox"])
        self._post, self._get, self._delete = post, get, delete

    @property
    def _cabecalhos(self) -> dict:
        # O Asaas exige User-Agent nas contas novas.
        return {
            "access_token": self.settings.asaas_api_key,
            "Content-Type": "application/json",
            "User-Agent": "LactAI",
        }

    def _verificar(self, resposta, acao: str) -> dict:
        if resposta.status_code >= 400:
            # O corpo é o erro do Asaas, não dado de usuário.
            raise ErroDoAsaas(f"{acao}: HTTP {resposta.status_code} {resposta.text[:300]}")
        return resposta.json() if resposta.text else {}

    def criar_link_recorrente(self) -> dict:
        corpo = {
            "name": "LactAI Plus",
            "description": (
                f"{self.settings.limite_mensal_assinante} perguntas por mês no LactAI, "
                "assistente de amamentação no WhatsApp. Renovação mensal; cancele quando quiser."
            ),
            "value": self.settings.valor_assinatura,
            "billingType": "UNDEFINED",  # a pessoa escolhe Pix, cartão ou boleto
            "chargeType": "RECURRENT",
            "subscriptionCycle": "MONTHLY",
            "dueDateLimitDays": 3,
        }
        resposta = self._post(f"{self.base}/paymentLinks", json=corpo,
                              headers=self._cabecalhos, timeout=TIMEOUT)
        return self._verificar(resposta, "criar link")

    def cancelar_assinatura(self, assinatura_id: str) -> None:
        resposta = self._delete(f"{self.base}/subscriptions/{assinatura_id}",
                                headers=self._cabecalhos, timeout=TIMEOUT)
        self._verificar(resposta, "cancelar assinatura")

    def garantir_webhook(self, url: str, token: str, email: str) -> str:
        """Registra o aviso de pagamentos no Asaas, se ainda não houver um
        para esta URL. Assim não há passo manual no painel do Asaas."""
        resposta = self._get(f"{self.base}/webhooks", headers=self._cabecalhos, timeout=TIMEOUT)
        existentes = self._verificar(resposta, "listar webhooks").get("data", [])
        if any(w.get("url") == url for w in existentes):
            return "ja_existia"
        corpo = {
            "name": "LactAI",
            "url": url,
            "email": email,
            "enabled": True,
            "interrupted": False,
            "authToken": token,
            "sendType": "SEQUENTIALLY",
            "events": sorted(EVENTOS_DE_PAGAMENTO | {"PAYMENT_REFUNDED"}),
        }
        resposta = self._post(f"{self.base}/webhooks", json=corpo,
                              headers=self._cabecalhos, timeout=TIMEOUT)
        self._verificar(resposta, "criar webhook")
        return "criado"


def token_valido(recebido: str | None, esperado: str) -> bool:
    return bool(esperado) and bool(recebido) and hmac.compare_digest(recebido, esperado)


# ----------------------------------------------------------------------
# Mensagens
# ----------------------------------------------------------------------

def mensagem_link(url: str, settings) -> str:
    return (
        f"💛 *LactAI Plus* — {settings.preco_assinatura}/mês\n"
        f"• {settings.limite_mensal_assinante} perguntas por mês\n"
        "• Modo família: até 3 pessoas dividem o plano e o histórico do bebê\n\n"
        f"Para assinar, toque no link e escolha Pix, cartão ou boleto:\n{url}\n\n"
        "Assim que o pagamento for confirmado, eu te aviso aqui. Para cancelar "
        "quando quiser: *CANCELAR ASSINATURA*."
    )


def mensagem_ja_assinante(pago_ate: dt.date) -> str:
    return (
        f"Você já é *LactAI Plus*, com o plano em dia até {pago_ate:%d/%m/%Y}. 💛 "
        "Para ver o uso do mês, *MEU PLANO*."
    )


def mensagem_ativado(pago_ate: dt.date, settings) -> str:
    return (
        f"Pagamento confirmado! ✅ Seu *LactAI Plus* está ativo: "
        f"{settings.limite_mensal_assinante} perguntas por mês. Próxima renovação "
        f"até {pago_ate:%d/%m/%Y}. Obrigada! 💛"
    )


MENSAGEM_INDISPONIVEL = (
    "Não consegui gerar o link de pagamento agora. 🙈 Tente de novo em alguns "
    "minutos ou escreva para clara@watanabeaitech.com."
)
MENSAGEM_SEM_ASSINATURA = "Você não tem assinatura ativa para cancelar."


def mensagem_cancelada(pago_ate: dt.date | None) -> str:
    vale = f" O que já foi pago vale até {pago_ate:%d/%m/%Y}." if pago_ate else ""
    return (
        "Pronto, cancelei a renovação da sua assinatura." + vale
        + " Depois disso, você volta ao plano gratuito, sem perder nada do que "
        "está guardado. 💛"
    )


# ----------------------------------------------------------------------
# Serviço: liga o Asaas à memória
# ----------------------------------------------------------------------

def _data(texto) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(texto)[:10]) if texto else None
    except ValueError:
        return None


class Cobranca:
    def __init__(self, settings, memoria, asaas: Asaas | None = None):
        self.settings = settings
        self.memoria = memoria
        self.asaas = asaas or Asaas(settings)

    @property
    def ligada(self) -> bool:
        return bool(self.settings.asaas_api_key) and self.memoria is not None

    def assinar(self, numero: str, hoje: dt.date) -> str:
        atual = self.memoria.assinatura(numero)
        if atual and atual["plano"] == "plus" and atual["pago_ate"] >= hoje:
            return mensagem_ja_assinante(atual["pago_ate"])
        url = self.memoria.link_de_pagamento(numero)
        if not url:
            try:
                link = self.asaas.criar_link_recorrente()
            except (ErroDoAsaas, requests.RequestException, ValueError) as exc:
                log.error("asaas: não consegui criar o link: %s", exc)
                return MENSAGEM_INDISPONIVEL
            url = link.get("url")
            if not link.get("id") or not url:
                log.error("asaas: link sem id ou url")
                return MENSAGEM_INDISPONIVEL
            self.memoria.guardar_link_de_pagamento(numero, link["id"], url)
            log.info("asaas: link de pagamento criado")
        return mensagem_link(url, self.settings)

    def cancelar(self, numero: str) -> str:
        assinatura_id = self.memoria.assinatura_asaas(numero)
        if not assinatura_id:
            return MENSAGEM_SEM_ASSINATURA
        try:
            self.asaas.cancelar_assinatura(assinatura_id)
        except (ErroDoAsaas, requests.RequestException) as exc:
            log.error("asaas: não consegui cancelar: %s", exc)
            return MENSAGEM_INDISPONIVEL.replace("gerar o link de pagamento", "cancelar")
        self.memoria.esquecer_assinatura_asaas(assinatura_id)
        log.info("asaas: assinatura cancelada a pedido")
        atual = self.memoria.assinatura(numero)
        return mensagem_cancelada(atual["pago_ate"] if atual else None)

    def cancelar_em_silencio(self, numero: str) -> None:
        """Quem apaga tudo não pode continuar sendo cobrado."""
        try:
            assinatura_id = self.memoria.assinatura_asaas(numero)
            if assinatura_id:
                self.asaas.cancelar_assinatura(assinatura_id)
                log.info("asaas: assinatura cancelada junto com a exclusão")
        except Exception as exc:  # noqa: BLE001
            log.error("asaas: falha ao cancelar na exclusão: %s", exc)

    def processar_aviso(self, aviso: dict, hoje: dt.date) -> tuple[str, str] | None:
        """Trata um aviso do Asaas. Devolve (número, mensagem) quando há algo
        a dizer à pessoa no WhatsApp."""
        evento = aviso.get("event", "")
        pagamento = aviso.get("payment") or {}
        pagamento_id = pagamento.get("id")
        if not pagamento_id:
            return None
        link_id, assinatura_id = pagamento.get("paymentLink"), pagamento.get("subscription")

        if evento == "PAYMENT_REFUNDED":
            status = self.memoria.estornar_pagamento(pagamento_id, link_id, assinatura_id, hoje)
            log.info("asaas: estorno %s (%s)", pagamento_id, status)
            return None
        if evento not in EVENTOS_DE_PAGAMENTO:
            return None

        vencimento = _data(pagamento.get("dueDate")) or hoje
        pago_ate = mais_um_mes(max(vencimento, hoje)) + TOLERANCIA
        resultado = self.memoria.aplicar_pagamento(
            pagamento_id, evento, link_id, assinatura_id, pago_ate, hoje
        )
        log.info("asaas: %s %s (%s)", evento, pagamento_id, resultado["status"])
        # Só a primeira ativação vira mensagem: renovação mensal cai fora da
        # janela de 24 h do WhatsApp e é silenciosa por natureza.
        if resultado["status"] == "novo" and resultado["primeiro"]:
            return resultado["numero"], mensagem_ativado(resultado["pago_ate"], self.settings)
        return None
