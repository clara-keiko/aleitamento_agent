"""Planos e limites de uso por pessoa.

A unidade é a pergunta que chama o modelo (pergunta comum, vacina, foto da
caderneta), contada em uso_do_modelo. Emergência, cumprimento, comandos e
questionário não contam. Cada plano tem um teto mensal e um diário, contra
abuso.

Ciclo: o gratuito vira no dia 1º; o pago, no dia da assinatura (assinou dia
20, renova todo dia 20 — no último dia do mês quando ele for mais curto).
"""

import calendar
import datetime as dt
from dataclasses import dataclass

from app.piloto import BRASILIA

GRATUITO = "gratuito"
PLUS = "plus"


@dataclass(frozen=True)
class Plano:
    nome: str
    rotulo: str
    mensal: int
    diario: int


def planos(settings) -> dict[str, Plano]:
    return {
        GRATUITO: Plano(GRATUITO, "gratuito", settings.limite_mensal_gratuito,
                        settings.limite_diario_gratuito),
        PLUS: Plano(PLUS, "LactAI Plus", settings.limite_mensal_assinante,
                           settings.limite_diario_assinante),
    }


def _no_mes(ano: int, mes: int, dia: int) -> dt.date:
    return dt.date(ano, mes, min(dia, calendar.monthrange(ano, mes)[1]))


def ciclo(ancora_dia: int, hoje: dt.date) -> tuple[dt.date, dt.date]:
    """(início, próximo início) do ciclo mensal que contém `hoje`."""
    inicio = _no_mes(hoje.year, hoje.month, ancora_dia)
    if inicio > hoje:
        ano, mes = (hoje.year - 1, 12) if hoje.month == 1 else (hoje.year, hoje.month - 1)
        inicio = _no_mes(ano, mes, ancora_dia)
    ano, mes = (inicio.year + 1, 1) if inicio.month == 12 else (inicio.year, inicio.month + 1)
    return inicio, _no_mes(ano, mes, ancora_dia)


@dataclass(frozen=True)
class Situacao:
    plano: Plano
    usados_mes: int
    usados_dia: int
    renova_em: dt.date

    @property
    def restantes(self) -> int:
        return max(self.plano.mensal - self.usados_mes, 0)


def situacao(memoria, settings, numero: str, agora: dt.datetime) -> Situacao:
    hoje = agora.astimezone(BRASILIA).date()
    assinatura = memoria.assinatura(numero)
    tabela = planos(settings)
    if assinatura and assinatura["pago_ate"] >= hoje:
        plano, ancora = tabela[assinatura["plano"]], assinatura["inicio"].day
    else:
        plano, ancora = tabela[GRATUITO], 1
    inicio, proximo = ciclo(ancora, hoje)

    def meia_noite(dia: dt.date) -> dt.datetime:
        return dt.datetime.combine(dia, dt.time(), tzinfo=BRASILIA)

    return Situacao(
        plano=plano,
        usados_mes=memoria.usos_entre(numero, meia_noite(inicio), meia_noite(proximo)),
        usados_dia=memoria.usos_entre(numero, meia_noite(hoje), meia_noite(hoje + dt.timedelta(days=1))),
        renova_em=proximo,
    )


# ----------------------------------------------------------------------
# Mensagens
# ----------------------------------------------------------------------

def _como_assinar(settings) -> str:
    if settings.asaas_api_key:
        return "Para assinar, mande *ASSINAR*."
    if settings.link_assinatura:
        return f"Para assinar: {settings.link_assinatura}"
    return "Para assinar, escreva para clara@watanabeaitech.com."


def mensagem_limite_mensal(s: Situacao, settings) -> str:
    # "do LactAI Plus", mas "do plano gratuito".
    nome = s.plano.rotulo if s.plano.nome == PLUS else f"plano {s.plano.rotulo}"
    texto = (
        f"Você usou as {s.plano.mensal} perguntas do {nome} deste mês. 💛 "
        f"Elas voltam em {s.renova_em:%d/%m}."
    )
    if s.plano.nome == GRATUITO:
        texto += (
            f"\n\nCom o *LactAI Plus* ({settings.preco_assinatura}/mês), são "
            f"{settings.limite_mensal_assinante} perguntas por mês. {_como_assinar(settings)}"
        )
    return texto + "\n\nEm caso de emergência com o bebê, ligue *192*."


def mensagem_limite_diario(s: Situacao) -> str:
    return (
        f"Você chegou ao limite de {s.plano.diario} perguntas de hoje. A contagem "
        "recomeça amanhã. 💛\n\nSe for uma situação urgente com o bebê, procure "
        "atendimento ou ligue *192*."
    )


def aviso_de_restantes(s: Situacao) -> str:
    """Depois de responder: avisa quando faltam poucas (já descontada esta)."""
    restantes = s.restantes - 1
    if s.plano.nome != GRATUITO or restantes > 2:
        return ""
    if restantes <= 0:
        return f"\n\n_Esta foi a última pergunta do plano gratuito deste mês. Elas voltam em {s.renova_em:%d/%m}._"
    return f"\n\n_{'Resta' if restantes == 1 else 'Restam'} {restantes} pergunta{'s' if restantes > 1 else ''} do plano gratuito este mês._"


def mensagem_meu_plano(s: Situacao, assinatura_ligada: bool = False) -> str:
    return (
        f"📋 *Seu plano: {s.plano.rotulo}*\n"
        f"• {s.usados_mes} de {s.plano.mensal} perguntas usadas neste ciclo\n"
        f"• Renova em {s.renova_em:%d/%m/%Y}\n\n"
        "Emergências, cumprimentos e comandos não contam."
        + (
            "\n\nPara assinar o LactAI Plus: *ASSINAR*."
            if s.plano.nome == GRATUITO and assinatura_ligada else
            "\n\nPara cancelar a renovação: *CANCELAR ASSINATURA*."
            if s.plano.nome == PLUS and assinatura_ligada else ""
        )
    )
