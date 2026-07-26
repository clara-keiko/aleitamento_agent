"""Triagem clínica antes de qualquer chamada ao modelo.

Duas mudanças em relação à primeira versão:

1. Comparação com limite de palavra e sem acento. Antes, `"roxo" in texto`
   casava dentro de outras palavras e `"não respira"` não casava se a mãe
   escrevesse "nao respira" — justamente o caso mais grave.

2. Separação entre *relato* e *pergunta geral*. Antes, "posso amamentar se eu
   estiver com febre?" caía em encaminhamento médico e a mãe nunca recebia a
   orientação educativa. Agora a pergunta genérica é respondida com uma nota
   de segurança anexada; só o relato de sintoma dispara o encaminhamento puro.
"""

from dataclasses import dataclass
import re
import unicodedata

EMERGENCY_NOW = "EMERGENCY_NOW"
REFER_MEDICAL_CARE = "REFER_MEDICAL_CARE"
EDUCATIONAL_OK = "EDUCATIONAL_OK"


def normalize(text: str) -> str:
    """Minúsculas, sem acento, espaços colapsados."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text.lower())
    without_accents = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", without_accents).strip()


# Sinais de emergência: sempre interrompem o fluxo, mesmo em pergunta geral.
EMERGENCY_PATTERNS = [
    r"\bnao (esta )?respira(ndo)?\b",
    r"\bparou de respirar\b",
    r"\bdificuldade (para|pra|de) respirar\b",
    r"\bfalta de ar\b",
    r"\bconvuls\w*",
    r"\b(roxo|roxa|arroxead\w+|cianotic\w+)\b",
    r"\b(desmaiou|inconsciente|nao acorda|desacordad\w+)\b",
    r"\bengasg\w+ (e|com) (nao|sem)\b",
    r"\bmole demais e nao responde\b",
]

# Sinais de alerta: exigem avaliação, mas admitem dúvida educativa.
REFERRAL_PATTERNS = [
    r"\bfebre\b",
    r"\bfebril\b",
    r"\b3[89](\.|,)?\d?\s*(graus|c)\b",
    r"\b4[0-2](\.|,)?\d?\s*(graus|c)\b",
    r"\bmuito molinh\w+\b",
    r"\bmuito sonolent\w+\b",
    r"\bnao (quer )?mama\w*\b",
    r"\brecusa (o peito|mamar|a mama)\b",
    r"\bdesidrat\w+\b",
    r"\b(sem|pouco|pouca) (xixi|urina|fralda molhada)\b",
    r"\bsangue nas fezes\b",
    r"\b(vomita|vomitando) tudo\b",
    r"\b(pele|olhos?) muito amarel\w+\b",
    r"\bictericia\b",
    r"\bnao (esta )?ganhando peso\b",
    r"\bperdendo peso\b",
    r"\bmastite\b",
    r"\bpus\b",
]

# Marcas de relato factual ("está acontecendo agora com a gente").
REPORT_PATTERNS = [
    r"\b(meu|minha) (bebe|filho|filha|neném|nenem|bb)\b",
    r"\b(ele|ela) (esta|ta|anda|fica|nao)\b",
    r"\b(estou|to|tou) com\b",
    r"\bestou sentindo\b",
    r"\besta com\b",
    r"\bta com\b",
    r"\bhoje (de manha|a noite|cedo)\b",
    r"\bdesde (ontem|hoje|essa|esta)\b",
    r"\bfaz \d+ (dias?|horas?)\b",
]

# Marcas de pergunta genérica ("quero entender o assunto").
GENERAL_QUESTION_PATTERNS = [
    r"^\s*(posso|pode|devo|deve)\b",
    r"\be (normal|verdade|seguro)\b",
    r"\bo que (e|significa|acontece)\b",
    r"\bcomo (faco|funciona|saber|identificar|prevenir)\b",
    r"\bem geral\b",
    r"\bquais (sao|os|as)\b",
    r"\bpor que\b",
    r"\bqual a diferenca\b",
]


def _matches(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


@dataclass(frozen=True)
class RiskAssessment:
    level: str
    # Quando True, a resposta educativa sai acompanhada da nota de alerta.
    needs_safety_note: bool = False


def classify_risk(text: str) -> RiskAssessment:
    normalized = normalize(text)

    if _matches(EMERGENCY_PATTERNS, normalized):
        return RiskAssessment(EMERGENCY_NOW)

    if _matches(REFERRAL_PATTERNS, normalized):
        is_report = _matches(REPORT_PATTERNS, normalized)
        is_general = _matches(GENERAL_QUESTION_PATTERNS, normalized)

        # Relato explícito, ou frase sem cara de pergunta: encaminha direto.
        if is_report or not is_general:
            return RiskAssessment(REFER_MEDICAL_CARE)

        # Dúvida conceitual sobre um tema sensível: responde e alerta.
        return RiskAssessment(EDUCATIONAL_OK, needs_safety_note=True)

    return RiskAssessment(EDUCATIONAL_OK)


# ----------------------------------------------------------------------
# Mensagens fixas
# ----------------------------------------------------------------------

def emergency_message() -> str:
    return (
        "⚠️ *Isso pode ser uma emergência.*\n\n"
        "Procure atendimento médico agora. Ligue *192* (SAMU) ou vá ao "
        "pronto-socorro mais próximo.\n\n"
        "Não consigo avaliar sinais de gravidade por mensagem."
    )


def medical_referral_message() -> str:
    return (
        "O que você descreveu pode ser um sinal de alerta, e isso precisa ser "
        "avaliado por um profissional — não dá para checar isso por mensagem.\n\n"
        "Procure a *UBS*, o pediatra ou o pronto-atendimento hoje. "
        "Se aparecer dificuldade para respirar, sonolência fora do comum, "
        "recusa em mamar ou piora rápida, vá ao pronto-socorro imediatamente "
        "ou ligue *192*."
    )


def safety_note() -> str:
    return (
        "\n\n⚠️ Isso vale como informação geral. Se estiver acontecendo com "
        "você ou com o bebê agora, procure avaliação de um profissional de saúde."
    )


def out_of_scope_message() -> str:
    return (
        "Só consigo ajudar com dúvidas de *amamentação e cuidados com o bebê*, "
        "usando o material educativo da nossa base.\n\n"
        "Não encontrei essa resposta no material. Pode reformular a pergunta, "
        "ou levar essa dúvida ao seu profissional de saúde?"
    )


def unsupported_media_message() -> str:
    return (
        "Por enquanto entendo *mensagens de texto e áudio*. "
        "Pode me contar sua dúvida por escrito ou mandar um áudio curto?"
    )


def rate_limited_message() -> str:
    return (
        "Recebi várias mensagens seguidas e preciso de uma pausa curta. "
        "Tente de novo em alguns minutos.\n\n"
        "Se for urgente, procure atendimento médico ou ligue *192*."
    )


def welcome_message() -> str:
    """Enviada no primeiro contato: escopo, natureza automatizada e dados.

    Cobre de uma vez a política da Meta (bot restrito a um caso de uso, sem
    se apresentar como assistente de IA genérico) e a transparência que a
    LGPD exige no tratamento de dado de saúde.
    """
    return (
        "Olá! Sou o assistente educativo de *amamentação e cuidados com o bebê*. "
        "Respondo de forma automatizada, com base em material técnico "
        "(Ministério da Saúde e literatura de aleitamento).\n\n"
        "*Importante:*\n"
        "• Não faço diagnóstico nem prescrevo tratamento.\n"
        "• Não substituo consulta com profissional de saúde.\n"
        "• Em emergência, ligue *192* (SAMU).\n\n"
        "Suas mensagens são usadas só para responder você. "
        "Envie *SAIR* a qualquer momento para encerrar e apagar seu histórico.\n\n"
        "Qual é a sua dúvida?"
    )


def opt_out_message() -> str:
    return (
        "Pronto, apaguei o histórico desta conversa. "
        "É só mandar uma mensagem quando quiser retomar. Cuide-se! 💙"
    )


OPT_OUT_KEYWORDS = {"sair", "parar", "cancelar", "stop", "descadastrar"}


def is_opt_out(text: str) -> bool:
    return normalize(text).strip(" .!") in OPT_OUT_KEYWORDS
