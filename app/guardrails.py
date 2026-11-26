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

import re
import unicodedata
from dataclasses import dataclass

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
    r"\bnao (quer |esta |ta |consegue )?mama\w*\b",
    r"\brecusa (o peito|mamar|a mama)\b",
    r"\bdesidrat\w+\b",
    r"\b(sem|pouco|pouca) (xixi|urina|fralda molhada)\b",
    r"\bsangue nas fezes\b",
    r"\b(vomita|vomitando) tudo\b",
    # Icterícia é descrita de várias formas: "olhos amarelos", "amarelo nos
    # olhos", "pele amarelada". Cobrimos as duas ordens.
    r"\b(pele|olhos?|corpo|rosto) \w{0,6} ?muito amarel\w+\b",
    r"\b(pele|olhos?|corpo|rosto) amarel\w+\b",
    r"\bamarel\w+ (n?[oa]s?|em) (pele|olhos?|corpo|rosto)\b",
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


# Mensagens sociais. Sem isso, "obrigada!" cai na checagem de fundamentação,
# não acha citação na base e a mãe recebe "não encontrei isso no material" —
# resposta rude e confusa para quem só está agradecendo.
#
# Em vez de listar frases inteiras, listamos *fragmentos* sociais e aceitamos
# a mensagem como social só se ela for feita inteiramente deles. Assim
# "ok, entendi" e "oi, bom dia, tudo bem?" funcionam sem enumerar combinações.
SOCIAL_TOKENS = [
    # O nome do serviço: a mãe escreve "oi, LactAI" ou "obrigada, LactAI",
    # e o botão da landing já abre a conversa com "Oi, LactAI". Sem isto,
    # o cumprimento ia para o modelo e voltava como "não encontrei".
    r"lact ?ai",
    r"oi+",
    r"ola",
    r"opa",
    r"e ?ai",
    r"hey",
    r"alo+",
    r"bom dia",
    r"boa tarde",
    r"boa noite",
    r"muito obrigad[ao]",
    r"mto obrigad[ao]",
    r"obrigad[ao]",
    r"brigad[ao]",
    r"vlw",
    r"valeu",
    r"agradec\w*",
    r"ok(ay)?",
    r"ta bom",
    r"tudo bem",
    r"td bem",
    r"como vai",
    r"blz",
    r"beleza",
    r"certo",
    r"entendi(do)?",
    r"show",
    r"perfeito",
    r"otimo",
    r"legal",
    r"tchau",
    r"ate logo",
    r"ate mais",
    r"falou",
    r"bj(o?s)?",
    r"abraco",
    r"sim",
    r"nao",
]

_SOCIAL_ALTERNATIVES = "|".join(SOCIAL_TOKENS)
# Um ou mais fragmentos sociais, separados por vírgula/espaço, e nada mais.
_ONLY_SOCIAL = re.compile(
    rf"^(?:{_SOCIAL_ALTERNATIVES})(?:[,\s]+(?:{_SOCIAL_ALTERNATIVES}))*$"
)

# Mensagem só com emoji/pontuação também é social.
_ONLY_SYMBOLS = re.compile(r"^[\W\d_]+$", re.UNICODE)


def _matches(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def is_small_talk(text: str) -> bool:
    """True se a mensagem inteira for social, sem conteúdo de dúvida.

    Deliberadamente conservador: exige que *tudo* na mensagem seja social.
    "oi, meu bebê está com febre" não é small talk — e, de qualquer forma, a
    triagem clínica roda antes desta checagem no pipeline.
    """
    normalized = normalize(text).strip(" .!?,;")
    if not normalized:
        return False
    if len(normalized) > 60:
        return False
    if _ONLY_SYMBOLS.match(normalized):
        return True
    return bool(_ONLY_SOCIAL.match(normalized))


# Pergunta sobre medicamento (ou outra substância) na amamentação. Não muda a
# triagem: só acrescenta à resposta a indicação do e-Lactancia.
MEDICATION_PATTERNS = [
    r"\bremedi\w*",
    r"\bmedica(mento|cao|coes)\w*",
    r"\b(antibiotic|anti-?inflamatori|anticoncepciona|analgesic|antidepressiv"
    r"|antialergic|antitermic|ansiolitic|corticoide)\w*",
    r"\b(pilula|comprimido|xarope|pomada)s?\b",
    r"\bposso (tomar|usar)\b(?! (banho|sol)\b)",
    r"\bcompativel com (a )?(amamentacao|amamentar|aleitamento)\b",
    r"\b(dipirona|novalgina|paracetamol|tylenol|ibuprofeno|nimesulida|diclofenaco"
    r"|cetoprofeno|dorflex|buscopan|amoxicilina|azitromicina|cefalexina|omeprazol"
    r"|loratadina|desloratadina|prednisona|sertralina|fluoxetina|escitalopram"
    r"|domperidona|levotiroxina|metformina)\b",
]


def is_medication_question(text: str) -> bool:
    return _matches(MEDICATION_PATTERNS, normalize(text))


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


def elactancia_note() -> str:
    return (
        "\n\n💊 Para conferir se um medicamento é compatível com a amamentação, "
        "consulte também o *e-Lactancia* (www.e-lactancia.org), referência "
        "gratuita mantida por especialistas, em espanhol e inglês. Antes de "
        "tomar qualquer remédio, confirme com seu médico."
    )


def small_talk_message() -> str:
    return (
        "Oi! Estou por aqui. 💙\n\n"
        "Pode perguntar sobre amamentação, pega, produção de leite, sono do "
        "bebê, doação de leite ou cuidados do dia a dia."
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


# ----------------------------------------------------------------------
# Memória persistente e direitos da titular (ver app/memoria_persistente.py)
# ----------------------------------------------------------------------
DELETE_KEYWORDS = {
    "apagar", "apagar tudo", "apague tudo", "apagar meus dados",
    "excluir", "excluir meus dados", "excluir tudo",
}
MY_DATA_KEYWORDS = {"meus dados", "quais sao meus dados", "o que voce guardou"}
_SIM = {"1", "sim", "pode", "pode guardar", "quero", "ok", "aceito", "pode sim"}
_NAO = {"2", "nao", "nao guarde", "nao quero", "nao precisa", "nao, obrigada", "nao obrigada"}


def is_delete_command(text: str) -> bool:
    return normalize(text).strip(" .!") in DELETE_KEYWORDS


CONFIRM_KEYWORDS = {"confirmar", "confirmo", "pode apagar", "sim, apagar", "sim apagar"}


def is_confirmation(text: str) -> bool:
    return normalize(text).strip(" .!") in CONFIRM_KEYWORDS


STORE_KEYWORDS = {"guardar", "quero guardar", "pode guardar minhas conversas"}


def is_store_command(text: str) -> bool:
    """GUARDAR não autoriza nada sozinho: abre a pergunta com o texto inteiro."""
    return normalize(text).strip(" .!") in STORE_KEYWORDS


def is_my_data_command(text: str) -> bool:
    return normalize(text).strip(" .!?") in MY_DATA_KEYWORDS


def consent_answer(text: str) -> bool | None:
    """Resposta à pergunta de memória: True, False ou None (não é resposta)."""
    resposta = normalize(text).strip(" .!")
    if resposta in _SIM:
        return True
    if resposta in _NAO:
        return False
    return None


def memory_consent_question(url_politica: str = "") -> str:
    """Consentimento específico e destacado (LGPD art. 11, I e art. 14).

    Diz o que é guardado, para quê, por quanto tempo, como apagar e onde ler
    a política inteira. Mudou o texto, muda VERSAO_MEMORIA: a prova do
    consentimento aponta a versão lida.
    """
    politica = f"Todos os detalhes: {url_politica}\n\n" if url_politica else ""
    return (
        "📒 *Posso guardar nossas conversas?*\n\n"
        "Com isso, eu consigo te ajudar melhor:\n"
        "• lembro do que você já me contou, sem você precisar repetir;\n"
        "• levo em conta a idade do bebê, como ele mama e o que já aconteceu, "
        "e a resposta fica mais certeira para o seu caso.\n\n"
        "Se preferir não guardar, *tudo bem*: continuo respondendo suas "
        "dúvidas do mesmo jeito, só não lembro de uma conversa para a outra.\n\n"
        "O que fica guardado inclui informações de saúde, protegidas e usadas "
        "só para te atender. Se você passar 1 ano sem conversar comigo, sua "
        "conta fica inativa e é apagada 2 anos depois. Para apagar antes, "
        "mande *APAGAR*.\n\n"
        f"{politica}"
        "*1* – Pode guardar\n"
        "*2* – Não guarde"
    )


def memory_already_stored_message() -> str:
    return (
        "Já estou guardando nossas conversas. 💛 Para ver o que guardei, mande "
        "*MEUS DADOS*. Para apagar tudo, *APAGAR*."
    )


def memory_consent_yes_message() -> str:
    return (
        "Combinado! A partir de agora eu lembro das nossas conversas. 💛\n\n"
        "Para ver o que guardei, mande *MEUS DADOS*. Para apagar tudo, *APAGAR*."
    )


def memory_consent_no_message() -> str:
    return (
        "Tudo bem, não vou guardar nada. Continuo respondendo suas dúvidas "
        "normalmente. Se mudar de ideia, é só mandar *GUARDAR*."
    )


def delete_confirmation_request() -> str:
    return (
        "Tem certeza? Isso apaga *tudo* o que eu guardei — nossas conversas e "
        "as informações do bebê — e não pode ser desfeito.\n\n"
        "Para confirmar, mande *CONFIRMAR*. Se mudou de ideia, é só continuar "
        "conversando normalmente."
    )


def deleted_message() -> str:
    return (
        "Pronto, apaguei tudo o que estava guardado sobre você e o bebê. "
        "Se quiser voltar, é só mandar uma mensagem. Cuide-se! 💙"
    )


def my_data_message(dados: dict) -> str:
    if not dados.get("conta_desde") and not dados.get("consentimentos"):
        return (
            "Não tenho nada guardado sobre você: nossas conversas não são "
            "armazenadas. Para eu passar a lembrar, mande *GUARDAR*."
        )
    nomes = {"memoria": "Guardar conversas", "lembretes": "Lembretes de vacina",
             "melhoria": "Uso para melhorar o serviço"}
    linhas = ["📒 *O que eu tenho guardado sobre você:*\n"]
    if dados.get("conta_desde"):
        linhas.append(f"• Conversas guardadas desde {dados['conta_desde']:%d/%m/%Y}")
        linhas.append(f"• {dados['mensagens']} mensagens (suas e minhas)")
        linhas.append(f"• {dados['bebes']} bebê(s), {dados['registros_do_bebe']} anotações")
    else:
        linhas.append("• Nenhuma conversa guardada")
    linhas.append("\n*Suas autorizações:*")
    for finalidade, decisao in dados.get("consentimentos", {}).items():
        estado = "sim" if decisao["aceito"] else "não"
        linhas.append(f"• {nomes.get(finalidade, finalidade)}: {estado} ({decisao['em']:%d/%m/%Y})")
    linhas.append(
        "\nPara apagar tudo, mande *APAGAR*. Para uma cópia completa, escreva "
        "para o e-mail da política de privacidade."
    )
    return "\n".join(linhas)


# ----------------------------------------------------------------------
# Fase de testes (ver app/piloto.py)
# ----------------------------------------------------------------------
# Toda mensagem desta seção que pode chegar a alguém fora do atendimento
# normal leva o 192. Quem escreve pode estar numa emergência, e o limite de
# uso nunca pode ser o motivo de uma mãe ficar sem saber para onde ligar.


def pilot_welcome_message(limite_diario: int) -> str:
    limite = (
        f"Nesta fase, cada pessoa pode fazer até *{limite_diario} perguntas por dia*. "
        if limite_diario
        else ""
    )
    return (
        "Pronto, seu número entrou na *fase de testes do LactAI*! 💛\n\n"
        "Pode mandar suas dúvidas sobre amamentação e cuidados com o bebê, "
        "por texto ou por áudio. "
        f"{limite}"
        "Sua opinião é o que vai fazer o serviço melhorar: quando quiser, "
        "conte o que achou das respostas."
    )


def pilot_late_welcome_message(limite_diario: int) -> str:
    """Para quem pediu para entrar e ficou sem resposta, liberado pelo painel."""
    return (
        "Oi! Você pediu para entrar na fase de testes do LactAI e nosso número "
        "ficou fora do ar por um tempo. Desculpe a demora!\n\n"
        + pilot_welcome_message(limite_diario)
    )


def pilot_already_member_message() -> str:
    return "Você já faz parte da fase de testes. É só mandar a sua dúvida. 😊"


def pilot_invite_full_message() -> str:
    return (
        "As vagas deste convite para a fase de testes já foram preenchidas. "
        "Se você acha que deveria participar, fale com quem te enviou o link.\n\n"
        "Em caso de emergência com o bebê, ligue *192*."
    )


def pilot_closed_message() -> str:
    return (
        "Oi! O LactAI está numa *fase de testes fechada*, só para quem "
        "recebeu convite. Se você recebeu um link de convite, toque nele "
        "para entrar.\n\n"
        "Em caso de emergência com o bebê, ligue *192*."
    )


def daily_limit_message(limite: int) -> str:
    return (
        f"Você chegou ao limite de {limite} perguntas de hoje nesta fase de "
        "testes. A contagem recomeça amanhã. 💛\n\n"
        "Se for uma situação urgente com o bebê, procure atendimento ou "
        "ligue *192*."
    )


def budget_exhausted_message() -> str:
    return (
        "O LactAI atingiu o limite de uso da fase de testes por enquanto, e "
        "não consegue responder novas perguntas agora.\n\n"
        "Se for uma situação urgente com o bebê, procure atendimento ou "
        "ligue *192*."
    )
