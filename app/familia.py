"""Modo família: até 3 pessoas no mesmo LactAI Plus.

O titular (quem assina) convida até 2 pessoas por link. Compartilham:
- o perfil e o diário do bebê (idade, peso, altura, aleitamento, vacinas);
- as perguntas do plano (o limite mensal é um só para todos);
- os lembretes de vacina.
Não compartilham: as conversas de cada um, que continuam individuais.

Tudo isso é dito antes, para os dois lados: o titular confirma que aceita
mostrar o bebê a quem convidar; quem é convidado lê o que vai ver e o que
vai acontecer com as perguntas e só entra com "1". Dado de saúde e de
criança, compartilhado com outras pessoas, só com consentimento explícito.
"""

import re
import secrets
from urllib.parse import quote

FAMILIA = "familia"
VERSAO_TITULAR = "familia-titular-v1"
VERSAO_MEMBRO = "familia-membro-v1"
MAXIMO_MEMBROS = 2  # além do titular: 3 pessoas no total
_ALFABETO = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # sem 0/O e 1/I
CODIGO = re.compile(r"\bFAM-([A-Z2-9]{6})\b")


def novo_codigo() -> str:
    return "FAM-" + "".join(secrets.choice(_ALFABETO) for _ in range(6))


def encontrar_codigo(texto: str) -> str | None:
    achado = CODIGO.search((texto or "").upper())
    return f"FAM-{achado[1]}" if achado else None


def link_de_convite(numero_whatsapp: str, codigo: str) -> str:
    texto = quote(f"Quero entrar na família no LactAI {codigo}", safe="")
    return f"https://wa.me/{numero_whatsapp}?text={texto}"


# ----------------------------------------------------------------------
# Titular
# ----------------------------------------------------------------------

def explicacao_para_o_titular(vagas: int) -> str:
    return (
        "👨‍👩‍👧 *Modo família*\n\n"
        f"Você pode convidar até {MAXIMO_MEMBROS} pessoas para usar o seu LactAI Plus. "
        f"Ainda {'cabe' if vagas == 1 else 'cabem'} *{vagas}*.\n\n"
        "*Quem você convidar vai:*\n"
        "• ver e atualizar o perfil e o histórico do bebê (idade, peso, altura, "
        "alimentação e vacinas);\n"
        "• usar as perguntas do seu plano — o limite do mês é um só para todos;\n"
        "• receber os lembretes de vacina.\n\n"
        "*Continua só seu:* as suas conversas comigo. Ninguém da família vê o "
        "que você me pergunta, nem você vê as conversas deles.\n\n"
        "Você pode remover alguém quando quiser.\n\n"
        "*1* – Gerar convite\n*2* – Agora não"
    )


def convite_gerado(link: str) -> str:
    return (
        "Pronto! Mande este link para a pessoa que você quer convidar:\n\n"
        f"{link}\n\n"
        "Ele vale por *7 dias* e para *uma pessoa*. Ela vai ver tudo o que vai "
        "ser compartilhado antes de aceitar."
    )


def situacao_da_familia(membros: list[dict]) -> str:
    if not membros:
        return "Ninguém entrou na sua família ainda."
    linhas = "\n".join(f"• número terminado em {m['final']}" for m in membros)
    return (
        f"*Na sua família:*\n{linhas}\n\n"
        "Para remover alguém, mande *REMOVER* e os 4 últimos números (ex.: "
        f"REMOVER {membros[0]['final']})."
    )


SO_NO_PLUS = (
    "O modo família faz parte do *LactAI Plus*: até 3 pessoas dividem o plano, "
    "o histórico do bebê e os lembretes. Mande *MEU PLANO* para ver o seu."
)
PRECISA_GUARDAR = (
    "Para compartilhar o histórico do bebê com a família, primeiro preciso da "
    "sua autorização para guardar os dados. Mande *GUARDAR* para ver como funciona."
)
FAMILIA_CHEIA = "Sua família já tem 3 pessoas, o máximo do plano."
CONVITE_CANCELADO = "Tudo bem, não gerei convite."
REMOVIDO = "Pronto, essa pessoa não faz mais parte da sua família."
NAO_ACHEI_PARA_REMOVER = (
    "Não achei ninguém da sua família com esse final de número. Mande *FAMÍLIA* "
    "para ver quem está nela."
)


# ----------------------------------------------------------------------
# Quem é convidado
# ----------------------------------------------------------------------

EXPLICACAO_PARA_O_CONVIDADO = (
    "👨‍👩‍👧 *Convite para a família no LactAI*\n\n"
    "Alguém convidou você para usar o LactAI Plus dessa pessoa. Antes de "
    "aceitar, veja como funciona:\n\n"
    "*Você vai:*\n"
    "• ver e poder atualizar o perfil e o histórico do bebê da família (idade, "
    "peso, altura, alimentação e vacinas). Informações de saúde, que ficam "
    "guardadas e protegidas;\n"
    "• usar as perguntas do plano de quem convidou — o limite do mês é "
    "dividido entre a família;\n"
    "• receber os lembretes de vacina do bebê (para parar: *PARAR LEMBRETES*).\n\n"
    "*Continua só seu:* as suas conversas comigo. Ninguém da família vê o que "
    "você me pergunta.\n\n"
    "Para sair da família quando quiser: *SAIR DA FAMÍLIA*.\n\n"
    "*1* – Entrar na família\n*2* – Não quero"
)

BEM_VINDA_A_FAMILIA = (
    "Pronto, você entrou na família! 💛 Já vejo o perfil do bebê e uso o "
    "histórico dele nas respostas."
)
NAO_ENTROU = "Tudo bem, você não entrou na família."
MOTIVOS = {
    "invalido": "Esse convite não existe. Confira o link com quem te convidou.",
    "expirado": "Esse convite venceu (eles valem 7 dias). Peça um novo a quem te convidou.",
    "usado": "Esse convite já foi usado. Cada link vale para uma pessoa — peça um novo.",
    "cheia": "Essa família já está completa (3 pessoas).",
    "proprio": "Esse é o seu próprio convite: mande o link para a pessoa que você quer convidar.",
    "ja_em_familia": (
        "Você já faz parte de uma família no LactAI. Para entrar em outra, "
        "primeiro mande *SAIR DA FAMÍLIA*."
    ),
}
SAIU = (
    "Pronto, você saiu da família. Suas conversas continuam como estavam; o "
    "histórico do bebê e o plano da família não aparecem mais para você."
)
NAO_ESTA_EM_FAMILIA = "Você não faz parte de nenhuma família no LactAI."


def situacao_do_membro() -> str:
    return (
        "Você faz parte de uma família no LactAI: vê o histórico do bebê e usa "
        "o plano de quem te convidou. Para sair, mande *SAIR DA FAMÍLIA*."
    )
