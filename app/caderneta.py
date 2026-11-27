"""Leitura da página de vacinas da Caderneta da Criança, por foto.

A foto nunca é guardada: é lida e descartada. O que fica, e só depois de a
pessoa confirmar, é vacina + dose + data no diário do bebê.

Leitura de foto erra — letra à mão, carimbo, foto torta. Por isso:
- o modelo só devolve o que consegue ler, com data; o resto fica de fora;
- a pessoa confere a lista antes de qualquer coisa ser gravada;
- o LactAI nunca diz "a vacinação está em dia": a conferência é na UBS.
"""

import base64
import datetime as dt
import json
from dataclasses import dataclass, field

from openai import OpenAIError

from app.llm import token_usage
from app.logging_utils import get_logger

log = get_logger(__name__)

PROMPT = """\
Você lê fotos da Caderneta da Criança (Ministério da Saúde, Brasil) ou de \
cartões de vacinação, para registrar as vacinas que JÁ FORAM APLICADAS.

Regras:
- Diga se a foto é de fato uma página de registro de vacinas. Foto do bebê, \
de pele, de peito, de exame ou de outra coisa: e_caderneta = false e lista vazia.
- Liste só vacinas com registro de aplicação (data, carimbo, rubrica ou lote). \
Campos em branco não são vacina tomada.
- Use nomes padronizados: BCG, Hepatite B, Pentavalente, Hexavalente, VIP, \
VOP, Rotavírus, Pneumocócica 10, Pneumocócica 13, Pneumocócica 15, \
Pneumocócica 20, Meningocócica C, Meningocócica ACWY, Meningocócica B, \
Febre amarela, Tríplice viral, Tetraviral, Varicela, Hepatite A, DTP, \
Influenza, Covid-19. Se não reconhecer, escreva como está na caderneta.
- dose: "1ª dose", "2ª dose", "3ª dose", "reforço", "1º reforço", \
"2º reforço" ou "dose única"; vazio se não estiver indicado.
- data no formato AAAA-MM-DD. Se a data não estiver legível, use null — \
nunca invente nem estime.
- Não leia nem devolva nome da criança, número do cartão SUS, endereço ou \
qualquer outro dado pessoal.
"""

ESQUEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["e_caderneta", "vacinas"],
    "properties": {
        "e_caderneta": {"type": "boolean"},
        "vacinas": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["vacina", "dose", "data"],
                "properties": {
                    "vacina": {"type": "string"},
                    "dose": {"type": "string"},
                    "data": {"type": ["string", "null"]},
                },
            },
        },
    },
}


@dataclass
class VacinaLida:
    vacina: str
    dose: str
    data: dt.date | None

    @property
    def rotulo(self) -> str:
        return f"{self.vacina} – {self.dose}" if self.dose else self.vacina


@dataclass
class LeituraCaderneta:
    e_caderneta: bool = False
    vacinas: list[VacinaLida] = field(default_factory=list)
    erro: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


def _data(texto: str | None) -> dt.date | None:
    try:
        return dt.date.fromisoformat(texto) if texto else None
    except ValueError:
        return None


def ler_caderneta(engine, imagem: bytes, mime: str) -> LeituraCaderneta:
    url = f"data:{mime or 'image/jpeg'};base64,{base64.b64encode(imagem).decode()}"
    parametros = {
        "model": engine.settings.openai_model,
        "input": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": [
                {"type": "input_text", "text": "Quais vacinas estão registradas nesta foto?"},
                {"type": "input_image", "image_url": url},
            ]},
        ],
        "text": {"format": {
            "type": "json_schema", "name": "caderneta", "schema": ESQUEMA, "strict": True,
        }},
        "max_output_tokens": engine.teto_de_saida,
    }
    if engine.raciocina:
        parametros["reasoning"] = {"effort": engine.settings.reasoning_effort}

    try:
        resposta = engine.client.responses.create(**parametros)
        dados = json.loads(getattr(resposta, "output_text", "") or "{}")
    except (OpenAIError, ValueError) as exc:
        log.error("falha ao ler a caderneta: %s", exc)
        return LeituraCaderneta(erro=True)

    entrada, saida = token_usage(resposta)
    return LeituraCaderneta(
        e_caderneta=bool(dados.get("e_caderneta")),
        vacinas=[
            VacinaLida(
                vacina=(v.get("vacina") or "").strip(),
                dose=(v.get("dose") or "").strip(),
                data=_data(v.get("data")),
            )
            for v in dados.get("vacinas", [])
            if (v.get("vacina") or "").strip()
        ],
        input_tokens=entrada,
        output_tokens=saida,
    )


def separar(leitura: LeituraCaderneta, nascimento: dt.date, hoje: dt.date):
    """(com data plausível, sem data ou com data impossível)."""
    validas, duvidosas = [], []
    for vacina in leitura.vacinas:
        if vacina.data and nascimento - dt.timedelta(days=1) <= vacina.data <= hoje:
            validas.append(vacina)
        else:
            duvidosas.append(vacina)
    return validas, duvidosas


# ----------------------------------------------------------------------
# Mensagens
# ----------------------------------------------------------------------

def mensagem_para_conferir(validas: list[VacinaLida], duvidosas: list[VacinaLida]) -> str:
    linhas = ["📒 *Encontrei estas vacinas na foto:*"]
    linhas += [f"• {v.rotulo} — {v.data:%d/%m/%Y}" for v in validas]
    if duvidosas:
        nomes = ", ".join(v.rotulo for v in duvidosas)
        linhas.append(f"\nNão consegui ler a data de: {nomes}. Essas eu não vou anotar.")
    linhas.append(
        "\n*Está certo?*\n1 – Sim, pode anotar\n2 – Tem erro, não anote"
        "\n\n_Leitura de foto pode errar. Confira sempre a caderneta com a UBS "
        "ou o pediatra._"
    )
    return "\n".join(linhas)


def mensagem_anotadas(novas: int, repetidas: int) -> str:
    texto = f"Anotado! ✅ {novas} vacina{'s' if novas != 1 else ''} no histórico do bebê."
    if repetidas:
        texto += f" ({repetidas} já estava{'m' if repetidas != 1 else ''} anotada{'s' if repetidas != 1 else ''}.)"
    return texto + (
        "\n\nSe tiver mais páginas com vacinas, pode mandar a foto de cada uma. "
        "Para ver o que guardei, *MEUS DADOS*."
    )


MENSAGEM_NAO_ANOTADO = (
    "Tudo bem, não anotei nada. Você pode mandar outra foto — de frente, com "
    "boa luz, uma página por vez — ou me dizer as vacinas por texto."
)

MENSAGEM_NAO_E_CADERNETA = (
    "Por enquanto, em fotos, eu só consigo ler a *página de vacinas da "
    "caderneta* da criança. Não avalio fotos do bebê, da pele ou do peito: "
    "se for algo sobre a saúde de vocês, me conte por texto — e, se for "
    "urgente, procure atendimento ou ligue *192*."
)

MENSAGEM_ILEGIVEL = (
    "Não consegui ler as vacinas com segurança nesta foto. 🙈 Tente de novo: "
    "de frente, com boa luz, sem reflexo, uma página por vez."
)

MENSAGEM_SEM_AUTORIZACAO = (
    "Consigo ler a página de vacinas da caderneta e anotar no histórico do "
    "bebê — mas, para guardar, preciso da sua autorização. Mande *GUARDAR* "
    "para ver como funciona."
)

MENSAGEM_SEM_BEBE = (
    "Para anotar as vacinas, primeiro preciso saber a data de nascimento do "
    "bebê. Mande *PERFIL* para responder rapidinho, e depois me mande a foto "
    "de novo."
)

MENSAGEM_FOTO_FALHOU = "Não consegui abrir a foto. 🙈 Pode mandar de novo?"
