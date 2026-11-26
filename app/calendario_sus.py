"""Calendário Nacional de Vacinação da criança, direto do Ministério da Saúde.

A busca na web não acha o calendário do SUS: no gov.br ele é publicado como
arquivo para baixar, que os buscadores mal indexam. Então o serviço baixa o
próprio arquivo oficial, extrai o texto e o mantém em memória, atualizando
uma vez por dia. A parte "No SUS" da resposta sai só desse texto, com o link
de onde veio; nada do calendário fica na base nem no código.

Sem o arquivo (gov.br fora do ar, formato mudou), a resposta volta a buscar
na web e, sem fonte, manda os links oficiais — nunca inventa calendário.
"""

import datetime as dt
import html
import io
import re
import threading
from dataclasses import dataclass

import requests

from app.logging_utils import get_logger

log = get_logger(__name__)

# Em ordem de preferência: o arquivo oficial da criança e, se ele falhar, a
# página do calendário.
FONTES = [
    "https://www.gov.br/saude/pt-br/vacinacao/arquivos/calendario-nacional-de-vacinacao-crianca/@@download/file",
    "https://www.gov.br/saude/pt-br/vacinacao/calendario",
]
PAGINA_PUBLICA = "https://www.gov.br/saude/pt-br/vacinacao/calendario"
LIMITE_CARACTERES = 20_000
# Menos que isto não é calendário: é página de erro ou de redirecionamento.
MINIMO_CARACTERES = 800


@dataclass(frozen=True)
class Calendario:
    texto: str
    url: str
    baixado_em: dt.datetime


def _texto_do_pdf(conteudo: bytes) -> str:
    from pypdf import PdfReader

    leitor = PdfReader(io.BytesIO(conteudo))
    return "\n".join((pagina.extract_text() or "") for pagina in leitor.pages)


def _texto_do_html(conteudo: str) -> str:
    conteudo = re.sub(r"(?is)<(script|style|nav|header|footer)\b.*?</\1>", " ", conteudo)
    conteudo = re.sub(r"(?i)<(br|/p|/tr|/li|/h\d)[^>]*>", "\n", conteudo)
    conteudo = re.sub(r"(?i)</t[dh]>", " | ", conteudo)
    texto = html.unescape(re.sub(r"<[^>]+>", " ", conteudo))
    return texto


def _limpar(texto: str) -> str:
    linhas = (re.sub(r"[ \t]+", " ", linha).strip() for linha in texto.splitlines())
    return "\n".join(linha for linha in linhas if linha)[:LIMITE_CARACTERES]


def baixar(fontes: list[str] = FONTES, get=requests.get) -> Calendario | None:
    for url in fontes:
        try:
            resposta = get(url, timeout=20, headers={"User-Agent": "LactAI/1.0"})
            resposta.raise_for_status()
        except requests.RequestException as exc:
            log.warning("calendário SUS: falha ao baixar %s: %s", url, exc)
            continue

        tipo = resposta.headers.get("content-type", "").lower()
        try:
            if "pdf" in tipo or resposta.content[:4] == b"%PDF":
                texto = _texto_do_pdf(resposta.content)
            else:
                texto = _texto_do_html(resposta.text)
        except Exception as exc:  # noqa: BLE001 - PDF malformado, encoding
            log.warning("calendário SUS: não consegui ler %s: %s", url, exc)
            continue

        texto = _limpar(texto)
        if len(texto) < MINIMO_CARACTERES or "vacina" not in texto.lower():
            log.warning("calendário SUS: %s veio sem calendário (%d caracteres)", url, len(texto))
            continue

        # Conteúdo público, não pessoal: o começo vai ao log para dar para
        # conferir o que o serviço está usando.
        log.info(
            "calendário SUS carregado de %s: %d caracteres. Início: %s",
            url, len(texto), texto[:300].replace("\n", " / "),
        )
        return Calendario(texto=texto, url=url, baixado_em=dt.datetime.now(dt.UTC))

    log.error("calendário SUS indisponível em todas as fontes")
    return None


class CalendarioSUS:
    """Guarda o último calendário bom; uma falha não apaga o anterior."""

    def __init__(self, baixar_agora=baixar):
        self._baixar = baixar_agora
        self._atual: Calendario | None = None
        self._lock = threading.Lock()

    def atualizar(self) -> Calendario | None:
        novo = self._baixar()
        if novo is not None:
            with self._lock:
                self._atual = novo
        return self.atual

    @property
    def atual(self) -> Calendario | None:
        with self._lock:
            return self._atual
