"""Entrypoint FastAPI do agente de aleitamento no WhatsApp.

O webhook responde 200 imediatamente e processa a mensagem em background.
Isso não é detalhe de performance: a Meta reentrega o evento quando o 200
demora, e uma chamada com file_search leva vários segundos. Processando de
forma síncrona, a mãe recebia a mesma resposta duas ou três vezes.
"""

import base64
import hashlib
import hmac
import html
import json
import re
import sqlite3
from pathlib import Path
from typing import Annotated
from urllib.parse import parse_qs, parse_qsl, quote, urlencode, urlparse

from fastapi import BackgroundTasks, FastAPI, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse

from app.channels import build_channel
from app.channels.meta_cloud import MetaCloudChannel
from app.channels.web import WebChannel
from app.config import PROVIDER_EVOLUTION, PROVIDER_TWILIO, settings
from app.llm import AssistantEngine
from app.logging_utils import configure_logging, get_logger
from app.piloto import (
    AUTORIZADO,
    CODIGO_INVALIDO,
    ESGOTADO,
    JA_AUTORIZADO,
    RegistroDoPiloto,
    ler_convites,
    ler_numeros,
)
from app.pipeline import MessagePipeline

configure_logging()
log = get_logger(__name__)

app = FastAPI(title="Agente de Aleitamento", version="2.1.0")

channel = build_channel(settings)
engine = AssistantEngine(settings)


def _abrir_registro() -> RegistroDoPiloto:
    """Abre o banco do piloto; se o disco falhar, segue em memória.

    Desligar os limites em silêncio seria o pior desfecho: o orçamento
    deixaria de proteger justamente quando algo deu errado. Em memória eles
    continuam valendo — só zeram a cada reinício, e o log avisa.
    """
    convites = ler_convites(settings.convites)
    try:
        return RegistroDoPiloto(settings.db_path, convites)
    except (OSError, sqlite3.Error) as exc:
        log.error(
            "não consegui abrir %s (%s); convites e limites ficam em memória "
            "e zeram a cada reinício", settings.db_path, exc,
        )
        return RegistroDoPiloto(":memory:", convites)


piloto = _abrir_registro()
pipeline = MessagePipeline(settings, channel, engine, piloto=piloto)

# Protótipo web: mesmo motor, mesmo pipeline, canal diferente. Ganha o próprio
# MessagePipeline só para que histórico e rate limit do navegador não se
# misturem com os do WhatsApp.
web_channel = WebChannel()
web_pipeline = MessagePipeline(settings, web_channel, engine, piloto=piloto)

CHAT_HTML = Path(__file__).parent / "app" / "static" / "chat.html"
PRIVACIDADE_HTML = Path(__file__).parent / "app" / "static" / "privacidade.html"

if not settings.ready:
    # Avisa alto, mas deixa o processo de pé: sem isso a plataforma reinicia
    # em loop e o log do erro nunca chega a ser lido.
    log.error(
        "configuração incompleta, faltam: %s. /webhook vai recusar mensagens.",
        ", ".join(settings.missing()),
    )


def _status_payload() -> dict:
    missing = settings.missing()
    return {
        "status": "ok" if not missing else "degraded",
        "provider": settings.provider,
        "model": settings.openai_model,
        "audio": settings.enable_audio,
        "signature_required": settings.require_signature,
        # Separados de propósito: o protótipo web funciona sem as credenciais
        # de WhatsApp, então dá para estar pronto para um e não para o outro.
        "web_ready": settings.web_ready,
        "whatsapp_ready": settings.ready,
        "missing_for_web": settings.missing_core(),
        "missing_for_whatsapp": settings.missing(),
        "missing_env": missing,
    }


@app.get("/health")
def health() -> JSONResponse:
    """Liveness: responde 200 sempre que o processo está de pé.

    Deliberadamente não devolve 503 com configuração incompleta. O
    healthcheck da plataforma aponta para cá; se ele falhasse por falta de
    variável, o deploy nunca subiria e voltaríamos ao problema original de
    não conseguir ler o log para descobrir o que faltava.
    """
    return JSONResponse(_status_payload(), status_code=200)


@app.get("/health/ready")
def readiness() -> JSONResponse:
    """Readiness: 503 enquanto o serviço não puder atender de verdade.

    Com a Evolution, "configurado" não basta: o número pode estar
    desconectado com todas as variáveis certas. Por isso, nesse canal, o
    estado da conexão também decide — é este endpoint que um monitor externo
    deve olhar para avisar quando o WhatsApp cair.
    """
    payload = _status_payload()
    pronto = not payload["missing_env"]

    if pronto and settings.provider == PROVIDER_EVOLUTION:
        estado = channel.estado_da_conexao()
        payload["whatsapp_connection"] = estado
        pronto = estado == "open"

    return JSONResponse(payload, status_code=200 if pronto else 503)


# ----------------------------------------------------------------------
# Protótipo web
# ----------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root() -> Response:
    if settings.enable_web:
        return RedirectResponse("/chat")
    return JSONResponse({"status": "ok"})


@app.get("/chat", response_class=HTMLResponse)
def chat_ui() -> Response:
    if not settings.enable_web:
        return PlainTextResponse("Not Found", status_code=404)
    return HTMLResponse(CHAT_HTML.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Painel do piloto
# ----------------------------------------------------------------------

def _admin_autorizado(headers) -> bool:
    """Autenticação básica do navegador: ele mesmo pede a senha.

    Senha na URL (?token=) ficaria no histórico e no log de acesso do
    servidor. Com Basic auth, o navegador abre a janela de login e guarda a
    senha só durante a sessão. O usuário é ignorado; vale a senha.
    """
    esperado = settings.admin_token
    cabecalho = headers.get("authorization", "")
    if not esperado or not cabecalho.lower().startswith("basic "):
        return False
    try:
        decodificado = base64.b64decode(cabecalho[6:]).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    _, _, senha = decodificado.partition(":")
    return hmac.compare_digest(senha, esperado)


def _negar_admin(request: Request) -> Response | None:
    """None = pode entrar. Senão, a resposta de recusa."""
    if not settings.admin_token:
        return PlainTextResponse("Not Found", status_code=404)
    if not _admin_autorizado(request.headers):
        return PlainTextResponse(
            "Senha necessária", status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="LactAI piloto", charset="UTF-8"'},
        )
    # O navegador reenvia a senha do Basic auth a qualquer pedido para este
    # endereço, inclusive um formulário escondido em outro site. A origem
    # diferente é o sinal de que não foi você quem clicou.
    origem = request.headers.get("origin")
    if origem and urlparse(origem).netloc != request.headers.get("host", ""):
        return PlainTextResponse("Origem não permitida", status_code=403)
    return None


def _link_de_convite(codigo: str) -> str:
    numero = "".join(c for c in settings.numero_whatsapp if c.isdigit())
    if not numero:
        return ""
    texto = quote(f"Quero participar da fase de testes do LactAI {codigo}", safe="")
    return f"https://wa.me/{numero}?text={texto}"


_ESTILO_PAINEL = """
body{font:16px/1.5 system-ui,sans-serif;max-width:46rem;margin:2rem auto;padding:0 1rem;
color:#1b1b1f;background:#fbfaf8}
h1{font-size:1.4rem;margin:0 0 .25rem} h2{font-size:1.1rem;margin:2rem 0 .25rem}
.sub{color:#5c5f66;margin:0 0 1.5rem}
.grade{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:.75rem;margin-bottom:1.5rem}
.num{background:#fff;border:1px solid #e2dfd9;border-radius:6px;padding:.8rem 1rem}
.num b{display:block;font-size:1.5rem;font-variant-numeric:tabular-nums}
.num span{color:#5c5f66;font-size:.85rem}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e2dfd9}
td,th{padding:.6rem .75rem;border-bottom:1px solid #e2dfd9;text-align:left;vertical-align:top}
input,textarea,select{width:100%;box-sizing:border-box;font:inherit;font-size:.9rem;padding:.4rem}
input[readonly]{font-size:.8rem;padding:.3rem} code{font-size:.95rem}
button{font:inherit;margin-top:.5rem;padding:.5rem 1rem;background:#1b1b1f;color:#fff;
border:0;border-radius:6px;cursor:pointer}
.alerta{background:#fdeee8;border-left:3px solid #b2451c;padding:.7rem 1rem}
"""


def _pagina_admin(titulo: str, corpo: str) -> HTMLResponse:
    pagina = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(titulo)}</title><style>{_ESTILO_PAINEL}</style></head><body>
{corpo}
</body></html>"""
    return HTMLResponse(pagina, headers={"Cache-Control": "no-store"})


@app.get("/admin/piloto", response_class=HTMLResponse)
def painel_do_piloto(request: Request) -> Response:
    negado = _negar_admin(request)
    if negado:
        return negado

    r = piloto.resumo()
    orcamento = settings.orcamento_mensal_usd
    limite = settings.limite_diario_por_pessoa
    e = html.escape

    linhas_convite = "".join(
        f"<tr><td><code>{e(c['codigo'])}</code></td>"
        f"<td>{c['usados']} de {c['limite'] or '∞'}</td>"
        f"<td>{_bloco_link(_link_de_convite(c['codigo']))}</td></tr>"
        for c in r["convites"]
    ) or '<tr><td colspan="3">Nenhum convite configurado (variável CONVITES).</td></tr>'

    gasto = r["gasto_do_mes_usd"]
    teto = f" de US$ {orcamento:.2f}" if orcamento else " (sem teto)"
    alerta = (
        '<p class="alerta">Orçamento do mês atingido: o agente parou de chamar o modelo.</p>'
        if orcamento and gasto >= orcamento else ""
    )
    modo = "convite (fechado)" if settings.modo_acesso == "convite" else "aberto"

    return _pagina_admin("Piloto LactAI", f"""
<h1>Fase de testes do LactAI</h1>
<p class="sub">Hoje: {e(r["dia"])} · acesso: {e(modo)}</p>
{alerta}
<div class="grade">
  <div class="num"><b>{r["testadores"]}</b><span>testadoras autorizadas</span></div>
  <div class="num"><b>{r["ativos_hoje"]}</b><span>perguntaram hoje</span></div>
  <div class="num"><b>{r["perguntas_hoje"]}</b><span>perguntas hoje</span></div>
  <div class="num"><b>US$ {gasto:.2f}</b><span>gasto do mês{e(teto)}</span></div>
</div>
<p class="sub">Limite por pessoa: {limite or "sem limite"} perguntas por dia.</p>
<table><tr><th>Convite</th><th>Vagas</th><th>Link para enviar</th></tr>{linhas_convite}</table>
{_formulario_liberar(r["convites"])}""")


def _bloco_link(link: str) -> str:
    if not link:
        return "<em>defina NUMERO_WHATSAPP para gerar o link</em>"
    return f'<input readonly value="{html.escape(link)}" onclick="this.select()">'


def _formulario_liberar(convites: list[dict]) -> str:
    if not convites:
        return ""
    e = html.escape
    if len(convites) == 1:
        escolha = f'<input type="hidden" name="codigo" value="{e(convites[0]["codigo"])}">'
    else:
        opcoes = "".join(
            f'<option value="{e(c["codigo"])}">{e(c["codigo"])}</option>' for c in convites
        )
        escolha = f'<p>Convite: <select name="codigo">{opcoes}</select></p>'
    return f"""
<h2>Liberar números à mão</h2>
<p class="sub">Para quem mandou a mensagem do convite e ficou sem resposta (por
exemplo, com o WhatsApp desconectado). Cada número ocupa uma vaga e recebe as
boas-vindas na hora. <b>Só cole números que pediram para entrar:</b> mensagem
para quem não pediu é o que faz o WhatsApp bloquear o número.</p>
<form method="post" action="/admin/piloto/liberar">{escolha}
<textarea name="numeros" rows="5" placeholder="(21) 99999-0000&#10;um número por linha"></textarea>
<button type="submit">Liberar e mandar boas-vindas</button>
</form>"""


_ROTULO_LIBERACAO = {
    AUTORIZADO: "liberado, boas-vindas enviadas",
    JA_AUTORIZADO: "já estava liberado; nada enviado",
    ESGOTADO: "não liberado: vagas esgotadas",
    CODIGO_INVALIDO: "não liberado: convite inexistente",
}


@app.post("/admin/piloto/liberar", response_class=HTMLResponse)
async def liberar_numeros(request: Request, background_tasks: BackgroundTasks) -> Response:
    negado = _negar_admin(request)
    if negado:
        return negado

    formulario = dict(parse_qsl((await request.body()).decode("utf-8", "replace")))
    codigo = formulario.get("codigo", "").strip().upper()
    brutos = [b for b in re.split(r"[\n,;]+", formulario.get("numeros", "")) if b.strip()]
    voltar = '<p><a href="/admin/piloto">← Voltar ao painel</a></p>'

    # Liberar com o WhatsApp caído gastaria a vaga e a mensagem se perderia:
    # a pessoa continuaria sem saber que entrou.
    if settings.provider == PROVIDER_EVOLUTION:
        estado = channel.estado_da_conexao()
        if estado != "open":
            return _pagina_admin("Piloto LactAI", f"""
<h1>WhatsApp desconectado</h1>
<p class="alerta">O número está <b>{html.escape(estado)}</b>. Reconecte pelo QR
antes de liberar, ou as boas-vindas não chegam. Nada foi alterado.</p>{voltar}""")

    linhas = []
    for bruto in brutos:
        numero = ler_numeros(bruto)[0]
        final = numero[-4:] if numero else "".join(c for c in bruto if c.isdigit())[-4:]
        if not numero:
            resultado = "não parece um telefone; confira e tente de novo"
        else:
            situacao = piloto.usar_convite(numero, codigo)
            resultado = _ROTULO_LIBERACAO[situacao]
            if situacao == AUTORIZADO:
                background_tasks.add_task(pipeline.boas_vindas_tardias, numero)
        linhas.append(f"<tr><td>…{html.escape(final)}</td><td>{html.escape(resultado)}</td></tr>")

    log.info("liberação manual: %d número(s) pelo painel", len(linhas))
    tabela = "".join(linhas) or '<tr><td colspan="2">Nenhum número informado.</td></tr>'
    return _pagina_admin("Piloto LactAI", f"""
<h1>Liberação manual</h1>
<table><tr><th>Final</th><th>Resultado</th></tr>{tabela}</table>{voltar}""")


@app.get("/privacidade", response_class=HTMLResponse)
def privacidade() -> Response:
    """Política de privacidade, sempre pública.

    Não depende de ENABLE_WEB nem de código de acesso: a Meta exige uma URL
    aberta para publicar o aplicativo, e um aviso de privacidade que só
    aparece para quem tem senha não cumpre o papel de aviso.
    """
    return HTMLResponse(PRIVACIDADE_HTML.read_text(encoding="utf-8"))


def _web_access_allowed(headers: dict) -> bool:
    """Sem código configurado, o protótipo é aberto.

    Numa URL pública isso significa deixar qualquer pessoa gastar sua cota
    da OpenAI — defina WEB_ACCESS_CODE antes de compartilhar o link.
    """
    expected = settings.web_access_code
    if not expected:
        return True
    provided = headers.get("x-access-code", "")
    return bool(provided) and hmac.compare_digest(expected, provided)


@app.post("/api/chat")
async def chat_api(request: Request) -> Response:
    if not settings.enable_web:
        return JSONResponse({"error": "web desabilitado"}, status_code=404)

    headers = {key.lower(): value for key, value in request.headers.items()}
    if not _web_access_allowed(headers):
        return JSONResponse({"error": "codigo_invalido"}, status_code=401)

    # Só o núcleo: o protótipo web não usa credencial de WhatsApp.
    if not settings.web_ready:
        faltando = settings.missing_core()
        log.error("protótipo web sem configuração: faltam %s", ", ".join(faltando))
        return JSONResponse(
            {"error": "nao_configurado", "missing_env": faltando},
            status_code=503,
        )

    try:
        payload = await request.json()
    except (ValueError, UnicodeDecodeError):
        return JSONResponse({"error": "json_invalido"}, status_code=400)

    mensagens = web_channel.parse_webhook(payload)
    if not mensagens:
        return JSONResponse({"error": "sessao_ou_texto_ausente"}, status_code=400)

    mensagem = mensagens[0]
    try:
        outcome = web_pipeline.handle(mensagem)
    except Exception as exc:  # noqa: BLE001
        log.exception("falha no protótipo web: %s", exc)
        web_channel.discard(mensagem.sender)
        return JSONResponse({"error": "falha_interna"}, status_code=500)

    # Diferente do WhatsApp, aqui devolvemos na mesma requisição.
    return JSONResponse(
        {"replies": web_channel.drain(mensagem.sender), "outcome": outcome}
    )


# ----------------------------------------------------------------------
# WhatsApp
# ----------------------------------------------------------------------

@app.get("/webhook", response_class=PlainTextResponse)
def verify_webhook(
    hub_mode: Annotated[str | None, Query(alias="hub.mode")] = None,
    hub_verify_token: Annotated[str | None, Query(alias="hub.verify_token")] = None,
    hub_challenge: Annotated[str | None, Query(alias="hub.challenge")] = None,
):
    """Handshake de verificação da Meta (não usado pela Twilio)."""
    if not isinstance(channel, MetaCloudChannel):
        return PlainTextResponse("Not Found", status_code=404)

    if channel.verify_challenge(hub_mode, hub_verify_token) and hub_challenge:
        return PlainTextResponse(hub_challenge)

    log.warning("handshake de verificação recusado")
    return PlainTextResponse("Forbidden", status_code=403)


def _assinaturas_presentes(headers: dict, url: str = "") -> str:
    """Diz *qual* provedor assinou, não só que veio assinatura.

    Um webhook da Twilio chegando num app em modo Meta é indistinguível de
    uma credencial errada, se o log só disser que havia algum header. Nomear
    o header separa "configurei a credencial errada" de "este webhook é de
    outro provedor" — dois problemas com correções opostas.
    """
    conhecidos = {
        "x-hub-signature-256": "meta",
        "x-twilio-signature": "twilio",
        "apikey": "evolution",
    }
    achados = [rotulo for header, rotulo in conhecidos.items() if headers.get(header)]
    if "apikey" in parse_qs(urlparse(url).query):
        achados.append("evolution(url)")
    return ",".join(achados)


def _sem_segredos(url: str) -> str:
    """A URL do webhook da Evolution leva a chave no query string; o log
    de falha imprime a URL, e não pode imprimir a chave junto."""
    partes = urlparse(url)
    if not partes.query:
        return url
    consulta = parse_qs(partes.query, keep_blank_values=True)
    for nome in ("apikey", "token", "key"):
        if nome in consulta:
            consulta[nome] = ["***"]
    return partes._replace(query=urlencode(consulta, doseq=True)).geturl()


def _impressao_da_credencial() -> str:
    """Identifica *qual* segredo está em uso, sem revelá-lo.

    Comparar o segredo do painel com o que o servidor carregou é o passo que
    fecha o diagnóstico de assinatura — e é o passo que ninguém consegue dar,
    porque o valor fica mascarado dos dois lados. O hash permite a comparação
    sem expor nada: quem tem o valor certo reproduz a mesma impressão.
    """
    segredos = {
        PROVIDER_TWILIO: ("TWILIO_AUTH_TOKEN", settings.twilio_auth_token),
        PROVIDER_EVOLUTION: ("EVOLUTION_API_KEY", settings.evolution_api_key),
    }
    nome, segredo = segredos.get(settings.provider, ("APP_SECRET", settings.app_secret))
    if not segredo:
        return f"{nome}=ausente"
    digest = hashlib.sha256(segredo.encode()).hexdigest()[:8]
    # Espaço em branco na colagem é invisível no painel e some no hash —
    # por isso o comprimento vai junto.
    return f"{nome}=sha256:{digest} len={len(segredo)}"


@app.post("/webhook")
async def receive_webhook(request: Request, background: BackgroundTasks) -> Response:
    raw_body = await request.body()
    headers = {key.lower(): value for key, value in request.headers.items()}

    # A Twilio assina a URL pública; atrás de proxy, request.url pode vir
    # como http:// interno e invalidar a assinatura.
    if settings.public_base_url:
        signed_url = settings.public_base_url.rstrip("/") + request.url.path
        if request.url.query:
            signed_url += f"?{request.url.query}"
    else:
        signed_url = str(request.url)

    if not channel.verify_signature(raw_body, headers, signed_url):
        # A mensagem antiga dizia só "inválida", e as três causas possíveis
        # (URL divergente, credencial errada, header ausente) exigem ações
        # opostas. Sem esses dados, depurar vira tentativa e erro.
        log.warning(
            "assinatura inválida no webhook; descartando "
            "(url_conferida=%s origem_da_url=%s assinaturas=%s credencial=%s)",
            _sem_segredos(signed_url),
            "PUBLIC_BASE_URL" if settings.public_base_url else "request.url",
            _assinaturas_presentes(headers, signed_url) or "nenhuma",
            _impressao_da_credencial(),
        )
        return JSONResponse({"status": "forbidden"}, status_code=403)

    if not settings.ready:
        log.error("mensagem descartada: configuração incompleta (%s)", settings.missing())
        return JSONResponse({"status": "not_configured"}, status_code=503)

    payload = _decode_body(raw_body, headers)
    if payload is None:
        return JSONResponse({"status": "ignored"}, status_code=200)

    for message in channel.parse_webhook(payload):
        background.add_task(_safe_handle, message)

    # Confirma na hora; o trabalho pesado corre depois da resposta.
    return JSONResponse({"status": "ok"}, status_code=200)


def _decode_body(raw_body: bytes, headers: dict) -> dict | None:
    content_type = headers.get("content-type", "")

    if settings.provider == PROVIDER_TWILIO or "application/x-www-form-urlencoded" in content_type:
        try:
            return dict(parse_qsl(raw_body.decode("utf-8"), keep_blank_values=True))
        except UnicodeDecodeError:
            log.warning("corpo do webhook não é utf-8; descartando")
            return None

    try:
        return json.loads(raw_body or b"{}")
    except (ValueError, UnicodeDecodeError):
        log.warning("corpo do webhook não é JSON válido; descartando")
        return None


def _safe_handle(message) -> None:
    """Uma exceção aqui roda fora do ciclo da requisição e sumiria calada."""
    try:
        pipeline.handle(message)
    except Exception as exc:  # noqa: BLE001
        log.exception("falha ao processar mensagem: %s", exc)
