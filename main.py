"""Entrypoint FastAPI do agente de aleitamento no WhatsApp.

O webhook responde 200 imediatamente e processa a mensagem em background.
Isso não é detalhe de performance: a Meta reentrega o evento quando o 200
demora, e uma chamada com file_search leva vários segundos. Processando de
forma síncrona, a mãe recebia a mesma resposta duas ou três vezes.
"""

import json
from typing import Annotated
from urllib.parse import parse_qsl

from fastapi import BackgroundTasks, FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from app.channels import build_channel
from app.channels.meta_cloud import MetaCloudChannel
from app.config import PROVIDER_TWILIO, settings
from app.llm import AssistantEngine
from app.logging_utils import configure_logging, get_logger
from app.pipeline import MessagePipeline

configure_logging()
log = get_logger(__name__)

app = FastAPI(title="Agente de Aleitamento", version="2.0.0")

channel = build_channel(settings)
engine = AssistantEngine(settings)
pipeline = MessagePipeline(settings, channel, engine)

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
    """Readiness: 503 enquanto faltar configuração para atender de verdade."""
    payload = _status_payload()
    return JSONResponse(payload, status_code=200 if not payload["missing_env"] else 503)


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
        log.warning("assinatura inválida no webhook; descartando")
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
