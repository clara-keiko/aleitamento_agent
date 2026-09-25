"""Canal Meta WhatsApp Cloud API (caminho oficial, recomendado)."""

import hashlib
import hmac

import requests

from app.channels.base import Channel, IncomingMessage, split_message
from app.config import Settings
from app.http import post_with_retry
from app.logging_utils import get_logger, pseudonymize

log = get_logger(__name__)

TIMEOUT = 30


class MetaCloudChannel(Channel):
    name = "meta"

    def __init__(self, settings: Settings):
        self.settings = settings

    # ------------------------------------------------------------------
    # Autenticidade
    # ------------------------------------------------------------------
    def verify_signature(self, raw_body: bytes, headers: dict, url: str) -> bool:
        """Valida o header X-Hub-Signature-256 assinado com o App Secret.

        Sem isso o endpoint é público: qualquer um que descubra a URL pode
        injetar mensagens falsas e gastar sua cota de OpenAI.
        """
        if not self.settings.require_signature:
            return True

        secret = self.settings.app_secret
        if not secret:
            log.error("APP_SECRET ausente com REQUIRE_SIGNATURE ligado; rejeitando webhook")
            return False

        provided = headers.get("x-hub-signature-256") or headers.get("X-Hub-Signature-256") or ""
        if not provided.startswith("sha256="):
            return False

        expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, provided[len("sha256=") :])

    def verify_challenge(self, mode: str | None, token: str | None) -> bool:
        expected = self.settings.verify_token
        if not expected or not token:
            return False
        return mode == "subscribe" and hmac.compare_digest(expected, token)

    # ------------------------------------------------------------------
    # Entrada
    # ------------------------------------------------------------------
    def parse_webhook(self, payload: dict) -> list[IncomingMessage]:
        messages: list[IncomingMessage] = []

        for entry in payload.get("entry", []) or []:
            for change in entry.get("changes", []) or []:
                value = change.get("value", {}) or {}

                # Recibos de entrega/leitura das nossas próprias mensagens.
                if "statuses" in value:
                    continue

                for raw in value.get("messages", []) or []:
                    parsed = self._parse_message(raw)
                    if parsed:
                        messages.append(parsed)

        return messages

    def _parse_message(self, raw: dict) -> IncomingMessage | None:
        message_id = raw.get("id", "")
        sender = raw.get("from", "")
        if not message_id or not sender:
            return None

        kind = raw.get("type")

        if kind == "text":
            return IncomingMessage(
                message_id=message_id,
                sender=sender,
                kind="text",
                text=(raw.get("text", {}) or {}).get("body", ""),
            )

        # No Brasil o áudio é o formato natural desse público; tratamos
        # mensagem de voz como texto depois da transcrição.
        if kind in {"audio", "voice"}:
            audio = raw.get("audio", {}) or {}
            return IncomingMessage(
                message_id=message_id,
                sender=sender,
                kind="audio",
                media_id=audio.get("id", ""),
                media_mime=audio.get("mime_type", "audio/ogg"),
            )

        # Botões e listas chegam como interactive; aproveitamos o título.
        if kind == "interactive":
            interactive = raw.get("interactive", {}) or {}
            reply = interactive.get("button_reply") or interactive.get("list_reply") or {}
            title = reply.get("title", "")
            if title:
                return IncomingMessage(
                    message_id=message_id, sender=sender, kind="text", text=title
                )

        return IncomingMessage(message_id=message_id, sender=sender, kind="unsupported")

    # ------------------------------------------------------------------
    # Saída
    # ------------------------------------------------------------------
    def send_text(self, to: str, body: str) -> bool:
        url = (
            f"https://graph.facebook.com/{self.settings.graph_version}"
            f"/{self.settings.phone_number_id}/messages"
        )
        headers = {
            "Authorization": f"Bearer {self.settings.whatsapp_token}",
            "Content-Type": "application/json",
        }

        ok = True
        for part in split_message(body):
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to,
                "type": "text",
                "text": {"body": part, "preview_url": False},
            }
            response = post_with_retry(
                url, headers=headers, json=payload, timeout=TIMEOUT
            )
            if response is None:
                log.error("envio falhou após retries para %s", pseudonymize(to))
                return False

            if response.status_code >= 400:
                # response.text traz o erro da Meta, não conteúdo do usuário.
                log.error(
                    "envio recusado para %s: HTTP %s %s",
                    pseudonymize(to),
                    response.status_code,
                    response.text[:500],
                )
                ok = False
                break

            log.info("mensagem enviada para %s", pseudonymize(to))

        return ok

    # ------------------------------------------------------------------
    # Mídia
    # ------------------------------------------------------------------
    def fetch_media(self, message: IncomingMessage) -> bytes | None:
        if not message.media_id:
            return None

        headers = {"Authorization": f"Bearer {self.settings.whatsapp_token}"}
        meta_url = f"https://graph.facebook.com/{self.settings.graph_version}/{message.media_id}"

        try:
            meta_response = requests.get(meta_url, headers=headers, timeout=TIMEOUT)
            meta_response.raise_for_status()
            download_url = meta_response.json().get("url")
            if not download_url:
                return None

            # A URL de download também exige o bearer token.
            media_response = requests.get(download_url, headers=headers, timeout=TIMEOUT)
            media_response.raise_for_status()
            return media_response.content
        except (requests.RequestException, ValueError) as exc:
            log.error("falha ao baixar mídia: %s", exc)
            return None
