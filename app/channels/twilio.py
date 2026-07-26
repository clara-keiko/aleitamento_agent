"""Canal Twilio WhatsApp.

Existe como plano B para o gargalo do número: a Twilio vende número online
(sem chip) e cuida do registro do sender junto à Meta. O webhook da Twilio é
form-encoded, e não JSON como o da Meta — por isso o parse recebe um dict já
normalizado pelo main.
"""

import base64
import hashlib
import hmac
from urllib.parse import urlencode

import requests

from app.channels.base import Channel, IncomingMessage, split_message
from app.config import Settings
from app.logging_utils import get_logger, pseudonymize

log = get_logger(__name__)

TIMEOUT = 30


def _strip_prefix(value: str) -> str:
    """"whatsapp:+5511999999999" -> "5511999999999"."""
    return value.replace("whatsapp:", "").lstrip("+").strip()


class TwilioChannel(Channel):
    name = "twilio"

    def __init__(self, settings: Settings):
        self.settings = settings

    # ------------------------------------------------------------------
    # Autenticidade
    # ------------------------------------------------------------------
    def verify_signature(self, raw_body: bytes, headers: dict, url: str) -> bool:
        """Valida X-Twilio-Signature: HMAC-SHA1 da URL + params ordenados."""
        if not self.settings.require_signature:
            return True

        token = self.settings.twilio_auth_token
        provided = headers.get("x-twilio-signature") or headers.get("X-Twilio-Signature") or ""
        if not token or not provided:
            return False

        from urllib.parse import parse_qsl

        params = dict(parse_qsl(raw_body.decode("utf-8"), keep_blank_values=True))
        payload = url + "".join(f"{k}{params[k]}" for k in sorted(params))
        digest = hmac.new(token.encode("utf-8"), payload.encode("utf-8"), hashlib.sha1).digest()
        expected = base64.b64encode(digest).decode("utf-8")
        return hmac.compare_digest(expected, provided)

    # ------------------------------------------------------------------
    # Entrada
    # ------------------------------------------------------------------
    def parse_webhook(self, payload: dict) -> list[IncomingMessage]:
        message_id = payload.get("MessageSid") or payload.get("SmsMessageSid") or ""
        sender = _strip_prefix(payload.get("From", ""))
        if not message_id or not sender:
            return []

        try:
            num_media = int(payload.get("NumMedia", "0") or 0)
        except ValueError:
            num_media = 0

        if num_media > 0:
            mime = payload.get("MediaContentType0", "")
            if mime.startswith("audio"):
                return [
                    IncomingMessage(
                        message_id=message_id,
                        sender=sender,
                        kind="audio",
                        media_url=payload.get("MediaUrl0", ""),
                        media_mime=mime,
                    )
                ]
            return [
                IncomingMessage(message_id=message_id, sender=sender, kind="unsupported")
            ]

        body = (payload.get("Body") or "").strip()
        if not body:
            return [IncomingMessage(message_id=message_id, sender=sender, kind="unsupported")]

        return [
            IncomingMessage(message_id=message_id, sender=sender, kind="text", text=body)
        ]

    # ------------------------------------------------------------------
    # Saída
    # ------------------------------------------------------------------
    def send_text(self, to: str, body: str) -> bool:
        url = (
            "https://api.twilio.com/2010-04-01/Accounts/"
            f"{self.settings.twilio_account_sid}/Messages.json"
        )
        auth = (self.settings.twilio_account_sid, self.settings.twilio_auth_token)

        for part in split_message(body):
            data = urlencode(
                {
                    "From": self.settings.twilio_whatsapp_from,
                    "To": f"whatsapp:+{to}",
                    "Body": part,
                }
            )
            try:
                response = requests.post(
                    url,
                    data=data,
                    auth=auth,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=TIMEOUT,
                )
            except requests.RequestException as exc:
                log.error("falha de rede ao enviar para %s: %s", pseudonymize(to), exc)
                return False

            if response.status_code >= 400:
                log.error(
                    "envio recusado para %s: HTTP %s %s",
                    pseudonymize(to),
                    response.status_code,
                    response.text[:500],
                )
                return False

            log.info("mensagem enviada para %s", pseudonymize(to))

        return True

    # ------------------------------------------------------------------
    # Mídia
    # ------------------------------------------------------------------
    def fetch_media(self, message: IncomingMessage) -> bytes | None:
        if not message.media_url:
            return None
        try:
            response = requests.get(
                message.media_url,
                auth=(self.settings.twilio_account_sid, self.settings.twilio_auth_token),
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            log.error("falha ao baixar mídia: %s", exc)
            return None
