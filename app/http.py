"""Envio HTTP com retry.

Uma falha transitória no envio (429 da Meta, 502 do proxy, timeout) hoje
significaria uma mãe sem resposta e sem saber por quê — não há tela de erro
no WhatsApp, a mensagem simplesmente não chega. Vale reenviar.

Só reenviamos o que é seguro reenviar: erro de rede, 429 e 5xx. Um 400 ou
401 é problema de configuração e não melhora com insistência.
"""

import random
import time

import requests

from app.logging_utils import get_logger

log = get_logger(__name__)

RETRIABLE_STATUS = {408, 429, 500, 502, 503, 504}
DEFAULT_ATTEMPTS = 3
DEFAULT_TIMEOUT_SECONDS = 30
BASE_DELAY_SECONDS = 1.0
MAX_DELAY_SECONDS = 8.0


def _delay_for(attempt: int, retry_after: str | None) -> float:
    """Backoff exponencial com jitter, respeitando Retry-After se vier."""
    if retry_after:
        try:
            return min(float(retry_after), MAX_DELAY_SECONDS)
        except ValueError:
            pass
    base = min(BASE_DELAY_SECONDS * (2**attempt), MAX_DELAY_SECONDS)
    # Jitter evita que várias mensagens presas reentrem todas no mesmo
    # instante. Não é uso criptográfico, então `random` serve.
    return base * (0.5 + random.random() / 2)  # noqa: S311


def post_with_retry(
    url: str,
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    sleep=time.sleep,
    **kwargs,
) -> requests.Response | None:
    """POST com reenvio em falha transitória. None se todas falharem."""
    # Rede de segurança: sem timeout, um servidor que aceita a conexão e não
    # responde prende a thread para sempre.
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT_SECONDS)
    last_response: requests.Response | None = None

    for attempt in range(attempts):
        try:
            response = requests.post(url, **kwargs)  # noqa: S113 - timeout garantido acima
        except requests.RequestException as exc:
            log.warning("falha de rede (tentativa %d/%d): %s", attempt + 1, attempts, exc)
            if attempt + 1 < attempts:
                sleep(_delay_for(attempt, None))
            continue

        if response.status_code not in RETRIABLE_STATUS:
            return response

        last_response = response
        log.warning(
            "resposta transitória HTTP %s (tentativa %d/%d)",
            response.status_code,
            attempt + 1,
            attempts,
        )
        if attempt + 1 < attempts:
            sleep(_delay_for(attempt, response.headers.get("Retry-After")))

    return last_response
