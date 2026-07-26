"""Logging que não vaza dado pessoal.

O conteúdo trafegado aqui é dado pessoal sensível (LGPD art. 5º, II: saúde de
mãe e bebê). Log de plataforma é retido por terceiros e lido por quem tiver
acesso ao painel, então nunca gravamos telefone nem texto da mensagem em
claro. Para correlacionar eventos do mesmo usuário usamos um pseudônimo
estável: HMAC do telefone com uma chave de servidor.
"""

import hashlib
import hmac
import logging
import os

from app.config import settings

_PSEUDONYM_KEY = (
    os.getenv("PSEUDONYM_KEY", "").strip()
    or settings.app_secret
    or settings.openai_api_key
    or "chave-local-de-desenvolvimento"
).encode("utf-8")


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def pseudonymize(phone: str) -> str:
    """Identificador estável e não reversível para um telefone."""
    if not phone:
        return "anon"
    digest = hmac.new(_PSEUDONYM_KEY, phone.encode("utf-8"), hashlib.sha256)
    return f"u_{digest.hexdigest()[:12]}"


def redact(text: str, keep: int = 0) -> str:
    """Descreve um texto sem revelá-lo.

    Por padrão só o tamanho aparece no log. `keep` libera um prefixo curto,
    útil em depuração pontual — não use com dado clínico em produção.
    """
    if text is None:
        return "<none>"
    length = len(text)
    if keep <= 0:
        return f"<{length} chars>"
    return f"{text[:keep]!r}… (<{length} chars>)"
