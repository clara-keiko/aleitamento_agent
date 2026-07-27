"""Configuração via variáveis de ambiente.

A aplicação nunca quebra no import por falta de variável: ela sobe, expõe
/health informando o que falta e recusa o processamento de mensagens. Isso
evita o ciclo "deploy falha e não dá para ler o log" em plataformas como
Render, e mantém o healthcheck respondendo durante uma rotação de segredo.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# Provedores de canal suportados.
PROVIDER_META = "meta"
PROVIDER_TWILIO = "twilio"


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "sim"}


@dataclass
class Settings:
    # Canal
    provider: str = field(default_factory=lambda: _env("WHATSAPP_PROVIDER", PROVIDER_META).lower())

    # Meta Cloud API
    verify_token: str = field(default_factory=lambda: _env("VERIFY_TOKEN"))
    whatsapp_token: str = field(default_factory=lambda: _env("WHATSAPP_TOKEN"))
    phone_number_id: str = field(default_factory=lambda: _env("PHONE_NUMBER_ID"))
    app_secret: str = field(default_factory=lambda: _env("APP_SECRET"))
    graph_version: str = field(default_factory=lambda: _env("GRAPH_VERSION", "v25.0"))

    # Twilio (caminho alternativo de número, ver docs/OPERACAO.md)
    twilio_account_sid: str = field(default_factory=lambda: _env("TWILIO_ACCOUNT_SID"))
    twilio_auth_token: str = field(default_factory=lambda: _env("TWILIO_AUTH_TOKEN"))
    twilio_whatsapp_from: str = field(default_factory=lambda: _env("TWILIO_WHATSAPP_FROM"))
    public_base_url: str = field(default_factory=lambda: _env("PUBLIC_BASE_URL"))

    # OpenAI
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    vector_store_id: str = field(default_factory=lambda: _env("VECTOR_STORE_ID"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    transcribe_model: str = field(
        default_factory=lambda: _env("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
    )
    # Segundos até desistir de uma chamada. O default do SDK é 600 s, o que
    # num webhook é uma thread presa por dez minutos.
    openai_timeout_seconds: int = field(
        default_factory=lambda: _env_int("OPENAI_TIMEOUT_SECONDS", 45)
    )
    # Trechos recuperados por pergunta: principal alavanca de custo e latência.
    max_retrieval_results: int = field(
        default_factory=lambda: _env_int("MAX_RETRIEVAL_RESULTS", 8)
    )
    max_output_tokens: int = field(default_factory=lambda: _env_int("MAX_OUTPUT_TOKENS", 500))

    # Protótipo web. Serve a mesma lógica do WhatsApp no navegador, para
    # validar conteúdo antes de existir número.
    enable_web: bool = field(default_factory=lambda: _env_bool("ENABLE_WEB", True))
    # Sem código, uma URL pública deixa qualquer um gastar sua cota da OpenAI.
    web_access_code: str = field(default_factory=lambda: _env("WEB_ACCESS_CODE"))

    # Comportamento
    enable_audio: bool = field(default_factory=lambda: _env_bool("ENABLE_AUDIO", True))
    # Turnos (usuário + assistente) mantidos como contexto curto por telefone.
    history_turns: int = field(default_factory=lambda: _env_int("HISTORY_TURNS", 6))
    history_ttl_seconds: int = field(default_factory=lambda: _env_int("HISTORY_TTL_SECONDS", 1800))
    # Teto simples de mensagens por telefone por janela, para conter abuso e custo.
    rate_limit_messages: int = field(default_factory=lambda: _env_int("RATE_LIMIT_MESSAGES", 20))
    rate_limit_window_seconds: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_WINDOW_SECONDS", 600)
    )
    max_audio_seconds: int = field(default_factory=lambda: _env_int("MAX_AUDIO_SECONDS", 120))
    # Em produção deixe ligado. Só desligue para testes locais sem APP_SECRET.
    require_signature: bool = field(
        default_factory=lambda: _env_bool("REQUIRE_SIGNATURE", True)
    )
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO").upper())

    def missing(self) -> list[str]:
        """Variáveis obrigatórias que faltam, dado o provedor escolhido."""
        required = {
            "OPENAI_API_KEY": self.openai_api_key,
            "VECTOR_STORE_ID": self.vector_store_id,
        }

        if self.provider == PROVIDER_TWILIO:
            required.update(
                {
                    "TWILIO_ACCOUNT_SID": self.twilio_account_sid,
                    "TWILIO_AUTH_TOKEN": self.twilio_auth_token,
                    "TWILIO_WHATSAPP_FROM": self.twilio_whatsapp_from,
                }
            )
        else:
            required.update(
                {
                    "VERIFY_TOKEN": self.verify_token,
                    "WHATSAPP_TOKEN": self.whatsapp_token,
                    "PHONE_NUMBER_ID": self.phone_number_id,
                }
            )
            if self.require_signature:
                required["APP_SECRET"] = self.app_secret

        return sorted(name for name, value in required.items() if not value)

    @property
    def ready(self) -> bool:
        return not self.missing()


settings = Settings()
