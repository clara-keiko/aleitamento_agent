"""Canal Evolution API — WhatsApp sem Meta e sem BSP.

A Evolution embrulha a biblioteca Baileys, que fala o protocolo do WhatsApp
Web multi-dispositivo. O número entra como aparelho vinculado, igual ao
WhatsApp Web no computador — por isso dispensa Cloud API, portfólio e
homologação.

⚠️ Conexão não oficial. Viola os termos do WhatsApp e o número pode ser
banido sem aviso e sem recurso. Serve para protótipo, demonstração e uso
interno. Para serviço com usuário real dependendo da resposta, use
`meta_cloud` ou `twilio`. Um número registrado na Cloud API *não* pode ser
pareado por aqui ao mesmo tempo: escolha um número que não esteja na WABA.

⚠️ O formato do webhook abaixo não foi conferido contra a documentação
corrente da Evolution — foi escrito a partir do formato conhecido da v2. Por
isso o parse aceita as variações de forma que a Baileys produz e registra em
log o que não reconhecer, em vez de descartar calado. Se a estrutura tiver
mudado, o log diz qual chave chegou e o ajuste fica localizado em
`_extrair_mensagem`.
"""

import base64
import hmac
import time

import requests

from app.channels.base import Channel, IncomingMessage, split_message
from app.config import Settings
from app.http import post_with_retry
from app.logging_utils import get_logger, pseudonymize

log = get_logger(__name__)

TIMEOUT = 30
EVENTO_DE_MENSAGEM = "messages.upsert"
SUFIXO_CONTATO = "@s.whatsapp.net"
SUFIXO_GRUPO = "@g.us"


def _numero_do_jid(jid: str) -> str:
    """\"5521999999999@s.whatsapp.net\" -> \"5521999999999\"."""
    return (jid or "").split("@", 1)[0].split(":", 1)[0].strip()


class EvolutionChannel(Channel):
    name = "evolution"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base = settings.evolution_base_url.rstrip("/")
        self.instancia = settings.evolution_instance

    # ------------------------------------------------------------------
    # Autenticidade
    # ------------------------------------------------------------------
    def verify_signature(self, raw_body: bytes, headers: dict, url: str) -> bool:
        """A Evolution não assina o corpo; ela repete a apikey no header.

        É autenticação mais fraca que a HMAC da Meta e da Twilio — quem tiver
        a chave forja qualquer mensagem. Como a URL do webhook é pública, a
        chave é a única barreira: trate-a como senha.
        """
        if not self.settings.require_signature:
            return True

        esperada = self.settings.evolution_api_key
        recebida = headers.get("apikey") or headers.get("Apikey") or ""
        if not esperada or not recebida:
            return False
        return hmac.compare_digest(esperada, recebida)

    # ------------------------------------------------------------------
    # Entrada
    # ------------------------------------------------------------------
    def parse_webhook(self, payload: dict) -> list[IncomingMessage]:
        if (payload.get("event") or "").lower() != EVENTO_DE_MENSAGEM:
            return []

        dados = payload.get("data")
        # A Evolution manda ora um objeto, ora uma lista, conforme a versão.
        eventos = dados if isinstance(dados, list) else [dados]

        mensagens = []
        for evento in eventos:
            if isinstance(evento, dict):
                mensagem = self._extrair_mensagem(evento)
                if mensagem:
                    mensagens.append(mensagem)
        return mensagens

    def _extrair_mensagem(self, evento: dict) -> IncomingMessage | None:
        chave = evento.get("key") or {}

        # Eco da própria resposta: responder a si mesmo vira laço infinito.
        if chave.get("fromMe"):
            return None

        jid = chave.get("remoteJid") or ""
        # Grupo não é atendimento individual, e bot em grupo é o padrão que
        # mais rápido leva a denúncia — logo, a banimento.
        if jid.endswith(SUFIXO_GRUPO):
            return None

        message_id = chave.get("id") or ""
        sender = _numero_do_jid(jid)
        if not message_id or not sender:
            return None

        conteudo = evento.get("message") or {}

        texto = conteudo.get("conversation")
        if not texto:
            estendida = conteudo.get("extendedTextMessage") or {}
            texto = estendida.get("text")

        if texto and texto.strip():
            return IncomingMessage(
                message_id=message_id, sender=sender, kind="text", text=texto.strip()
            )

        audio = conteudo.get("audioMessage")
        if audio:
            return IncomingMessage(
                message_id=message_id,
                sender=sender,
                kind="audio",
                media_id=message_id,
                media_mime=audio.get("mimetype", "audio/ogg"),
            )

        # Nem texto nem áudio: pode ser imagem, figurinha, contato — ou uma
        # forma que este parse não conhece. O log distingue os dois casos.
        log.info(
            "mensagem não suportada de %s; chaves recebidas: %s",
            pseudonymize(sender),
            sorted(conteudo)[:6],
        )
        return IncomingMessage(message_id=message_id, sender=sender, kind="unsupported")

    # ------------------------------------------------------------------
    # Saída
    # ------------------------------------------------------------------
    def send_text(self, to: str, body: str, sleep=time.sleep) -> bool:
        url = f"{self.base}/message/sendText/{self.instancia}"
        cabecalhos = {
            "apikey": self.settings.evolution_api_key,
            "Content-Type": "application/json",
        }

        partes = split_message(body)
        for indice, parte in enumerate(partes):
            # Rajada de mensagens no mesmo segundo é padrão de robô, e é o
            # que os detectores procuram. A pausa entre balões também deixa a
            # conversa mais natural para quem lê.
            if indice:
                sleep(self.settings.evolution_send_delay_ms / 1000)

            resposta = post_with_retry(
                url,
                json={"number": to, "text": parte},
                headers=cabecalhos,
                timeout=TIMEOUT,
            )
            if resposta is None:
                log.error("envio falhou após retries para %s", pseudonymize(to))
                return False

            if resposta.status_code >= 400:
                log.error(
                    "envio recusado para %s: HTTP %s %s",
                    pseudonymize(to),
                    resposta.status_code,
                    resposta.text[:500],
                )
                if resposta.status_code in (401, 403):
                    log.error("  → apikey errada, ou a instância não existe.")
                if resposta.status_code == 404:
                    log.error(
                        "  → instância %r não encontrada. Ela existe e está"
                        " pareada?", self.instancia,
                    )
                return False

            log.info("mensagem enviada para %s", pseudonymize(to))

        return True

    # ------------------------------------------------------------------
    # Mídia
    # ------------------------------------------------------------------
    def fetch_media(self, message: IncomingMessage) -> bytes | None:
        """A Evolution devolve a mídia em base64, não por URL."""
        if not message.media_id:
            return None

        url = f"{self.base}/chat/getBase64FromMediaMessage/{self.instancia}"
        try:
            resposta = requests.post(
                url,
                json={"message": {"key": {"id": message.media_id}}},
                headers={
                    "apikey": self.settings.evolution_api_key,
                    "Content-Type": "application/json",
                },
                timeout=TIMEOUT,
            )
            resposta.raise_for_status()
            conteudo = resposta.json().get("base64") or ""
        except requests.RequestException as exc:
            log.error("falha ao baixar mídia: %s", exc)
            return None
        except ValueError as exc:
            log.error("resposta de mídia não era JSON: %s", exc)
            return None

        if not conteudo:
            log.error("resposta de mídia veio sem o campo base64")
            return None

        try:
            return base64.b64decode(conteudo)
        except (ValueError, TypeError) as exc:
            log.error("base64 de mídia inválido: %s", exc)
            return None
