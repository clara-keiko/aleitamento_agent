import base64
import hashlib
import hmac

from app.channels.base import split_message
from app.channels.meta_cloud import MetaCloudChannel
from app.channels.twilio import TwilioChannel
from app.config import Settings

APP_SECRET = "segredo-do-app"


def meta_channel(**overrides):
    kwargs = {
        "app_secret": APP_SECRET,
        "verify_token": "token-de-verificacao",
        "whatsapp_token": "wa-token",
        "phone_number_id": "123",
        "openai_api_key": "k",
        "vector_store_id": "vs",
    }
    kwargs.update(overrides)
    return MetaCloudChannel(Settings(**kwargs))


def sign_meta(body: bytes, secret: str = APP_SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class TestAssinaturaMeta:
    def test_assinatura_valida(self):
        channel = meta_channel()
        body = b'{"entry":[]}'
        assert channel.verify_signature(body, {"x-hub-signature-256": sign_meta(body)}, "") is True

    def test_assinatura_de_outro_segredo_e_rejeitada(self):
        channel = meta_channel()
        body = b'{"entry":[]}'
        forjada = sign_meta(body, "segredo-errado")
        assert channel.verify_signature(body, {"x-hub-signature-256": forjada}, "") is False

    def test_corpo_adulterado_e_rejeitado(self):
        channel = meta_channel()
        assinatura = sign_meta(b'{"entry":[]}')
        assert channel.verify_signature(b'{"entry":[1]}', {"x-hub-signature-256": assinatura}, "") is False

    def test_sem_header_e_rejeitado(self):
        assert meta_channel().verify_signature(b"{}", {}, "") is False

    def test_pode_ser_desligada_para_teste_local(self):
        channel = meta_channel(require_signature=False)
        assert channel.verify_signature(b"{}", {}, "") is True

    def test_sem_app_secret_rejeita_em_vez_de_liberar(self):
        channel = meta_channel(app_secret="")
        body = b"{}"
        assert channel.verify_signature(body, {"x-hub-signature-256": sign_meta(body)}, "") is False


class TestHandshakeMeta:
    def test_token_correto(self):
        assert meta_channel().verify_challenge("subscribe", "token-de-verificacao") is True

    def test_token_errado(self):
        assert meta_channel().verify_challenge("subscribe", "outro") is False

    def test_modo_errado(self):
        assert meta_channel().verify_challenge("unsubscribe", "token-de-verificacao") is False


class TestParseMeta:
    def _envelope(self, message: dict) -> dict:
        return {"entry": [{"changes": [{"value": {"messages": [message]}}]}]}

    def test_texto(self):
        payload = self._envelope(
            {"id": "wamid.1", "from": "5511999999999", "type": "text", "text": {"body": "oi"}}
        )
        mensagens = meta_channel().parse_webhook(payload)
        assert len(mensagens) == 1
        assert mensagens[0].kind == "text"
        assert mensagens[0].text == "oi"

    def test_audio(self):
        payload = self._envelope(
            {
                "id": "wamid.2",
                "from": "5511999999999",
                "type": "audio",
                "audio": {"id": "media-1", "mime_type": "audio/ogg; codecs=opus"},
            }
        )
        mensagem = meta_channel().parse_webhook(payload)[0]
        assert mensagem.kind == "audio"
        assert mensagem.media_id == "media-1"

    def test_evento_de_status_e_ignorado(self):
        payload = {"entry": [{"changes": [{"value": {"statuses": [{"status": "delivered"}]}}]}]}
        assert meta_channel().parse_webhook(payload) == []

    def test_payload_vazio(self):
        assert meta_channel().parse_webhook({}) == []

    def test_botao_interativo_vira_texto(self):
        payload = self._envelope(
            {
                "id": "wamid.3",
                "from": "5511999999999",
                "type": "interactive",
                "interactive": {"button_reply": {"id": "b1", "title": "Pega correta"}},
            }
        )
        mensagem = meta_channel().parse_webhook(payload)[0]
        assert mensagem.kind == "text"
        assert mensagem.text == "Pega correta"

    def test_imagem_e_marcada_como_nao_suportada(self):
        payload = self._envelope(
            {"id": "wamid.4", "from": "5511999999999", "type": "image", "image": {"id": "i"}}
        )
        assert meta_channel().parse_webhook(payload)[0].kind == "unsupported"


def twilio_channel(**overrides):
    kwargs = {
        "provider": "twilio",
        "twilio_account_sid": "AC123",
        "twilio_auth_token": "auth-token",
        "twilio_whatsapp_from": "whatsapp:+14155238886",
        "openai_api_key": "k",
        "vector_store_id": "vs",
    }
    kwargs.update(overrides)
    return TwilioChannel(Settings(**kwargs))


class TestTwilio:
    def test_assinatura_valida(self):
        channel = twilio_channel()
        url = "https://exemplo.com/webhook"
        body = b"Body=oi&From=whatsapp%3A%2B5511999999999&MessageSid=SM1&NumMedia=0"

        from urllib.parse import parse_qsl

        params = dict(parse_qsl(body.decode()))
        payload = url + "".join(f"{k}{params[k]}" for k in sorted(params))
        esperado = base64.b64encode(
            hmac.new(b"auth-token", payload.encode(), hashlib.sha1).digest()
        ).decode()

        assert channel.verify_signature(body, {"x-twilio-signature": esperado}, url) is True

    def test_assinatura_invalida(self):
        channel = twilio_channel()
        body = b"Body=oi&MessageSid=SM1"
        assert channel.verify_signature(body, {"x-twilio-signature": "errada"}, "u") is False

    def test_parse_texto_normaliza_o_telefone(self):
        mensagem = twilio_channel().parse_webhook(
            {"MessageSid": "SM1", "From": "whatsapp:+5511999999999", "Body": "oi", "NumMedia": "0"}
        )[0]
        assert mensagem.sender == "5511999999999"
        assert mensagem.text == "oi"

    def test_parse_audio(self):
        mensagem = twilio_channel().parse_webhook(
            {
                "MessageSid": "SM2",
                "From": "whatsapp:+5511999999999",
                "NumMedia": "1",
                "MediaContentType0": "audio/ogg",
                "MediaUrl0": "https://api.twilio.com/media/1",
            }
        )[0]
        assert mensagem.kind == "audio"
        assert mensagem.media_url.endswith("/media/1")

    def test_imagem_nao_suportada(self):
        mensagem = twilio_channel().parse_webhook(
            {
                "MessageSid": "SM3",
                "From": "whatsapp:+5511999999999",
                "NumMedia": "1",
                "MediaContentType0": "image/jpeg",
                "MediaUrl0": "https://api.twilio.com/media/2",
            }
        )[0]
        assert mensagem.kind == "unsupported"


class TestSplitMessage:
    def test_texto_curto_fica_inteiro(self):
        assert split_message("oi") == ["oi"]

    def test_texto_vazio(self):
        assert split_message("") == []

    def test_texto_longo_e_dividido(self):
        texto = "frase muito longa. " * 500
        partes = split_message(texto, limit=1000)
        assert len(partes) > 1
        assert all(len(p) <= 1000 for p in partes)

    def test_divisao_preserva_o_conteudo(self):
        texto = "a" * 100 + "\n\n" + "b" * 100
        partes = split_message(texto, limit=120)
        assert "".join(partes).replace("\n", "") == texto.replace("\n", "")
