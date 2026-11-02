"""Webhook pelo caminho da Twilio, de ponta a ponta.

O canal já tem teste de unidade em test_channels.py. O que falta é o trajeto
inteiro pelo main: corpo form-encoded em vez de JSON, e assinatura calculada
sobre a URL *pública* — não a que o servidor enxerga.

Esse segundo ponto é o que quebra em produção e não quebra em teste ingênuo:
atrás do proxy do Render, `request.url` chega como http:// interno, a
assinatura não bate e todo webhook vira 403 com um log que só diz
"assinatura inválida".
"""

import base64
import hashlib
import hmac
import importlib
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient

AUTH_TOKEN = "token-da-twilio"
URL_PUBLICA = "https://agente.onrender.com"


def assinar(url: str, campos: dict) -> str:
    """Reproduz o algoritmo da Twilio: HMAC-SHA1 de URL + params ordenados."""
    payload = url + "".join(f"{k}{campos[k]}" for k in sorted(campos))
    digest = hmac.new(AUTH_TOKEN.encode(), payload.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


def mensagem(texto: str = "qual a melhor posição para amamentar?",
             sid: str = "SM123") -> dict:
    return {
        "MessageSid": sid,
        "From": "whatsapp:+5511999999999",
        "To": "whatsapp:+14155238886",
        "Body": texto,
        "NumMedia": "0",
    }


def montar(monkeypatch, *, public_base_url: str = URL_PUBLICA):
    monkeypatch.setenv("WHATSAPP_PROVIDER", "twilio")
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", AUTH_TOKEN)
    monkeypatch.setenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-teste")
    monkeypatch.setenv("VECTOR_STORE_ID", "vs-teste")
    monkeypatch.setenv("REQUIRE_SIGNATURE", "true")
    if public_base_url:
        monkeypatch.setenv("PUBLIC_BASE_URL", public_base_url)
    else:
        monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    for nome in ["APP_SECRET", "VERIFY_TOKEN", "WHATSAPP_TOKEN", "PHONE_NUMBER_ID"]:
        monkeypatch.delenv(nome, raising=False)

    import app.config

    importlib.reload(app.config)
    import main

    importlib.reload(main)

    from app.llm import Answer

    enviados: list[tuple[str, str]] = []
    main.channel.send_text = lambda to, body: enviados.append((to, body)) or True
    main.pipeline.channel = main.channel
    main.pipeline.engine.answer = lambda texto, history=None: Answer(
        text="Resposta fundamentada.", grounded=True
    )
    return main, enviados


@pytest.fixture
def twilio(monkeypatch):
    main, enviados = montar(monkeypatch)
    with TestClient(main.app) as cliente:
        cliente.enviados = enviados
        yield cliente


def postar(cliente, campos: dict, *, url_da_assinatura: str = URL_PUBLICA + "/webhook"):
    corpo = urlencode(campos)
    return cliente.post(
        "/webhook",
        content=corpo,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Twilio-Signature": assinar(url_da_assinatura, campos),
        },
    )


class TestCaminhoFeliz:
    def test_mensagem_assinada_e_respondida(self, twilio):
        resposta = postar(twilio, mensagem())

        assert resposta.status_code == 200
        assert any("Resposta fundamentada." in corpo for _, corpo in twilio.enviados)

    def test_telefone_chega_sem_o_prefixo_whatsapp(self, twilio):
        postar(twilio, mensagem())
        destinatario = twilio.enviados[0][0]
        assert destinatario == "5511999999999"

    def test_reentrega_do_mesmo_sid_nao_duplica(self, twilio):
        """A Twilio também reenvia quando o 200 demora."""
        postar(twilio, mensagem())
        antes = len(twilio.enviados)
        postar(twilio, mensagem())

        assert len(twilio.enviados) == antes


class TestAssinaturaAtrasDeProxy:
    """A regressão que só aparece em produção.

    O TestClient fala com http://testserver; o Render entrega http:// interno.
    Em ambos os casos a Twilio assinou a URL https:// pública. Se o main usar
    request.url, a conta dá diferente e nada funciona.
    """

    def test_assinatura_da_url_publica_e_aceita(self, twilio):
        resposta = postar(twilio, mensagem(),
                          url_da_assinatura=URL_PUBLICA + "/webhook")
        assert resposta.status_code == 200

    def test_assinatura_da_url_interna_e_recusada(self, twilio):
        """Prova que é a URL pública que vale — não qualquer uma."""
        resposta = postar(twilio, mensagem(),
                          url_da_assinatura="http://testserver/webhook")
        assert resposta.status_code == 403
        assert twilio.enviados == []

    def test_sem_public_base_url_a_assinatura_publica_falha(self, monkeypatch):
        """Documenta o sintoma: esquecer PUBLIC_BASE_URL derruba tudo com 403.

        Serve de referência para quem for depurar "a Twilio não responde".
        """
        main, enviados = montar(monkeypatch, public_base_url="")
        with TestClient(main.app) as cliente:
            cliente.enviados = enviados
            resposta = postar(cliente, mensagem())

        assert resposta.status_code == 403
        assert enviados == []


class TestAutenticidade:
    def test_sem_assinatura_e_403(self, twilio):
        resposta = twilio.post(
            "/webhook",
            content=urlencode(mensagem()),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resposta.status_code == 403
        assert twilio.enviados == []

    def test_assinatura_de_outro_token_e_403(self, twilio):
        campos = mensagem()
        resposta = twilio.post(
            "/webhook",
            content=urlencode(campos),
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Twilio-Signature": base64.b64encode(b"x" * 20).decode(),
            },
        )
        assert resposta.status_code == 403

    def test_corpo_adulterado_apos_assinar_e_403(self, twilio):
        campos = mensagem()
        assinatura = assinar(URL_PUBLICA + "/webhook", campos)
        campos["Body"] = "pergunta trocada depois da assinatura"

        resposta = twilio.post(
            "/webhook",
            content=urlencode(campos),
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Twilio-Signature": assinatura,
            },
        )
        assert resposta.status_code == 403


class TestGarantiasClinicas:
    """As mesmas do canal da Meta. Trocar de canal não pode afrouxar triagem."""

    def test_emergencia_nao_chama_o_modelo(self, twilio):
        postar(twilio, mensagem("meu bebe nao respira", sid="SM-emerg"))
        corpos = "\n".join(c for _, c in twilio.enviados)

        assert "192" in corpos
        assert "Resposta fundamentada." not in corpos

    def test_sinal_de_alerta_encaminha(self, twilio):
        postar(twilio, mensagem("meu bebê está com febre desde ontem", sid="SM-febre"))
        corpos = "\n".join(c for _, c in twilio.enviados)

        assert "sinal de alerta" in corpos


class TestConfiguracao:
    def test_health_cobra_as_variaveis_da_twilio(self, twilio):
        dados = twilio.get("/health").json()
        assert dados["whatsapp_ready"] is True
        assert dados["missing_for_whatsapp"] == []

    def test_faltando_credencial_da_twilio_o_health_avisa(self, monkeypatch):
        monkeypatch.setenv("WHATSAPP_PROVIDER", "twilio")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-teste")
        monkeypatch.setenv("VECTOR_STORE_ID", "vs-teste")
        for nome in ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM"]:
            monkeypatch.delenv(nome, raising=False)

        import app.config

        importlib.reload(app.config)
        import main

        importlib.reload(main)

        with TestClient(main.app) as cliente:
            dados = cliente.get("/health").json()
            assert dados["whatsapp_ready"] is False
            assert "TWILIO_ACCOUNT_SID" in dados["missing_for_whatsapp"]
            # Liveness continua 200 para o log ser legível no painel.
            assert cliente.get("/health").status_code == 200
