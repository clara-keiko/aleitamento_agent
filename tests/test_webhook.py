"""Testes do endpoint. Recarrega o main com env controlado."""

import hashlib
import hmac
import importlib
import json

import pytest
from fastapi.testclient import TestClient

APP_SECRET = "segredo-do-app"
VERIFY_TOKEN = "token-de-verificacao"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("WHATSAPP_PROVIDER", "meta")
    monkeypatch.setenv("APP_SECRET", APP_SECRET)
    monkeypatch.setenv("VERIFY_TOKEN", VERIFY_TOKEN)
    monkeypatch.setenv("WHATSAPP_TOKEN", "wa-token")
    monkeypatch.setenv("PHONE_NUMBER_ID", "123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-teste")
    monkeypatch.setenv("VECTOR_STORE_ID", "vs-teste")

    import app.config

    importlib.reload(app.config)
    import main

    importlib.reload(main)

    enviados: list[tuple[str, str]] = []
    main.channel.send_text = lambda to, body: enviados.append((to, body)) or True
    main.pipeline.channel = main.channel
    main.pipeline.engine.answer = lambda text, history=None: _fake_answer()

    with TestClient(main.app) as test_client:
        test_client.enviados = enviados
        yield test_client


def _fake_answer():
    from app.llm import Answer

    return Answer(text="Resposta fundamentada.", grounded=True)


def sign(body: bytes) -> dict:
    assinatura = hmac.new(APP_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return {"X-Hub-Signature-256": f"sha256={assinatura}", "Content-Type": "application/json"}


def mensagem_de_texto(texto: str = "qual a melhor posição para amamentar?") -> bytes:
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "id": "wamid.teste",
                                    "from": "5511999999999",
                                    "type": "text",
                                    "text": {"body": texto},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
    return json.dumps(payload).encode()


class TestHealth:
    def test_ok_com_tudo_configurado(self, client):
        resposta = client.get("/health")
        assert resposta.status_code == 200
        assert resposta.json()["status"] == "ok"
        assert resposta.json()["missing_env"] == []


class TestHandshake:
    def test_token_correto_devolve_o_challenge(self, client):
        resposta = client.get(
            "/webhook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": VERIFY_TOKEN,
                "hub.challenge": "12345",
            },
        )
        assert resposta.status_code == 200
        assert resposta.text == "12345"

    def test_token_errado_e_403(self, client):
        resposta = client.get(
            "/webhook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "errado",
                "hub.challenge": "12345",
            },
        )
        assert resposta.status_code == 403


class TestWebhookPost:
    def test_mensagem_assinada_e_processada(self, client):
        body = mensagem_de_texto()
        resposta = client.post("/webhook", content=body, headers=sign(body))

        assert resposta.status_code == 200
        assert any("Resposta fundamentada." in b for _, b in client.enviados)

    def test_sem_assinatura_e_403_e_nao_processa(self, client):
        body = mensagem_de_texto()
        resposta = client.post(
            "/webhook", content=body, headers={"Content-Type": "application/json"}
        )

        assert resposta.status_code == 403
        assert client.enviados == []

    def test_assinatura_forjada_e_403(self, client):
        body = mensagem_de_texto()
        resposta = client.post(
            "/webhook",
            content=body,
            headers={
                "X-Hub-Signature-256": "sha256=" + "0" * 64,
                "Content-Type": "application/json",
            },
        )
        assert resposta.status_code == 403
        assert client.enviados == []

    def test_json_invalido_nao_derruba_o_servico(self, client):
        body = b"nao sou json"
        resposta = client.post("/webhook", content=body, headers=sign(body))
        assert resposta.status_code == 200

    def test_evento_de_status_e_ignorado(self, client):
        body = json.dumps(
            {"entry": [{"changes": [{"value": {"statuses": [{"status": "read"}]}}]}]}
        ).encode()
        resposta = client.post("/webhook", content=body, headers=sign(body))

        assert resposta.status_code == 200
        assert client.enviados == []

    def test_reentrega_do_mesmo_evento_nao_duplica_resposta(self, client):
        """Cenário real: a Meta reenvia o webhook quando o 200 demora."""
        body = mensagem_de_texto()
        client.post("/webhook", content=body, headers=sign(body))
        total_apos_primeira = len(client.enviados)
        client.post("/webhook", content=body, headers=sign(body))

        assert len(client.enviados) == total_apos_primeira

    def test_emergencia_responde_sem_chamar_o_modelo(self, client):
        body = mensagem_de_texto("meu bebe nao respira")
        client.post("/webhook", content=body, headers=sign(body))

        corpos = "\n".join(b for _, b in client.enviados)
        assert "192" in corpos
        assert "Resposta fundamentada." not in corpos


class TestConfiguracaoIncompleta:
    def test_sem_openai_key_o_app_sobe_e_health_avisa(self, monkeypatch):
        """Faltando variável, o processo tem que subir para o log ser lido."""
        for nome in ["APP_SECRET", "VERIFY_TOKEN", "WHATSAPP_TOKEN", "PHONE_NUMBER_ID"]:
            monkeypatch.setenv(nome, "x")
        monkeypatch.setenv("VECTOR_STORE_ID", "vs")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        import app.config

        importlib.reload(app.config)
        import main

        importlib.reload(main)

        with TestClient(main.app) as test_client:
            # Liveness continua 200: se caísse, o healthcheck da plataforma
            # derrubaria o deploy e o log do erro nunca seria lido.
            vivo = test_client.get("/health")
            assert vivo.status_code == 200
            assert vivo.json()["status"] == "degraded"
            assert "OPENAI_API_KEY" in vivo.json()["missing_env"]

            # Readiness é quem sinaliza que não dá para atender.
            pronto = test_client.get("/health/ready")
            assert pronto.status_code == 503
            assert "OPENAI_API_KEY" in pronto.json()["missing_env"]

    def test_webhook_recusa_mensagem_sem_configuracao(self, monkeypatch):
        for nome in ["APP_SECRET", "VERIFY_TOKEN", "WHATSAPP_TOKEN", "PHONE_NUMBER_ID"]:
            monkeypatch.setenv(nome, "x")
        monkeypatch.setenv("VECTOR_STORE_ID", "vs")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        import app.config

        importlib.reload(app.config)
        import main

        importlib.reload(main)

        corpo = mensagem_de_texto()
        assinatura = hmac.new(b"x", corpo, hashlib.sha256).hexdigest()

        with TestClient(main.app) as test_client:
            resposta = test_client.post(
                "/webhook",
                content=corpo,
                headers={
                    "X-Hub-Signature-256": f"sha256={assinatura}",
                    "Content-Type": "application/json",
                },
            )
            assert resposta.status_code == 503
