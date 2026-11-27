"""Número oficial atendendo em paralelo à Evolution, em /webhook/meta.

Permite trocar de número sem deixar o LactAI mudo e sem aviso em massa:
os dois números atendem até todas as testadoras migrarem.
"""

import importlib

import pytest
from fastapi.testclient import TestClient

from tests.test_webhook import APP_SECRET, VERIFY_TOKEN, mensagem_de_texto, sign

BASE = {
    "WHATSAPP_PROVIDER": "evolution",
    "EVOLUTION_BASE_URL": "https://evo.exemplo.com",
    "EVOLUTION_API_KEY": "chave",
    "EVOLUTION_INSTANCE": "lactai",
    "OPENAI_API_KEY": "sk-teste",
    "VECTOR_STORE_ID": "vs-teste",
    "VIGIA_INTERVALO_S": "0",
    "DB_PATH": ":memory:",
}
META = {
    "APP_SECRET": APP_SECRET,
    "VERIFY_TOKEN": VERIFY_TOKEN,
    "WHATSAPP_TOKEN": "wa-token",
    "PHONE_NUMBER_ID": "123",
}


@pytest.fixture
def montar(monkeypatch):
    def _montar(**env):
        for nome in list(META) + ["MODO_ACESSO", "CONVITES"]:
            monkeypatch.delenv(nome, raising=False)
        for nome, valor in {**BASE, **env}.items():
            monkeypatch.setenv(nome, valor)

        import app.config

        importlib.reload(app.config)
        import main

        importlib.reload(main)
        return main, TestClient(main.app)

    return _montar


def _ligar_fakes(main):
    from app.llm import Answer

    enviados = []
    main.canal_meta.send_text = lambda to, body: enviados.append((to, body)) or True
    main.pipeline_meta.engine.answer = lambda t, history=None: Answer(text="Resposta.", grounded=True)
    return enviados


class TestViaOficialEmParalelo:
    def test_sem_credenciais_da_meta_nao_existe(self, montar):
        main, client = montar()
        assert main.canal_meta is None
        assert client.get("/webhook/meta").status_code == 404
        assert client.get("/health").json()["meta_paralela"] is False

    def test_handshake_com_evolution_como_principal(self, montar):
        _, client = montar(**META)
        resposta = client.get(
            "/webhook/meta",
            params={"hub.mode": "subscribe", "hub.verify_token": VERIFY_TOKEN, "hub.challenge": "42"},
        )
        assert resposta.status_code == 200
        assert resposta.text == "42"

    def test_handshake_com_token_errado(self, montar):
        _, client = montar(**META)
        resposta = client.get(
            "/webhook/meta",
            params={"hub.mode": "subscribe", "hub.verify_token": "chute", "hub.challenge": "42"},
        )
        assert resposta.status_code == 403

    def test_mensagem_pelo_numero_oficial_e_respondida_pela_meta(self, montar):
        main, client = montar(**META)
        enviados = _ligar_fakes(main)
        corpo = mensagem_de_texto()
        assert client.post("/webhook/meta", content=corpo, headers=sign(corpo)).status_code == 200
        assert enviados and enviados[-1] == ("5511999999999", "Resposta.")

    def test_sem_assinatura_e_recusada(self, montar):
        main, client = montar(**META)
        enviados = _ligar_fakes(main)
        corpo = mensagem_de_texto()
        resposta = client.post("/webhook/meta", content=corpo, headers={"Content-Type": "application/json"})
        assert resposta.status_code == 403
        assert enviados == []

    def test_testadora_liberada_no_numero_antigo_ja_entra_no_novo(self, montar):
        main, client = montar(**META, MODO_ACESSO="convite", CONVITES="PILOTO30:40")
        enviados = _ligar_fakes(main)
        main.piloto.usar_convite("5511999999999", "PILOTO30")
        corpo = mensagem_de_texto()
        client.post("/webhook/meta", content=corpo, headers=sign(corpo))
        assert enviados[-1][1] == "Resposta."

    def test_health_mostra_o_que_falta_para_a_meta(self, montar):
        _, client = montar(WHATSAPP_TOKEN="wa-token")
        corpo = client.get("/health").json()
        assert corpo["missing_for_meta"] == ["APP_SECRET", "PHONE_NUMBER_ID", "VERIFY_TOKEN"]
