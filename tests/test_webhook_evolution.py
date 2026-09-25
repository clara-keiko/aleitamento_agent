"""Webhook pelo caminho da Evolution, de ponta a ponta.

Existe pela mesma razão do equivalente da Twilio: trocar de canal não pode
afrouxar a triagem clínica. O canal é o "plano C", não oficial — mas uma mãe
que escreve "meu bebê não respira" por ele recebe a mesma resposta que
receberia pelos outros.
"""

import importlib
import json

import pytest
from fastapi.testclient import TestClient

CHAVE = "chave-da-evolution"


@pytest.fixture
def evolution(monkeypatch):
    monkeypatch.setenv("WHATSAPP_PROVIDER", "evolution")
    monkeypatch.setenv("EVOLUTION_BASE_URL", "https://evo.exemplo.com")
    monkeypatch.setenv("EVOLUTION_API_KEY", CHAVE)
    monkeypatch.setenv("EVOLUTION_INSTANCE", "lactai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-teste")
    monkeypatch.setenv("VECTOR_STORE_ID", "vs-teste")
    monkeypatch.setenv("REQUIRE_SIGNATURE", "true")
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
    with TestClient(main.app) as cliente:
        cliente.enviados = enviados
        yield cliente


def postar(cliente, texto, *, id_="3EB0A1", apikey=CHAVE, from_me=False):
    corpo = json.dumps({
        "event": "messages.upsert",
        "instance": "lactai",
        "data": {
            "key": {"remoteJid": "5521999999999@s.whatsapp.net",
                    "fromMe": from_me, "id": id_},
            "message": {"conversation": texto},
        },
    }).encode()
    cabecalhos = {"Content-Type": "application/json"}
    if apikey is not None:
        cabecalhos["apikey"] = apikey
    return cliente.post("/webhook", content=corpo, headers=cabecalhos)


class TestCaminhoFeliz:
    def test_mensagem_e_respondida(self, evolution):
        resposta = postar(evolution, "qual a melhor posição para amamentar?")

        assert resposta.status_code == 200
        assert any("Resposta fundamentada." in c for _, c in evolution.enviados)
        assert evolution.enviados[0][0] == "5521999999999"

    def test_reentrega_do_mesmo_id_nao_duplica(self, evolution):
        postar(evolution, "como aumentar a produção de leite?")
        antes = len(evolution.enviados)
        postar(evolution, "como aumentar a produção de leite?")
        assert len(evolution.enviados) == antes


class TestAutenticidade:
    def test_sem_apikey_e_403(self, evolution):
        assert postar(evolution, "oi", apikey=None).status_code == 403
        assert evolution.enviados == []

    def test_apikey_errada_e_403(self, evolution):
        assert postar(evolution, "oi", apikey="outra").status_code == 403
        assert evolution.enviados == []


class TestGarantiasClinicas:
    def test_emergencia_nao_chama_o_modelo(self, evolution):
        postar(evolution, "meu bebe nao respira", id_="3EB0-emerg")
        corpos = "\n".join(c for _, c in evolution.enviados)

        assert "192" in corpos
        assert "Resposta fundamentada." not in corpos

    def test_sinal_de_alerta_encaminha(self, evolution):
        postar(evolution, "meu bebê está com febre desde ontem", id_="3EB0-febre")
        corpos = "\n".join(c for _, c in evolution.enviados)
        assert "sinal de alerta" in corpos


class TestLacoInfinito:
    def test_eco_da_propria_resposta_nao_gera_resposta(self, evolution):
        """Sem isto, cada resposta do bot dispara a seguinte, para sempre."""
        resposta = postar(evolution, "Resposta fundamentada.", from_me=True)

        assert resposta.status_code == 200
        assert evolution.enviados == []


class TestConfiguracao:
    def test_health_reconhece_o_canal(self, evolution):
        dados = evolution.get("/health").json()
        assert dados["provider"] == "evolution"
        assert dados["whatsapp_ready"] is True
        assert dados["missing_for_whatsapp"] == []
