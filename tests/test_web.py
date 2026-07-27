"""Testes do protótipo web.

O ponto crítico não é a interface — é que o canal web passe *pelo mesmo*
pipeline do WhatsApp. Se divergir, a validação clínica feita no protótipo
não vale para o que a mãe recebe.
"""

import importlib

import pytest
from fastapi.testclient import TestClient

from app.channels.web import WebChannel


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("WHATSAPP_PROVIDER", "meta")
    monkeypatch.setenv("APP_SECRET", "s")
    monkeypatch.setenv("VERIFY_TOKEN", "v")
    monkeypatch.setenv("WHATSAPP_TOKEN", "t")
    monkeypatch.setenv("PHONE_NUMBER_ID", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-teste")
    monkeypatch.setenv("VECTOR_STORE_ID", "vs-teste")
    monkeypatch.setenv("ENABLE_WEB", "true")
    monkeypatch.delenv("WEB_ACCESS_CODE", raising=False)

    import app.config

    importlib.reload(app.config)
    import main

    importlib.reload(main)

    from app.llm import Answer

    main.web_pipeline.engine.answer = lambda texto, history=None: Answer(
        text="Resposta fundamentada da base.", grounded=True
    )

    with TestClient(main.app) as test_client:
        yield test_client


def conversar(client, texto, session="s1"):
    return client.post("/api/chat", json={"session": session, "text": texto})


class TestInterface:
    def test_raiz_redireciona_para_o_chat(self, client):
        resposta = client.get("/", follow_redirects=False)
        assert resposta.status_code in (307, 302)
        assert resposta.headers["location"] == "/chat"

    def test_pagina_carrega(self, client):
        resposta = client.get("/chat")
        assert resposta.status_code == 200
        assert "Assistente de Aleitamento" in resposta.text

    def test_pagina_avisa_que_nao_e_atendimento_medico(self, client):
        texto = client.get("/chat").text
        assert "não é atendimento médico" in texto
        assert "192" in texto


class TestConversa:
    def test_primeiro_contato_recebe_boas_vindas_e_resposta(self, client):
        dados = conversar(client, "como sei se a pega está correta?").json()
        assert len(dados["replies"]) >= 2
        assert "assistente educativo" in dados["replies"][0]
        assert dados["outcome"] == "respondido"

    def test_boas_vindas_nao_se_repetem(self, client):
        conversar(client, "primeira pergunta sobre amamentação")
        segunda = conversar(client, "e como aumentar a produção?").json()
        assert not any("assistente educativo" in r for r in segunda["replies"])

    def test_sessoes_sao_isoladas(self, client):
        conversar(client, "pergunta na sessão A", session="A")
        outra = conversar(client, "pergunta na sessão B", session="B").json()
        # Sessão nova recebe boas-vindas próprias.
        assert any("assistente educativo" in r for r in outra["replies"])


class TestMesmasGarantiasDoWhatsApp:
    """Se algum destes falhar, o protótipo não representa o produto."""

    def test_emergencia_e_interceptada(self, client):
        dados = conversar(client, "meu bebe nao respira").json()
        assert dados["outcome"] == "emergencia"
        assert any("192" in r for r in dados["replies"])
        assert not any("Resposta fundamentada" in r for r in dados["replies"])

    def test_sinal_de_alerta_encaminha(self, client):
        dados = conversar(client, "meu bebê está com febre desde ontem").json()
        assert dados["outcome"] == "encaminhamento"

    def test_pergunta_sensivel_recebe_nota(self, client):
        dados = conversar(client, "posso amamentar se eu estiver com febre?").json()
        assert dados["outcome"] == "respondido_com_nota"
        assert any("informação geral" in r for r in dados["replies"])

    def test_saudacao_e_social(self, client):
        assert conversar(client, "obrigada!").json()["outcome"] == "social"

    def test_opt_out_funciona(self, client):
        conversar(client, "uma pergunta qualquer sobre amamentação")
        assert conversar(client, "SAIR").json()["outcome"] == "opt_out"

    def test_resposta_sem_fundamentacao_e_recusada(self, client, monkeypatch):
        import main
        from app.llm import Answer

        main.web_pipeline.engine.answer = lambda t, history=None: Answer(
            text="A capital da França é Paris.", grounded=False
        )
        dados = conversar(client, "qual a capital da frança?").json()
        assert dados["outcome"] == "fora_de_escopo"
        assert not any("Paris" in r for r in dados["replies"])


class TestValidacaoDeEntrada:
    def test_sem_texto_e_400(self, client):
        assert client.post("/api/chat", json={"session": "s1"}).status_code == 400

    def test_sem_sessao_e_400(self, client):
        assert client.post("/api/chat", json={"text": "oi"}).status_code == 400

    def test_json_invalido_e_400(self, client):
        resposta = client.post(
            "/api/chat", content=b"nao sou json",
            headers={"Content-Type": "application/json"},
        )
        assert resposta.status_code == 400


class TestCodigoDeAcesso:
    @pytest.fixture
    def protegido(self, monkeypatch):
        for nome, valor in [
            ("APP_SECRET", "s"), ("VERIFY_TOKEN", "v"), ("WHATSAPP_TOKEN", "t"),
            ("PHONE_NUMBER_ID", "1"), ("OPENAI_API_KEY", "k"),
            ("VECTOR_STORE_ID", "vs"), ("WEB_ACCESS_CODE", "segredo123"),
        ]:
            monkeypatch.setenv(nome, valor)

        import app.config

        importlib.reload(app.config)
        import main

        importlib.reload(main)

        from app.llm import Answer

        main.web_pipeline.engine.answer = lambda t, history=None: Answer(
            text="ok", grounded=True
        )
        with TestClient(main.app) as c:
            yield c

    def test_sem_codigo_e_401(self, protegido):
        resposta = protegido.post("/api/chat", json={"session": "s", "text": "oi"})
        assert resposta.status_code == 401

    def test_codigo_errado_e_401(self, protegido):
        resposta = protegido.post(
            "/api/chat", json={"session": "s", "text": "oi"},
            headers={"X-Access-Code": "errado"},
        )
        assert resposta.status_code == 401

    def test_codigo_certo_passa(self, protegido):
        resposta = protegido.post(
            "/api/chat", json={"session": "s", "text": "como está a pega?"},
            headers={"X-Access-Code": "segredo123"},
        )
        assert resposta.status_code == 200


class TestWebChannel:
    def test_acumula_e_drena_por_sessao(self):
        canal = WebChannel()
        canal.send_text("s1", "primeira")
        canal.send_text("s1", "segunda")
        canal.send_text("s2", "de outra sessão")

        assert canal.drain("s1") == ["primeira", "segunda"]
        assert canal.drain("s1") == []
        assert canal.drain("s2") == ["de outra sessão"]

    def test_quebra_mensagem_longa_como_no_whatsapp(self):
        """A pré-visualização precisa mostrar os mesmos balões."""
        canal = WebChannel()
        canal.send_text("s1", "frase longa. " * 500)
        assert len(canal.drain("s1")) > 1

    def test_descarta_sem_devolver(self):
        canal = WebChannel()
        canal.send_text("s1", "texto")
        canal.discard("s1")
        assert canal.drain("s1") == []

    def test_parse_ignora_payload_incompleto(self):
        canal = WebChannel()
        assert canal.parse_webhook({"session": "s"}) == []
        assert canal.parse_webhook({"text": "oi"}) == []
        assert canal.parse_webhook({"session": " ", "text": " "}) == []
