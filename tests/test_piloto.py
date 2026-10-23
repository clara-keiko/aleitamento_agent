"""Fase de testes: convite por link, limites de uso e painel.

O que importa garantir:
- quem toca no link entra sozinho, e o código tem teto de vagas;
- quem não foi convidado não conversa, mas uma emergência ainda recebe o 192;
- os limites cortam chamadas ao modelo, nunca a orientação de emergência;
- o telefone não é gravado em lugar nenhum.
"""

import base64
import datetime as dt
import importlib

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import Answer
from app.piloto import (
    AUTORIZADO,
    BRASILIA,
    CODIGO_INVALIDO,
    ESGOTADO,
    JA_AUTORIZADO,
    RegistroDoPiloto,
    custo_da_resposta,
    encontrar_codigo,
    ler_convites,
    ler_numeros,
    numero_canonico,
)
from app.pipeline import MessagePipeline
from tests.test_pipeline import FakeChannel, FakeEngine, text_message

CONVITE = "Quero participar da fase de testes do LactAI MAMAE2026"


class Relogio:
    def __init__(self, inicio: dt.datetime):
        self.agora = inicio

    def __call__(self) -> dt.datetime:
        return self.agora

    def avancar(self, **delta) -> None:
        self.agora += dt.timedelta(**delta)


def registro(convites="MAMAE2026:30", relogio=None):
    return RegistroDoPiloto(":memory:", ler_convites(convites), agora=relogio)


def build(convites="MAMAE2026:30", relogio=None, answer=None, canal="fake", **overrides):
    kwargs = {
        "openai_api_key": "k",
        "vector_store_id": "vs",
        "whatsapp_token": "t",
        "phone_number_id": "p",
        "verify_token": "v",
        "app_secret": "s",
        "modo_acesso": "convite",
    }
    kwargs.update(overrides)
    settings = Settings(**kwargs)
    channel = FakeChannel()
    channel.name = canal
    engine = FakeEngine(answer=answer)
    piloto = registro(convites, relogio)
    pipeline = MessagePipeline(settings, channel, engine, piloto=piloto)
    return pipeline, channel, engine, piloto


# ----------------------------------------------------------------------
# Peças isoladas
# ----------------------------------------------------------------------

class TestLeitura:
    def test_ler_convites(self):
        assert ler_convites("mamae2026:30, equipe:5,livre") == {
            "MAMAE2026": 30, "EQUIPE": 5, "LIVRE": 0,
        }

    def test_convite_com_limite_invalido_e_ignorado(self):
        assert ler_convites("BOM:3,RUIM:abc") == {"BOM": 3}

    def test_vazio(self):
        assert ler_convites("") == {}

    def test_codigo_como_palavra_inteira(self):
        assert encontrar_codigo(CONVITE, ["MAMAE2026"]) == "MAMAE2026"
        assert encontrar_codigo("quero participar mamae2026!", ["MAMAE2026"]) == "MAMAE2026"
        assert encontrar_codigo("MAMAE20267", ["MAMAE2026"]) is None
        assert encontrar_codigo("oi", []) is None

    def test_celular_sem_nove_vira_com_nove(self):
        assert numero_canonico("552181814170") == "5521981814170"
        assert numero_canonico("+55 (21) 98181-4170") == "5521981814170"

    def test_fixo_nao_ganha_nove(self):
        assert numero_canonico("552133334444") == "552133334444"

    def test_numeros_colados_do_whatsapp(self):
        assert ler_numeros("+55 21 98181-4170\n(21) 8181-4170, 11 91234-5678;\n\nabc") == [
            "5521981814170", "5521981814170", "5511912345678", "",
        ]

    def test_custo(self):
        # 1M de entrada a US$ 0,25 + 1M de saída a US$ 2,00
        assert custo_da_resposta(1_000_000, 1_000_000, 0.25, 2.0) == pytest.approx(2.25)


class TestRegistro:
    def test_convite_autoriza_e_conta_vaga(self):
        r = registro()
        assert r.usar_convite("5521999990000", "MAMAE2026") == AUTORIZADO
        assert r.autorizado("5521999990000")
        assert r.resumo()["convites"][0]["usados"] == 1

    def test_mesmo_numero_nao_gasta_duas_vagas(self):
        r = registro()
        r.usar_convite("5521981814170", "MAMAE2026")
        # Mesmo celular, entregue sem o 9.
        assert r.usar_convite("552181814170", "MAMAE2026") == JA_AUTORIZADO
        assert r.resumo()["convites"][0]["usados"] == 1

    def test_vagas_acabam(self):
        r = registro("MAMAE2026:2")
        assert r.usar_convite("5521900000001", "MAMAE2026") == AUTORIZADO
        assert r.usar_convite("5521900000002", "MAMAE2026") == AUTORIZADO
        assert r.usar_convite("5521900000003", "MAMAE2026") == ESGOTADO
        assert not r.autorizado("5521900000003")

    def test_sem_teto(self):
        r = registro("LIVRE")
        for i in range(50):
            assert r.usar_convite(f"55219000{i:05d}", "LIVRE") == AUTORIZADO

    def test_codigo_desconhecido(self):
        assert registro().usar_convite("5521900000001", "OUTRO") == CODIGO_INVALIDO

    def test_telefone_nao_e_gravado(self):
        r = registro()
        r.usar_convite("5521981814170", "MAMAE2026")
        r.registrar_pergunta("5521981814170", 0.001)
        r.deve_avisar_recusa("5521977776666")
        despejo = "\n".join(r._con.iterdump())
        for pedaco in ("981814170", "81814170", "977776666"):
            assert pedaco not in despejo

    def test_aumentar_vagas_preserva_quem_entrou(self, tmp_path):
        caminho = str(tmp_path / "piloto.db")
        r = RegistroDoPiloto(caminho, ler_convites("MAMAE2026:1"))
        r.usar_convite("5521900000001", "MAMAE2026")
        assert r.usar_convite("5521900000002", "MAMAE2026") == ESGOTADO

        # Novo deploy com mais vagas: mesmo arquivo, configuração nova.
        r = RegistroDoPiloto(caminho, ler_convites("MAMAE2026:5"))
        assert r.autorizado("5521900000001")
        assert r.usar_convite("5521900000002", "MAMAE2026") == AUTORIZADO

    def test_codigo_retirado_para_de_aceitar_mas_quem_entrou_fica(self, tmp_path):
        caminho = str(tmp_path / "piloto.db")
        r = RegistroDoPiloto(caminho, ler_convites("ANTIGO:30"))
        r.usar_convite("5521900000001", "ANTIGO")

        r = RegistroDoPiloto(caminho, ler_convites("NOVO:30"))
        assert r.codigos == ["NOVO"]
        assert r.autorizado("5521900000001")
        assert r.usar_convite("5521900000002", "ANTIGO") == CODIGO_INVALIDO

    def test_aviso_de_recusa_uma_vez_por_dia(self):
        relogio = Relogio(dt.datetime(2026, 10, 20, 10, 0, tzinfo=BRASILIA))
        r = registro(relogio=relogio)
        assert r.deve_avisar_recusa("5521900000009")
        relogio.avancar(hours=23)
        assert not r.deve_avisar_recusa("5521900000009")
        relogio.avancar(hours=2)
        assert r.deve_avisar_recusa("5521900000009")

    def test_dia_vira_a_meia_noite_de_brasilia(self):
        # 23h50 em Brasília já é dia seguinte em UTC.
        relogio = Relogio(dt.datetime(2026, 10, 20, 23, 50, tzinfo=BRASILIA))
        r = registro(relogio=relogio)
        r.registrar_pergunta("5521900000001", 0.0)
        assert r.perguntas_hoje("5521900000001") == 1
        relogio.avancar(minutes=20)
        assert r.perguntas_hoje("5521900000001") == 0

    def test_gasto_soma_no_mes_e_zera_no_seguinte(self):
        relogio = Relogio(dt.datetime(2026, 10, 31, 12, 0, tzinfo=BRASILIA))
        r = registro(relogio=relogio)
        r.registrar_pergunta("5521900000001", 0.5)
        r.registrar_pergunta(None, 0.25)  # protótipo web: só gasto
        assert r.gasto_do_mes() == pytest.approx(0.75)
        assert r.resumo()["perguntas_hoje"] == 1
        relogio.avancar(days=1)
        assert r.gasto_do_mes() == 0.0


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

class TestPortaoDoConvite:
    def test_link_do_convite_autoriza_sem_chamar_o_modelo(self):
        pipeline, channel, engine, piloto = build(limite_diario_por_pessoa=20)

        assert pipeline.handle(text_message(CONVITE, "m1")) == "convite_aceito"
        assert piloto.autorizado("5511999999999")
        assert engine.calls == []
        assert "fase de testes do LactAI" in channel.bodies
        assert "20 perguntas por dia" in channel.bodies
        assert "192" in channel.bodies

    def test_autorizada_conversa_normalmente_sem_repetir_boas_vindas(self):
        pipeline, channel, engine, _ = build()
        pipeline.handle(text_message(CONVITE, "m1"))
        channel.sent.clear()

        assert pipeline.handle(text_message("como melhorar a pega?", "m2")) == "respondido"
        assert len(engine.calls) == 1
        assert "assistente educativo" not in channel.bodies

    def test_tocar_no_link_de_novo(self):
        pipeline, channel, engine, _ = build()
        pipeline.handle(text_message(CONVITE, "m1"))
        assert pipeline.handle(text_message(CONVITE, "m2")) == "convite_repetido"
        assert engine.calls == []
        assert "já faz parte" in channel.bodies

    def test_convite_esgotado(self):
        pipeline, channel, engine, _ = build("MAMAE2026:1")
        pipeline.handle(text_message(CONVITE, "m1", sender="5521900000001"))
        desfecho = pipeline.handle(text_message(CONVITE, "m2", sender="5521900000002"))
        assert desfecho == "convite_esgotado"
        assert "preenchidas" in channel.sent[-1][1]
        assert engine.calls == []

    def test_nao_convidada_recebe_aviso_uma_vez_e_depois_silencio(self):
        pipeline, channel, engine, _ = build()
        assert pipeline.handle(text_message("oi", "m1")) == "nao_convidado"
        assert "fase de testes fechada" in channel.bodies
        assert "192" in channel.bodies

        assert pipeline.handle(text_message("como amamentar?", "m2")) == "nao_convidado_silencio"
        assert len(channel.sent) == 1
        assert engine.calls == []

    def test_emergencia_de_nao_convidada_e_respondida(self):
        pipeline, channel, engine, _ = build()
        desfecho = pipeline.handle(text_message("meu bebê não está respirando", "m1"))
        assert desfecho == "emergencia_nao_convidado"
        assert "192" in channel.bodies
        assert engine.calls == []

    def test_audio_de_nao_convidada_nao_e_transcrito(self):
        from app.channels.base import IncomingMessage

        pipeline, _, engine, _ = build()
        audio = IncomingMessage(message_id="a1", sender="5521900000001", kind="audio")
        assert pipeline.handle(audio) == "nao_convidado"
        assert engine.calls == []

    def test_lista_antiga_continua_valendo(self):
        pipeline, _, engine, _ = build(allowed_senders="5511999999999")
        assert pipeline.handle(text_message("como melhorar a pega?", "m1")) == "respondido"
        assert len(engine.calls) == 1

    def test_modo_aberto_nao_tem_portao(self):
        pipeline, _, engine, _ = build(modo_acesso="aberto")
        assert pipeline.handle(text_message("como melhorar a pega?", "m1")) == "respondido"
        assert len(engine.calls) == 1

    def test_prototipo_web_nao_tem_portao(self):
        pipeline, _, engine, _ = build(canal="web")
        assert pipeline.handle(text_message("como melhorar a pega?", "m1")) == "respondido"
        assert len(engine.calls) == 1


class TestLimites:
    def entrar(self, pipeline, sender="5511999999999"):
        pipeline.handle(text_message(CONVITE, f"entrada-{sender}", sender=sender))

    def test_limite_diario(self):
        pipeline, channel, engine, _ = build(limite_diario_por_pessoa=2)
        self.entrar(pipeline)

        assert pipeline.handle(text_message("dúvida sobre pega", "m1")) == "respondido"
        assert pipeline.handle(text_message("dúvida sobre posição", "m2")) == "respondido"
        assert pipeline.handle(text_message("dúvida sobre ordenha", "m3")) == "limite_diario"
        assert len(engine.calls) == 2
        assert "limite de 2 perguntas" in channel.sent[-1][1]

    def test_limite_volta_no_dia_seguinte(self):
        relogio = Relogio(dt.datetime(2026, 10, 20, 9, 0, tzinfo=BRASILIA))
        pipeline, _, _, _ = build(relogio=relogio, limite_diario_por_pessoa=1)
        self.entrar(pipeline)
        pipeline.handle(text_message("dúvida sobre pega", "m1"))
        assert pipeline.handle(text_message("dúvida sobre posição", "m2")) == "limite_diario"

        relogio.avancar(days=1)
        assert pipeline.handle(text_message("dúvida sobre posição", "m3")) == "respondido"

    def test_limite_e_por_pessoa(self):
        pipeline, _, _, _ = build(limite_diario_por_pessoa=1)
        self.entrar(pipeline, "5521900000001")
        self.entrar(pipeline, "5521900000002")
        pipeline.handle(text_message("dúvida sobre pega", "m1", sender="5521900000001"))
        desfecho = pipeline.handle(text_message("dúvida sobre pega", "m2", sender="5521900000002"))
        assert desfecho == "respondido"

    def test_cumprimento_e_convite_nao_contam(self):
        pipeline, _, _, piloto = build(limite_diario_por_pessoa=1)
        self.entrar(pipeline)
        pipeline.handle(text_message("oi", "m1"))
        pipeline.handle(text_message("obrigada!", "m2"))
        assert piloto.perguntas_hoje("5511999999999") == 0

    def test_falha_sem_token_nao_cobra_busca(self):
        pipeline, _, _, piloto = build(answer=Answer(text="", grounded=False, error=True))
        self.entrar(pipeline)
        assert pipeline.handle(text_message("dúvida sobre pega", "m1")) == "erro_modelo"
        assert piloto.gasto_do_mes() == 0.0

    def test_resposta_sem_fundamentacao_tambem_conta(self):
        """Ela gastou token mesmo sem ser entregue."""
        pipeline, _, _, piloto = build(
            answer=Answer(text="x", grounded=False, input_tokens=1000, output_tokens=500),
        )
        self.entrar(pipeline)
        assert pipeline.handle(text_message("dúvida sobre pega", "m1")) == "fora_de_escopo"
        assert piloto.perguntas_hoje("5511999999999") == 1
        assert piloto.gasto_do_mes() > 0

    def test_emergencia_passa_do_limite(self):
        pipeline, channel, _, _ = build(limite_diario_por_pessoa=1)
        self.entrar(pipeline)
        pipeline.handle(text_message("dúvida sobre pega", "m1"))
        desfecho = pipeline.handle(text_message("meu bebê está roxo e não respira", "m2"))
        assert desfecho == "emergencia"
        assert "192" in channel.sent[-1][1]

    def test_orcamento_do_mes(self):
        # Cada resposta: 1M de entrada (US$ 0,25) + 1M de saída (US$ 2,00).
        cara = Answer(text="ok", grounded=True, input_tokens=1_000_000, output_tokens=1_000_000)
        pipeline, channel, engine, piloto = build(answer=cara, orcamento_mensal_usd=4.0)
        self.entrar(pipeline)

        assert pipeline.handle(text_message("dúvida sobre pega", "m1")) == "respondido"
        assert pipeline.handle(text_message("dúvida sobre posição", "m2")) == "respondido"
        # 2 × (US$ 2,25 de tokens + US$ 0,0025 da busca na base)
        assert piloto.gasto_do_mes() == pytest.approx(4.505)
        assert pipeline.handle(text_message("dúvida sobre ordenha", "m3")) == "orcamento_esgotado"
        assert len(engine.calls) == 2
        assert "192" in channel.sent[-1][1]

    def test_orcamento_vale_tambem_no_prototipo_web(self):
        cara = Answer(text="ok", grounded=True, input_tokens=1_000_000, output_tokens=1_000_000)
        pipeline, _, engine, _ = build(answer=cara, orcamento_mensal_usd=1.0, canal="web")
        pipeline.handle(text_message("dúvida sobre pega", "m1"))
        assert pipeline.handle(text_message("dúvida sobre posição", "m2")) == "orcamento_esgotado"
        assert len(engine.calls) == 1

    def test_sem_registro_nada_muda(self):
        settings = Settings(openai_api_key="k", vector_store_id="vs", modo_acesso="convite")
        pipeline = MessagePipeline(settings, FakeChannel(), FakeEngine())
        assert pipeline.handle(text_message("dúvida sobre pega", "m1")) == "respondido"


# ----------------------------------------------------------------------
# Painel
# ----------------------------------------------------------------------

def _basic(senha: str) -> dict:
    return {"Authorization": "Basic " + base64.b64encode(f"admin:{senha}".encode()).decode()}


@pytest.fixture
def painel(monkeypatch):
    def montar(**env):
        base = {
            "WHATSAPP_PROVIDER": "meta", "APP_SECRET": "s", "VERIFY_TOKEN": "v",
            "WHATSAPP_TOKEN": "t", "PHONE_NUMBER_ID": "1", "OPENAI_API_KEY": "k",
            "VECTOR_STORE_ID": "vs", "DB_PATH": ":memory:",
            "MODO_ACESSO": "convite", "CONVITES": "MAMAE2026:30",
            "NUMERO_WHATSAPP": "+55 21 93618-6344",
        }
        base.update(env)
        for nome, valor in base.items():
            monkeypatch.setenv(nome, valor)

        import app.config

        importlib.reload(app.config)
        import main

        importlib.reload(main)
        return main, TestClient(main.app)

    return montar


class TestPainel:
    def test_sem_senha_configurada_o_painel_nao_existe(self, painel, monkeypatch):
        monkeypatch.delenv("ADMIN_TOKEN", raising=False)
        _, client = painel()
        assert client.get("/admin/piloto").status_code == 404

    def test_pede_senha(self, painel):
        _, client = painel(ADMIN_TOKEN="segredo-forte")
        resposta = client.get("/admin/piloto")
        assert resposta.status_code == 401
        assert resposta.headers["www-authenticate"].startswith("Basic")

    def test_senha_errada(self, painel):
        _, client = painel(ADMIN_TOKEN="segredo-forte")
        assert client.get("/admin/piloto", headers=_basic("chute")).status_code == 401

    def test_mostra_link_e_numeros(self, painel):
        main, client = painel(ADMIN_TOKEN="segredo-forte")
        main.piloto.usar_convite("5521900000001", "MAMAE2026")

        resposta = client.get("/admin/piloto", headers=_basic("segredo-forte"))
        assert resposta.status_code == 200
        assert resposta.headers["cache-control"] == "no-store"
        corpo = resposta.text
        assert "https://wa.me/5521936186344?text=" in corpo
        assert "MAMAE2026" in corpo
        assert "1 de 30" in corpo
        # Telefone de testadora nunca aparece no painel.
        assert "900000001" not in corpo

    def test_link_abre_com_a_frase_do_convite(self, painel):
        from urllib.parse import parse_qs, urlparse

        main, _ = painel(ADMIN_TOKEN="x")
        link = main._link_de_convite("MAMAE2026")
        texto = parse_qs(urlparse(link).query)["text"][0]
        assert encontrar_codigo(texto, ["MAMAE2026"]) == "MAMAE2026"


class TestLiberacaoManual:
    @pytest.fixture
    def liberar(self, painel):
        main, client = painel(ADMIN_TOKEN="segredo-forte", CONVITES="MAMAE2026:2")
        canal = FakeChannel()
        main.pipeline.channel = canal

        def enviar(numeros, **extra):
            return client.post(
                "/admin/piloto/liberar",
                data={"codigo": "MAMAE2026", "numeros": numeros},
                headers={**_basic("segredo-forte"), **extra},
            )

        return main, canal, enviar

    def test_libera_e_manda_boas_vindas(self, liberar):
        main, canal, enviar = liberar
        resposta = enviar("(21) 98181-4170")
        assert resposta.status_code == 200
        assert "liberado, boas-vindas enviadas" in resposta.text
        assert main.piloto.autorizado("552181814170")
        assert {para for para, _ in canal.sent} == {"5521981814170"}
        assert "Desculpe a demora" in canal.bodies
        assert "192" in canal.bodies

    def test_primeira_pergunta_depois_nao_repete_boas_vindas(self, liberar):
        main, canal, enviar = liberar
        enviar("21 98181-4170")
        main.pipeline.engine.answer = lambda t, history=None: Answer(text="ok", grounded=True)
        canal.sent.clear()
        # A resposta dela chega sem o 9.
        desfecho = main.pipeline.handle(text_message("como amamentar?", "m1", "552181814170"))
        assert desfecho == "respondido"
        assert "assistente educativo" not in canal.bodies

    def test_ja_liberado_nao_recebe_de_novo(self, liberar):
        main, canal, enviar = liberar
        main.piloto.usar_convite("5521981814170", "MAMAE2026")
        resposta = enviar("21981814170")
        assert "já estava liberado" in resposta.text
        assert canal.sent == []

    def test_respeita_as_vagas(self, liberar):
        _, canal, enviar = liberar
        resposta = enviar("21900000001\n21900000002\n21900000003\nlixo")
        assert resposta.text.count("boas-vindas enviadas") == 2
        assert "vagas esgotadas" in resposta.text
        assert "não parece um telefone" in resposta.text
        # O número inteiro não aparece na página, só o final.
        assert "900000003" not in resposta.text

    def test_exige_senha(self, liberar):
        main, _, _ = liberar
        resposta = TestClient(main.app).post(
            "/admin/piloto/liberar", data={"codigo": "MAMAE2026", "numeros": "21900000001"}
        )
        assert resposta.status_code == 401
        assert not main.piloto.autorizado("21900000001")

    def test_recusa_formulario_de_outro_site(self, liberar):
        main, canal, enviar = liberar
        resposta = enviar("21900000001", Origin="https://site-malicioso.example")
        assert resposta.status_code == 403
        assert not main.piloto.autorizado("5521900000001")
        assert canal.sent == []

    def test_whatsapp_desconectado_nao_libera(self, liberar, monkeypatch):
        main, canal, enviar = liberar
        monkeypatch.setattr(main.settings, "provider", "evolution")
        main.channel.estado_da_conexao = lambda: "close"
        resposta = enviar("21900000001")
        assert "WhatsApp desconectado" in resposta.text
        assert not main.piloto.autorizado("5521900000001")
        assert canal.sent == []

    def test_painel_mostra_o_formulario(self, liberar):
        main, _, _ = liberar
        resposta = TestClient(main.app).get("/admin/piloto", headers=_basic("segredo-forte"))
        assert 'action="/admin/piloto/liberar"' in resposta.text
