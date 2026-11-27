"""Planos e limites por pessoa: 10/mês no gratuito, 150/mês no LactAI Plus."""

import datetime as dt

import pytest

from app import planos
from app.config import Settings
from app.cripto import Cofre
from app.llm import Answer
from app.memoria_persistente import MemoriaPersistente
from app.piloto import BRASILIA
from app.pipeline import MessagePipeline
from tests.test_pipeline import FakeChannel, FakeEngine, text_message

NUMERO = "5521981814170"
AGORA = dt.datetime(2026, 11, 27, 10, 0, tzinfo=BRASILIA)


class TestCiclo:
    @pytest.mark.parametrize("ancora,hoje,inicio,proximo", [
        (1, dt.date(2026, 11, 27), dt.date(2026, 11, 1), dt.date(2026, 12, 1)),
        (20, dt.date(2026, 11, 27), dt.date(2026, 11, 20), dt.date(2026, 12, 20)),
        (20, dt.date(2026, 11, 5), dt.date(2026, 10, 20), dt.date(2026, 11, 20)),
        (31, dt.date(2027, 2, 28), dt.date(2027, 2, 28), dt.date(2027, 3, 31)),
        (15, dt.date(2027, 1, 3), dt.date(2026, 12, 15), dt.date(2027, 1, 15)),
    ])
    def test_ciclo(self, ancora, hoje, inicio, proximo):
        assert planos.ciclo(ancora, hoje) == (inicio, proximo)


@pytest.fixture
def memoria(banco):
    return MemoriaPersistente(banco, Cofre("chave-de-teste"), agora=lambda: AGORA)


def montar(memoria, ativos=True, **extra):
    settings = Settings(openai_api_key="k", vector_store_id="vs", planos_ativos=ativos, **extra)
    channel = FakeChannel()
    engine = FakeEngine(answer=Answer(text="Resposta.", grounded=True, input_tokens=100))
    pipeline = MessagePipeline(settings, channel, engine, memoria=memoria)
    pipeline.questionario._agora = lambda: AGORA
    return pipeline, channel, engine


def usar(memoria, n, quando=AGORA, numero=NUMERO):
    for _ in range(n):
        memoria._agora = lambda q=quando: q
        memoria.registrar_uso(numero, "pergunta", 0.005)
    memoria._agora = lambda: AGORA


def perguntar(pipeline, texto="como melhorar a pega?", id_="m"):
    return pipeline.handle(text_message(texto, f"{id_}-{texto}-{id(object())}", NUMERO))


class TestGratuito:
    def test_dez_por_mes_e_a_decima_primeira_e_barrada(self, memoria):
        usar(memoria, 10, AGORA - dt.timedelta(days=5))
        pipeline, channel, engine = montar(memoria)
        assert perguntar(pipeline) == "limite_mensal"
        aviso = channel.sent[-1][1]
        assert "10 perguntas do plano gratuito" in aviso
        assert "01/12" in aviso and "LactAI Plus" in aviso and "192" in aviso
        assert engine.calls == []

    def test_mes_anterior_nao_conta(self, memoria):
        usar(memoria, 10, dt.datetime(2026, 10, 28, tzinfo=BRASILIA))
        pipeline, _, _ = montar(memoria)
        assert perguntar(pipeline) == "respondido"

    def test_limite_diario(self, memoria):
        usar(memoria, 5)
        pipeline, channel, _ = montar(memoria)
        assert perguntar(pipeline) == "limite_diario"
        assert "5 perguntas de hoje" in channel.sent[-1][1]

    def test_avisa_quando_restam_poucas(self, memoria):
        usar(memoria, 7, AGORA - dt.timedelta(days=3))
        pipeline, channel, _ = montar(memoria)
        perguntar(pipeline)
        assert "Restam 2 perguntas do plano gratuito" in channel.sent[-1][1]

    def test_ultima_pergunta_avisada(self, memoria):
        usar(memoria, 9, AGORA - dt.timedelta(days=3))
        pipeline, channel, _ = montar(memoria)
        perguntar(pipeline)
        assert "Esta foi a última pergunta" in channel.sent[-1][1]

    def test_emergencia_passa_mesmo_sem_perguntas(self, memoria):
        usar(memoria, 10, AGORA - dt.timedelta(days=3))
        pipeline, channel, _ = montar(memoria)
        assert perguntar(pipeline, "meu bebê não está respirando") == "emergencia"

    def test_cumprimento_nao_conta(self, memoria):
        usar(memoria, 10, AGORA - dt.timedelta(days=3))
        pipeline, _, _ = montar(memoria)
        assert perguntar(pipeline, "obrigada!") == "social"

    def test_meu_plano(self, memoria):
        usar(memoria, 3, AGORA - dt.timedelta(days=3))
        pipeline, channel, _ = montar(memoria)
        assert perguntar(pipeline, "MEU PLANO") == "meu_plano"
        assert "3 de 10 perguntas" in channel.sent[-1][1]


class TestAssinante:
    def test_cento_e_cinquenta_no_ciclo_da_assinatura(self, memoria):
        memoria.definir_assinatura(NUMERO, "plus", dt.date(2026, 11, 20), dt.date(2026, 12, 20))
        usar(memoria, 25, AGORA - dt.timedelta(days=2))
        pipeline, channel, _ = montar(memoria)
        assert perguntar(pipeline) == "respondido"
        assert "Restam" not in channel.sent[-1][1]
        situacao = planos.situacao(memoria, pipeline.settings, NUMERO, AGORA)
        assert situacao.plano.mensal == 150 and situacao.renova_em == dt.date(2026, 12, 20)

    def test_uso_antes_do_inicio_do_ciclo_nao_conta(self, memoria):
        memoria.definir_assinatura(NUMERO, "plus", dt.date(2026, 11, 20), dt.date(2026, 12, 20))
        usar(memoria, 150, dt.datetime(2026, 11, 19, 12, tzinfo=BRASILIA))
        pipeline, _, _ = montar(memoria)
        assert perguntar(pipeline) == "respondido"

    def test_vencida_volta_ao_gratuito(self, memoria):
        memoria.definir_assinatura(NUMERO, "plus", dt.date(2026, 10, 1), dt.date(2026, 11, 1))
        usar(memoria, 10, AGORA - dt.timedelta(days=3))
        pipeline, _, _ = montar(memoria)
        assert perguntar(pipeline) == "limite_mensal"

    def test_limite_esgotado_do_assinante_nao_oferece_assinar(self, memoria):
        memoria.definir_assinatura(NUMERO, "plus", dt.date(2026, 11, 20), dt.date(2026, 12, 20))
        usar(memoria, 150, AGORA - dt.timedelta(days=1))
        pipeline, channel, _ = montar(memoria, limite_diario_assinante=1000)
        assert perguntar(pipeline) == "limite_mensal"
        assert "Para assinar" not in channel.sent[-1][1]
        assert "150 perguntas do LactAI Plus deste mês" in channel.sent[-1][1]


class TestDesligado:
    def test_sem_planos_vale_o_que_era(self, memoria):
        usar(memoria, 50, AGORA - dt.timedelta(days=3))
        pipeline, _, _ = montar(memoria, ativos=False)
        assert perguntar(pipeline) == "respondido"

    def test_banco_fora_nao_bloqueia(self):
        quebrada = MemoriaPersistente("postgresql://ninguem@127.0.0.1:1/nada", Cofre("x"))
        pipeline, _, _ = montar(quebrada)
        assert perguntar(pipeline) == "respondido"


class TestPainel:
    def test_ativar_e_cancelar_pelo_painel(self, memoria, monkeypatch):
        import base64
        import importlib

        from fastapi.testclient import TestClient

        for nome, valor in {
            "WHATSAPP_PROVIDER": "meta", "APP_SECRET": "s", "VERIFY_TOKEN": "v",
            "WHATSAPP_TOKEN": "t", "PHONE_NUMBER_ID": "1", "OPENAI_API_KEY": "k",
            "VECTOR_STORE_ID": "vs", "ADMIN_TOKEN": "senha",
        }.items():
            monkeypatch.setenv(nome, valor)
        import app.config

        importlib.reload(app.config)
        import main

        importlib.reload(main)
        monkeypatch.setattr(main, "memoria", memoria)
        cliente = TestClient(main.app)
        senha = {"Authorization": "Basic " + base64.b64encode(b"admin:senha").decode()}

        r = cliente.post("/admin/piloto/assinatura", headers=senha,
                         data={"numero": "(21) 98181-4170", "pago_ate": "2026-12-27", "acao": "ativar"})
        assert "LactAI Plus ativo para …4170" in r.text
        assert memoria.assinatura(NUMERO)["pago_ate"] == dt.date(2026, 12, 27)
        assert "Assinaturas ativas: <b>1</b>" in cliente.get("/admin/piloto", headers=senha).text

        cliente.post("/admin/piloto/assinatura", headers=senha,
                     data={"numero": "21981814170", "acao": "cancelar"})
        assert memoria.assinatura(NUMERO) is None
