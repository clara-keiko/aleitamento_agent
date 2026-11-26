"""Foto da caderneta: leitura, conferência e gravação das vacinas."""

import datetime as dt
import json
from types import SimpleNamespace

import psycopg
import pytest

from app.caderneta import LeituraCaderneta, VacinaLida, ler_caderneta, separar
from app.channels.base import IncomingMessage
from app.config import Settings
from app.cripto import Cofre
from app.llm import AssistantEngine
from app.memoria_persistente import MemoriaPersistente
from app.piloto import BRASILIA
from app.pipeline import MessagePipeline
from tests.test_pipeline import FakeChannel, FakeEngine, text_message

NUMERO = "5521981814170"
NASC = dt.date(2026, 9, 15)
HOJE = dt.date(2026, 11, 26)


def foto(id_="f1", legenda=""):
    return IncomingMessage(
        message_id=id_, sender=NUMERO, kind="image", text=legenda,
        media_id="m", media_mime="image/jpeg",
    )


class EngineComFoto(FakeEngine):
    def __init__(self, leitura):
        super().__init__()
        self.leitura = leitura
        self.fotos = []

    def ler_caderneta(self, imagem, mime="image/jpeg"):
        self.fotos.append((imagem, mime))
        return self.leitura


LEITURA_BOA = LeituraCaderneta(
    e_caderneta=True,
    vacinas=[
        VacinaLida("BCG", "dose única", dt.date(2026, 9, 15)),
        VacinaLida("Hepatite B", "1ª dose", dt.date(2026, 9, 15)),
        VacinaLida("Pentavalente", "1ª dose", dt.date(2026, 11, 16)),
        VacinaLida("Rotavírus", "1ª dose", None),
    ],
    input_tokens=1000, output_tokens=200,
)


@pytest.fixture
def memoria(banco):
    return MemoriaPersistente(banco, Cofre("chave-de-teste"))


def montar(memoria, leitura=LEITURA_BOA, ativa=True, com_bebe=True, consentiu=True):
    if consentiu:
        memoria.decidir(NUMERO, True, "w0")
        if com_bebe:
            memoria.salvar_relacao(NUMERO, "mae")
            memoria.salvar_bebe(NUMERO, NASC)
    settings = Settings(openai_api_key="k", vector_store_id="vs", memoria_ativa=ativa)
    channel = FakeChannel(media=b"\xff\xd8 jpeg")
    engine = EngineComFoto(leitura)
    pipeline = MessagePipeline(settings, channel, engine, memoria=memoria)
    if pipeline.questionario:
        pipeline.questionario._agora = lambda: dt.datetime(2026, 11, 26, 12, tzinfo=BRASILIA)
    return pipeline, channel, engine


class TestFluxo:
    def test_le_mostra_e_so_grava_depois_do_1(self, memoria, banco):
        pipeline, channel, engine = montar(memoria)
        assert pipeline.handle(foto()) == "foto_caderneta_lida"
        conferir = channel.sent[-1][1]
        assert "BCG – dose única — 15/09/2026" in conferir
        assert "Pentavalente – 1ª dose — 16/11/2026" in conferir
        assert "Não consegui ler a data de: Rotavírus – 1ª dose" in conferir
        assert "UBS" in conferir
        assert memoria.perfil(NUMERO)["bebes"][0]["vacinas"] == []

        assert pipeline.handle(text_message("1", "m2", NUMERO)) == "vacinas_anotadas"
        assert "Anotado! ✅ 3 vacinas" in channel.sent[-1][1]
        vacinas = memoria.perfil(NUMERO)["bebes"][0]["vacinas"]
        assert ("Pentavalente – 1ª dose", dt.date(2026, 11, 16)) in vacinas
        assert len(vacinas) == 3
        with psycopg.connect(banco) as con:
            origens = con.execute(
                "SELECT DISTINCT origem FROM registros_do_bebe WHERE tipo = 'vacina_tomada'"
            ).fetchall()
        assert origens == [("foto_caderneta",)]

    def test_mesma_foto_duas_vezes_nao_duplica(self, memoria):
        pipeline, channel, _ = montar(memoria)
        pipeline.handle(foto("f1"))
        pipeline.handle(text_message("1", "m2", NUMERO))
        pipeline.handle(foto("f2"))
        pipeline.handle(text_message("1", "m3", NUMERO))
        assert "0 vacinas" in channel.sent[-1][1] and "3 já estavam" in channel.sent[-1][1]
        assert len(memoria.perfil(NUMERO)["bebes"][0]["vacinas"]) == 3

    def test_2_nao_grava_nada(self, memoria):
        pipeline, channel, _ = montar(memoria)
        pipeline.handle(foto())
        assert pipeline.handle(text_message("2", "m2", NUMERO)) == "vacinas_recusadas"
        assert memoria.perfil(NUMERO)["bebes"][0]["vacinas"] == []

    def test_vacinas_entram_no_perfil_das_respostas(self, memoria):
        pipeline, _, engine = montar(memoria)
        pipeline.handle(foto())
        pipeline.handle(text_message("1", "m2", NUMERO))
        pipeline.handle(text_message("quais vacinas vêm aos 2 meses?", "m3", NUMERO))
        contexto = engine.calls[-1][1][0]["content"]
        assert "Pentavalente – 1ª dose (16/11/2026)" in contexto

    def test_sem_autorizacao(self, memoria):
        pipeline, channel, engine = montar(memoria, consentiu=False)
        assert pipeline.handle(foto()) == "foto_sem_autorizacao"
        assert "GUARDAR" in channel.sent[-1][1]
        assert engine.fotos == []

    def test_sem_bebe(self, memoria):
        pipeline, channel, engine = montar(memoria, com_bebe=False)
        assert pipeline.handle(foto()) == "foto_sem_bebe"
        assert "PERFIL" in channel.sent[-1][1]
        assert engine.fotos == []

    def test_foto_que_nao_e_caderneta(self, memoria):
        pipeline, channel, _ = montar(memoria, leitura=LeituraCaderneta(e_caderneta=False))
        assert pipeline.handle(foto()) == "foto_nao_caderneta"
        assert "192" in channel.sent[-1][1]

    def test_foto_ilegivel(self, memoria):
        ilegivel = LeituraCaderneta(e_caderneta=True, vacinas=[VacinaLida("BCG", "", None)])
        pipeline, channel, _ = montar(memoria, leitura=ilegivel)
        assert pipeline.handle(foto()) == "foto_ilegivel"

    def test_legenda_de_emergencia_vem_primeiro(self, memoria):
        pipeline, channel, engine = montar(memoria)
        assert pipeline.handle(foto(legenda="meu bebê está roxo")) == "emergencia"
        assert engine.fotos == []

    def test_memoria_desligada_continua_nao_suportado(self, memoria):
        pipeline, channel, _ = montar(memoria, ativa=False)
        assert pipeline.handle(foto()) == "nao_suportado"

    def test_conta_no_gasto(self, memoria):
        from app.piloto import RegistroDoPiloto

        pipeline, _, _ = montar(memoria)
        pipeline.piloto = RegistroDoPiloto(":memory:", {})
        pipeline.handle(foto())
        assert pipeline.piloto.gasto_do_mes() > 0


def test_separa_datas_impossiveis():
    leitura = LeituraCaderneta(e_caderneta=True, vacinas=[
        VacinaLida("BCG", "", dt.date(2026, 9, 15)),
        VacinaLida("Hepatite B", "", dt.date(2020, 1, 1)),   # antes do nascimento
        VacinaLida("VIP", "", dt.date(2027, 1, 1)),          # no futuro
    ])
    validas, duvidosas = separar(leitura, NASC, HOJE)
    assert [v.vacina for v in validas] == ["BCG"]
    assert [v.vacina for v in duvidosas] == ["Hepatite B", "VIP"]


class TestChamadaAoModelo:
    def engine(self, resposta=None, erro=None):
        chamadas = []

        def create(**kw):
            chamadas.append(kw)
            if erro:
                raise erro
            return resposta

        settings = Settings(openai_api_key="k", vector_store_id="vs")
        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        return AssistantEngine(settings, client=client), chamadas

    def test_manda_a_imagem_e_pede_json_estrito(self):
        corpo = {"e_caderneta": True, "vacinas": [
            {"vacina": "BCG", "dose": "dose única", "data": "2026-09-15"},
            {"vacina": "VIP", "dose": "1ª dose", "data": None},
        ]}
        resposta = SimpleNamespace(
            output_text=json.dumps(corpo),
            usage=SimpleNamespace(input_tokens=900, output_tokens=120),
        )
        engine, chamadas = self.engine(resposta)
        leitura = ler_caderneta(engine, b"img", "image/png")

        conteudo = chamadas[0]["input"][1]["content"]
        assert conteudo[1]["image_url"].startswith("data:image/png;base64,")
        assert chamadas[0]["text"]["format"]["strict"] is True
        assert "tools" not in chamadas[0]  # sem file_search: é só leitura
        assert leitura.e_caderneta
        assert leitura.vacinas[0].data == dt.date(2026, 9, 15)
        assert leitura.vacinas[1].data is None
        assert leitura.input_tokens == 900

    def test_erro_da_api(self):
        from openai import APIConnectionError

        engine, _ = self.engine(erro=APIConnectionError(request=None))
        assert ler_caderneta(engine, b"img", "image/jpeg").erro

    def test_json_quebrado(self):
        engine, _ = self.engine(SimpleNamespace(output_text="{quebrado", usage=None))
        assert ler_caderneta(engine, b"img", "image/jpeg").erro
