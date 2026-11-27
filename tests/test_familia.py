"""Modo família: até 3 pessoas no mesmo LactAI Plus."""

import datetime as dt

import psycopg
import pytest

from app import familia
from app.config import Settings
from app.cripto import Cofre
from app.lembretes import LEMBRETES
from app.memoria_persistente import MemoriaPersistente
from app.piloto import BRASILIA, RegistroDoPiloto
from app.pipeline import MessagePipeline
from tests.test_pipeline import FakeChannel, FakeEngine, text_message

TITULAR = "5521981814170"
AVO = "5521900000001"
PAI = "5521900000002"
OUTRA = "5521900000003"
AGORA = dt.datetime(2026, 11, 27, 10, 0, tzinfo=BRASILIA)


@pytest.fixture
def memoria(banco):
    return MemoriaPersistente(banco, Cofre("chave-de-teste"), agora=lambda: AGORA)


def montar(memoria, piloto=None, **extra):
    settings = Settings(
        openai_api_key="k", vector_store_id="vs", memoria_ativa=True,
        numero_whatsapp="5521923681754", **extra,
    )
    channel = FakeChannel()
    pipeline = MessagePipeline(settings, channel, FakeEngine(), memoria=memoria, piloto=piloto)
    pipeline.questionario._agora = lambda: AGORA
    return pipeline, channel


_ids = iter(range(10**6))


def diz(pipeline, numero, texto):
    return pipeline.handle(text_message(texto, f"f{next(_ids)}", numero))


def titular_com_plus(memoria):
    memoria.decidir(TITULAR, True, "w0")
    memoria.salvar_relacao(TITULAR, "mae")
    bebe = memoria.salvar_bebe(TITULAR, dt.date(2026, 9, 15))
    memoria.salvar_sexo(bebe, "feminino")
    memoria.definir_assinatura(TITULAR, "plus", dt.date(2026, 11, 20), dt.date(2026, 12, 20))
    return bebe


def convidar(pipeline, channel):
    assert diz(pipeline, TITULAR, "FAMÍLIA") == "familia_explicada"
    assert diz(pipeline, TITULAR, "1") == "familia_convite_gerado"
    link = channel.sent[-1][1]
    return familia.encontrar_codigo(link.replace("%20", " "))


class TestTitular:
    def test_explica_antes_e_so_gera_convite_com_1(self, memoria):
        titular_com_plus(memoria)
        pipeline, channel = montar(memoria)
        diz(pipeline, TITULAR, "família")
        explicacao = channel.sent[-1][1]
        assert "ver e atualizar o perfil e o histórico do bebê" in explicacao
        assert "Continua só seu" in explicacao and "cabem* *2*" not in explicacao
        assert memoria.decisao(TITULAR, familia.FAMILIA) is None
        diz(pipeline, TITULAR, "1")
        assert "https://wa.me/5521923681754?text=" in channel.sent[-1][1]
        assert "7 dias" in channel.sent[-1][1]
        assert memoria.decisao(TITULAR, familia.FAMILIA) is True

    def test_so_no_plus(self, memoria):
        memoria.decidir(TITULAR, True, "w0")
        pipeline, channel = montar(memoria)
        assert diz(pipeline, TITULAR, "FAMÍLIA") == "familia_so_no_plus"

    def test_plus_vencido_tambem_nao(self, memoria):
        titular_com_plus(memoria)
        memoria.definir_assinatura(TITULAR, "plus", dt.date(2026, 10, 1), dt.date(2026, 11, 1))
        pipeline, _ = montar(memoria)
        assert diz(pipeline, TITULAR, "FAMÍLIA") == "familia_so_no_plus"

    def test_sem_memoria_pede_guardar(self, memoria):
        pipeline, _ = montar(memoria)
        assert diz(pipeline, TITULAR, "FAMÍLIA") == "familia_sem_memoria"


class TestConvidado:
    def test_le_tudo_antes_e_entra_com_1(self, memoria):
        bebe = titular_com_plus(memoria)
        pipeline, channel = montar(memoria)
        codigo = convidar(pipeline, channel)

        assert diz(pipeline, AVO, f"Quero entrar na família no LactAI {codigo}") == "familia_convite"
        explicacao = channel.sent[-1][1]
        assert "receber os lembretes de vacina" in explicacao
        assert "Ninguém da família vê o que você me pergunta" in explicacao
        assert memoria.familia(AVO) is None

        assert diz(pipeline, AVO, "1") == "familia_entrou"
        assert "você é:" in channel.sent[-1][1]
        assert diz(pipeline, AVO, "4") == "questionario"
        # Vê o bebê da família e usa o histórico dele.
        assert memoria.perfil(AVO)["bebes"][0]["id"] == bebe
        for finalidade in ("memoria", familia.FAMILIA, LEMBRETES):
            assert memoria.decisao(AVO, finalidade) is True
        assert [m["final"] for m in memoria.familia(TITULAR)["membros"]] == ["0001"]

    def test_2_nao_entra(self, memoria):
        titular_com_plus(memoria)
        pipeline, channel = montar(memoria)
        codigo = convidar(pipeline, channel)
        diz(pipeline, AVO, codigo)
        assert diz(pipeline, AVO, "2") == "familia_recusada"
        assert memoria.familia(AVO) is None

    def test_convite_de_uso_unico(self, memoria):
        titular_com_plus(memoria)
        pipeline, channel = montar(memoria)
        codigo = convidar(pipeline, channel)
        diz(pipeline, AVO, codigo)
        diz(pipeline, AVO, "1")
        assert diz(pipeline, PAI, codigo) == "familia_convite_invalido"
        assert "já foi usado" in channel.sent[-1][1]

    def test_convite_vence_em_7_dias(self, memoria):
        titular_com_plus(memoria)
        pipeline, channel = montar(memoria)
        codigo = convidar(pipeline, channel)
        memoria._agora = lambda: AGORA + dt.timedelta(days=8)
        assert diz(pipeline, AVO, codigo) == "familia_convite_invalido"
        assert "venceu" in channel.sent[-1][1]

    def test_no_maximo_3_pessoas(self, memoria):
        titular_com_plus(memoria)
        pipeline, channel = montar(memoria)
        for pessoa in (AVO, PAI):
            codigo = convidar(pipeline, channel)
            diz(pipeline, pessoa, codigo)
            diz(pipeline, pessoa, "1")
        assert diz(pipeline, TITULAR, "FAMÍLIA") == "familia_cheia"
        assert "0001" in channel.sent[-1][1] and "0002" in channel.sent[-1][1]

    def test_link_da_familia_vale_no_modo_convite_do_piloto(self, memoria):
        titular_com_plus(memoria)
        piloto = RegistroDoPiloto(":memory:", {})
        pipeline, channel = montar(memoria, piloto=piloto, modo_acesso="convite")
        piloto.autorizar_sem_convite(TITULAR, "TESTE")
        codigo = convidar(pipeline, channel)
        assert diz(pipeline, AVO, codigo) == "familia_convite"
        diz(pipeline, AVO, "1")
        assert piloto.autorizado(AVO)
        diz(pipeline, AVO, "pular")
        assert diz(pipeline, AVO, "como melhorar a pega?") == "respondido"


class TestCompartilhado:
    def entrar(self, memoria, pipeline, channel, pessoa=AVO):
        codigo = convidar(pipeline, channel)
        diz(pipeline, pessoa, codigo)
        diz(pipeline, pessoa, "1")
        diz(pipeline, pessoa, "pular")

    def test_conversas_continuam_individuais(self, memoria):
        titular_com_plus(memoria)
        pipeline, channel = montar(memoria)
        self.entrar(memoria, pipeline, channel)
        memoria.registrar_troca(TITULAR, "estou com mastite", "resposta")
        assert memoria.historico(AVO, 10) == []

    def test_perguntas_do_plano_sao_divididas(self, memoria):
        titular_com_plus(memoria)
        pipeline, channel = montar(memoria, planos_ativos=True)
        self.entrar(memoria, pipeline, channel)
        for _ in range(3):
            memoria.registrar_uso(TITULAR, "pergunta", 0.01)
        memoria.registrar_uso(AVO, "pergunta", 0.01)
        diz(pipeline, AVO, "MEU PLANO")
        assert "LactAI Plus" in channel.sent[-1][1] and "4 de 150" in channel.sent[-1][1]

    def test_lembretes_vao_para_o_nucleo(self, memoria):
        titular_com_plus(memoria)
        memoria.decidir(TITULAR, True, "w1", LEMBRETES, "lembretes-v1")
        memoria.salvar_telefone_para_lembretes(TITULAR)
        pipeline, channel = montar(memoria)
        self.entrar(memoria, pipeline, channel)
        destinos = {i["telefone"] for i in memoria.lembretes_do_dia(dt.date(2026, 12, 15))}
        assert destinos == {TITULAR, AVO}

    def test_membro_atualiza_o_bebe_da_familia(self, memoria):
        bebe = titular_com_plus(memoria)
        pipeline, channel = montar(memoria)
        codigo = convidar(pipeline, channel)
        diz(pipeline, PAI, codigo)
        diz(pipeline, PAI, "1")
        diz(pipeline, PAI, "2")  # pai
        diz(pipeline, PAI, "ela pesou 5,3 kg")
        diz(pipeline, PAI, "1")
        assert memoria.perfil(TITULAR)["bebes"][0]["id"] == bebe
        assert memoria.perfil(TITULAR)["bebes"][0]["peso_g"][0] == 5300


class TestSaidaERemocao:
    def preparar(self, memoria):
        titular_com_plus(memoria)
        pipeline, channel = montar(memoria)
        codigo = convidar(pipeline, channel)
        diz(pipeline, AVO, codigo)
        diz(pipeline, AVO, "1")
        diz(pipeline, AVO, "pular")
        return pipeline, channel

    def test_membro_sai(self, memoria):
        pipeline, channel = self.preparar(memoria)
        assert diz(pipeline, AVO, "SAIR DA FAMÍLIA") == "familia_saiu"
        assert memoria.perfil(AVO)["bebes"] == []

    def test_titular_remove_pelo_final(self, memoria):
        pipeline, channel = self.preparar(memoria)
        assert diz(pipeline, TITULAR, "REMOVER 0001") == "familia_removeu"
        assert memoria.familia(TITULAR)["membros"] == []

    def test_membro_nao_remove_ninguem(self, memoria):
        pipeline, _ = self.preparar(memoria)
        assert diz(pipeline, AVO, "REMOVER 4170") == "familia_remocao_falhou"

    def test_titular_apagando_desfaz_a_familia(self, memoria, banco):
        pipeline, _ = self.preparar(memoria)
        memoria.apagar_tudo(TITULAR)
        with psycopg.connect(banco) as con:
            assert con.execute("SELECT count(*) FROM membros_da_familia").fetchone()[0] == 0
        assert memoria.familia(AVO) is None

    def test_telefone_do_membro_cifrado(self, memoria, banco):
        self.preparar(memoria)
        with psycopg.connect(banco) as con:
            assert "900000001" not in str(con.execute("SELECT * FROM membros_da_familia").fetchall())


def test_codigo():
    codigo = familia.novo_codigo()
    assert familia.encontrar_codigo(f"quero entrar {codigo.lower()}") == codigo
    assert familia.encontrar_codigo("quero entrar FAM-12") is None
