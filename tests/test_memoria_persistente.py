"""Memória persistente no Postgres: consentimento, conversa, direitos, retenção."""

import datetime as dt

import psycopg
import pytest

from app.config import Settings
from app.cripto import ChaveAusente, Cofre
from app.memoria_persistente import MemoriaPersistente
from app.pipeline import MessagePipeline
from tests.test_pipeline import FakeChannel, FakeEngine, text_message

NUMERO = "5521981814170"


class Relogio:
    def __init__(self):
        self.t = dt.datetime(2026, 11, 26, 12, 0, tzinfo=dt.UTC)

    def __call__(self):
        return self.t


@pytest.fixture
def relogio():
    return Relogio()


@pytest.fixture
def memoria(banco, relogio):
    return MemoriaPersistente(banco, Cofre("chave-de-teste"), agora=relogio)


class TestCofre:
    def test_ida_e_volta(self):
        cofre = Cofre("x")
        assert cofre.decifrar(cofre.cifrar("rachadura no mamilo")) == "rachadura no mamilo"

    def test_chave_errada_e_recusada(self):
        with pytest.raises(ValueError):
            Cofre("outra").decifrar(Cofre("x").cifrar("segredo"))

    def test_sem_chave_nao_liga(self):
        with pytest.raises(ChaveAusente):
            Cofre("")


class TestMigracoes:
    def test_aplicar_de_novo_nao_faz_nada(self, banco):
        from app.migrar import aplicar

        assert aplicar(banco) == []


class TestConsentimento:
    def test_nunca_perguntado(self, memoria):
        assert memoria.decisao(NUMERO) is None

    def test_vale_a_decisao_mais_recente(self, memoria, relogio):
        memoria.decidir(NUMERO, False, "wamid.1")
        assert memoria.decisao(NUMERO) is False
        relogio.t += dt.timedelta(days=1)
        memoria.decidir(NUMERO, True, "wamid.2")
        assert memoria.decisao(NUMERO) is True

    def test_mesmo_numero_sem_o_nove(self, memoria):
        memoria.decidir(NUMERO, True, "wamid.1")
        assert memoria.decisao("552181814170") is True


class TestConversa:
    def test_guarda_cifrado_e_devolve_em_ordem(self, memoria, banco):
        memoria.registrar_troca(NUMERO, "como melhorar a pega?", "Resposta sobre pega.")
        memoria.registrar_troca(NUMERO, "e a posição?", "Resposta sobre posição.")
        assert memoria.historico(NUMERO, 3) == [
            {"role": "assistant", "content": "Resposta sobre pega."},
            {"role": "user", "content": "e a posição?"},
            {"role": "assistant", "content": "Resposta sobre posição."},
        ]
        with psycopg.connect(banco) as con:
            despejo = str(con.execute("SELECT * FROM mensagens").fetchall())
            contas = str(con.execute("SELECT * FROM cuidadores").fetchall())
        assert "pega" not in despejo
        assert "81814170" not in contas

    def test_meus_dados(self, memoria):
        memoria.decidir(NUMERO, True, "wamid.1")
        memoria.registrar_troca(NUMERO, "p", "r")
        dados = memoria.meus_dados(NUMERO)
        assert dados["mensagens"] == 2
        assert dados["consentimentos"]["memoria"]["aceito"] is True


class TestApagar:
    def test_apaga_tudo_e_guarda_so_a_prova(self, memoria, banco):
        memoria.decidir(NUMERO, True, "wamid.1")
        memoria.registrar_troca(NUMERO, "p", "r")
        memoria.apagar_tudo(NUMERO)

        assert not memoria.tem_dados(NUMERO)
        assert memoria.historico(NUMERO, 10) == []
        assert memoria.decisao(NUMERO) is False
        with psycopg.connect(banco) as con:
            assert con.execute("SELECT count(*) FROM mensagens").fetchone()[0] == 0
            prova = con.execute("SELECT aceito, revogado_em FROM consentimentos").fetchone()
        assert prova[0] is True and prova[1] is not None


class TestRetencao:
    def test_inativa_apos_12_meses_e_apaga_24_depois(self, memoria, relogio):
        memoria.registrar_troca(NUMERO, "p", "r")

        relogio.t += dt.timedelta(days=366)
        assert memoria.aplicar_retencao(12, 24) == (1, 0)
        assert memoria.tem_dados(NUMERO)

        relogio.t += dt.timedelta(days=365)
        assert memoria.aplicar_retencao(12, 24) == (0, 0)

        relogio.t += dt.timedelta(days=400)
        assert memoria.aplicar_retencao(12, 24) == (0, 1)
        assert not memoria.tem_dados(NUMERO)

    def test_voltar_a_conversar_reativa(self, memoria, relogio, banco):
        memoria.registrar_troca(NUMERO, "p", "r")
        relogio.t += dt.timedelta(days=400)
        memoria.aplicar_retencao(12, 24)
        memoria.registrar_troca(NUMERO, "voltei", "que bom")
        with psycopg.connect(banco) as con:
            assert con.execute("SELECT status FROM cuidadores").fetchone()[0] == "ativa"
        assert len(memoria.historico(NUMERO, 10)) == 4


# ----------------------------------------------------------------------
# No pipeline
# ----------------------------------------------------------------------

def montar(memoria, ativa=True):
    settings = Settings(openai_api_key="k", vector_store_id="vs", memoria_ativa=ativa)
    channel = FakeChannel()
    engine = FakeEngine()
    return MessagePipeline(settings, channel, engine, memoria=memoria), channel, engine


class TestNoPipeline:
    def test_boas_vindas_trazem_a_pergunta(self, memoria):
        pipeline, channel, _ = montar(memoria)
        pipeline.handle(text_message("como melhorar a pega?", "m1", NUMERO))
        assert "Posso guardar nossas conversas?" in channel.bodies
        assert channel.bodies.count("Posso guardar") == 1

    def test_sim_guarda_e_lembra(self, memoria):
        pipeline, channel, engine = montar(memoria)
        pipeline.handle(text_message("como melhorar a pega?", "m1", NUMERO))
        assert pipeline.handle(text_message("1", "m2", NUMERO)) == "memoria_aceita"
        pipeline.handle(text_message("e a posição?", "m3", NUMERO))

        # Simula um deploy: pipeline novo, memória curta vazia.
        pipeline2, _, engine2 = montar(memoria)
        pipeline2.handle(text_message("e à noite?", "m4", NUMERO))
        historico = engine2.calls[-1][1]
        assert {"role": "user", "content": "e a posição?"} in historico

    def test_nao_nao_guarda(self, memoria):
        pipeline, channel, _ = montar(memoria)
        pipeline.handle(text_message("como melhorar a pega?", "m1", NUMERO))
        assert pipeline.handle(text_message("2", "m2", NUMERO)) == "memoria_recusada"
        pipeline.handle(text_message("e a posição?", "m3", NUMERO))
        assert memoria.historico(NUMERO, 10) == []
        assert "Posso guardar" not in channel.sent[-1][1]

    def test_sim_solto_nao_vira_consentimento(self, memoria):
        """Sem a pergunta no ar, "sim" é só "sim"."""
        memoria.decidir(NUMERO, False, "wamid.0")
        pipeline, _, _ = montar(memoria)
        pipeline.handle(text_message("sim", "m1", NUMERO))
        assert memoria.decisao(NUMERO) is False

    def test_guardar_muda_de_ideia(self, memoria):
        memoria.decidir(NUMERO, False, "wamid.0")
        pipeline, _, _ = montar(memoria)
        assert pipeline.handle(text_message("GUARDAR", "m1", NUMERO)) == "memoria_aceita"

    def test_sair_com_dados_pede_confirmacao_e_apagar_apaga(self, memoria):
        memoria.decidir(NUMERO, True, "wamid.0")
        memoria.registrar_troca(NUMERO, "p", "r")
        pipeline, channel, _ = montar(memoria)
        assert pipeline.handle(text_message("sair", "m1", NUMERO)) == "pedido_de_exclusao"
        assert memoria.tem_dados(NUMERO)
        assert pipeline.handle(text_message("APAGAR", "m2", NUMERO)) == "apagou_tudo"
        assert not memoria.tem_dados(NUMERO)

    def test_meus_dados(self, memoria):
        memoria.decidir(NUMERO, True, "wamid.0")
        memoria.registrar_troca(NUMERO, "p", "r")
        pipeline, channel, _ = montar(memoria)
        assert pipeline.handle(text_message("meus dados", "m1", NUMERO)) == "meus_dados"
        assert "2 mensagens" in channel.sent[-1][1]

    def test_desligada_nao_pergunta_nem_guarda(self, memoria):
        pipeline, channel, _ = montar(memoria, ativa=False)
        pipeline.handle(text_message("como melhorar a pega?", "m1", NUMERO))
        assert "Posso guardar" not in channel.bodies
        assert not memoria.tem_dados(NUMERO)

    def test_banco_fora_do_ar_nao_impede_resposta(self):
        quebrada = MemoriaPersistente("postgresql://ninguem@127.0.0.1:1/nada", Cofre("x"))
        pipeline, channel, _ = montar(quebrada)
        assert pipeline.handle(text_message("como melhorar a pega?", "m1", NUMERO)) == "respondido"
        assert "Resposta educativa." in channel.bodies
