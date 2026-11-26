"""Questionário inicial depois do "pode guardar"."""

import datetime as dt

import psycopg
import pytest

from app.config import Settings
from app.cripto import Cofre
from app.memoria_persistente import MemoriaPersistente
from app.piloto import BRASILIA
from app.pipeline import MessagePipeline
from app.questionario import (
    descrever_perfil,
    idade_em_texto,
    ler_comprimento_cm,
    ler_data,
    ler_idade,
    ler_peso_g,
)
from tests.test_pipeline import FakeChannel, FakeEngine, text_message

NUMERO = "5521981814170"
HOJE = dt.date(2026, 11, 26)


class TestLeitura:
    @pytest.mark.parametrize("texto,esperado", [
        ("03/10/2026", dt.date(2026, 10, 3)),
        ("3/10/26", dt.date(2026, 10, 3)),
        ("nasceu dia 03-10", dt.date(2026, 10, 3)),
        ("15/12", dt.date(2025, 12, 15)),  # sem ano, nascimento: passado
    ])
    def test_data_de_nascimento(self, texto, esperado):
        assert ler_data(texto, HOJE, futuro=False) == esperado

    def test_data_do_parto_sem_ano_vai_para_o_futuro(self):
        assert ler_data("15/03", HOJE, futuro=True) == dt.date(2027, 3, 15)

    def test_data_invalida(self):
        assert ler_data("31/02/2026", HOJE, futuro=False) is None
        assert ler_data("ontem", HOJE, futuro=False) is None

    @pytest.mark.parametrize("texto,gramas", [
        ("4,2", 4200), ("4.2 kg", 4200), ("4200", 4200), ("4200 g", 4200), ("3kg", 3000),
    ])
    def test_peso(self, texto, gramas):
        assert ler_peso_g(texto) == gramas

    def test_peso_absurdo(self):
        assert ler_peso_g("80 kg") is None
        assert ler_peso_g("abc") is None

    @pytest.mark.parametrize("texto,cm", [("55", 55), ("55,5 cm", 55.5), ("0,58 m", 58)])
    def test_comprimento(self, texto, cm):
        assert ler_comprimento_cm(texto) == cm

    def test_idade(self):
        assert ler_idade("31 anos") == 31
        assert ler_idade("8") is None

    def test_idade_do_bebe(self):
        assert idade_em_texto(dt.date(2026, 11, 20), HOJE) == "6 dias"
        assert idade_em_texto(dt.date(2026, 9, 15), HOJE) == "2 meses"
        assert idade_em_texto(dt.date(2026, 10, 27), HOJE) == "30 dias"


# ----------------------------------------------------------------------
# Fluxo completo no pipeline
# ----------------------------------------------------------------------

class Relogio:
    def __call__(self):
        return dt.datetime(2026, 11, 26, 15, 0, tzinfo=BRASILIA)


@pytest.fixture
def memoria(banco):
    return MemoriaPersistente(banco, Cofre("chave-de-teste"))


def montar(memoria):
    settings = Settings(openai_api_key="k", vector_store_id="vs", memoria_ativa=True)
    channel = FakeChannel()
    engine = FakeEngine()
    pipeline = MessagePipeline(settings, channel, engine, memoria=memoria)
    pipeline.questionario._agora = Relogio()
    return pipeline, channel, engine


_ids = iter(range(10**6))


def conversar(pipeline, *mensagens):
    for texto in mensagens:
        pipeline.handle(text_message(texto, f"q{next(_ids)}", NUMERO))


class TestQuestionario:
    def test_comeca_logo_depois_do_sim(self, memoria):
        pipeline, channel, _ = montar(memoria)
        conversar(pipeline, "oi", "1")
        pergunta = channel.sent[-1][1]
        assert "Você é:" in pergunta
        assert "4 – Outra pessoa\n" in pergunta
        # O aviso da LGPD vem antes da escolha, não só depois.
        assert "não guardo informações sobre o bebê" in pergunta

    def test_fluxo_completo_grava_tudo_e_resume(self, memoria, banco):
        pipeline, channel, _ = montar(memoria)
        conversar(pipeline, "oi", "1", "1", "2", "15/09/2026", "1", "5,1 kg", "58", "1", "31")

        resumo = channel.sent[-1][1]
        assert "Pronto, anotei" in resumo
        assert "15/09/2026" in resumo and "menina" in resumo and "31 anos" in resumo

        perfil = memoria.perfil(NUMERO)
        assert perfil["relacao"] == "mae"
        assert perfil["ano_nascimento_mae"] == 1995
        bebe = perfil["bebes"][0]
        assert bebe["nascimento"] == dt.date(2026, 9, 15)
        assert bebe["sexo"] == "feminino"
        assert bebe["peso_g"][0] == 5100
        assert bebe["altura_cm"][0] == 58
        assert bebe["aleitamento"][0] == "exclusivo"

    def test_perfil_entra_na_resposta(self, memoria):
        pipeline, _, engine = montar(memoria)
        conversar(pipeline, "oi", "1", "1", "2", "15/09/2026", "1", "5,1", "58", "1", "31")
        conversar(pipeline, "como saber se ela mama o suficiente?")
        contexto = engine.calls[-1][1][0]
        assert contexto["role"] == "system"
        assert "menina" in contexto["content"] and "2 meses" in contexto["content"]
        assert "só peito" in contexto["content"]

    def test_acumulativo_responder_de_novo_nao_apaga_o_peso_antigo(self, memoria, banco):
        pipeline, _, _ = montar(memoria)
        conversar(pipeline, "oi", "1", "1", "2", "15/09/2026", "1", "4,2", "pular", "pular", "pular")
        conversar(pipeline, "PERFIL", "1", "2", "15/09/2026", "1", "5,1", "pular", "pular", "pular")
        with psycopg.connect(banco) as con:
            pesos = con.execute(
                "SELECT valor_numero FROM registros_do_bebe WHERE tipo = 'peso_g' ORDER BY id"
            ).fetchall()
            bebes = con.execute("SELECT count(*) FROM bebes").fetchone()[0]
        assert [float(p[0]) for p in pesos] == [4200, 5100]
        assert bebes == 1
        assert memoria.perfil(NUMERO)["bebes"][0]["peso_g"][0] == 5100

    def test_outra_pessoa_nao_guarda_dado_do_bebe(self, memoria):
        pipeline, channel, _ = montar(memoria)
        conversar(pipeline, "oi", "1", "4")
        assert "não guardo informações sobre o bebê" in channel.sent[-1][1]
        assert not pipeline.questionario.ativo(NUMERO)
        assert memoria.perfil(NUMERO)["bebes"] == []

    def test_pular_quem_e_tambem_nao_guarda_o_bebe(self, memoria):
        pipeline, channel, _ = montar(memoria)
        conversar(pipeline, "oi", "1", "pular")
        assert not pipeline.questionario.ativo(NUMERO)
        assert "não guardo informações sobre o bebê" in channel.sent[-1][1]

    def test_gestante(self, memoria):
        pipeline, channel, _ = montar(memoria)
        conversar(pipeline, "oi", "1", "1", "1", "15/03/2027", "29")
        perfil = memoria.perfil(NUMERO)
        assert perfil["gestante"] is True
        assert perfil["data_provavel_parto"] == dt.date(2027, 3, 15)
        assert "Pronto, anotei" in channel.sent[-1][1]

    def test_parar_no_meio(self, memoria):
        pipeline, channel, _ = montar(memoria)
        conversar(pipeline, "oi", "1", "1", "parar")
        assert not pipeline.questionario.ativo(NUMERO)
        assert "PERFIL" in channel.sent[-1][1]

    def test_resposta_que_nao_serve_pede_de_novo(self, memoria):
        pipeline, channel, _ = montar(memoria)
        conversar(pipeline, "oi", "1", "1", "2", "ontem")
        assert "Não entendi" in channel.sent[-1][1]
        assert pipeline.questionario.ativo(NUMERO)

    def test_duvida_no_meio_e_respondida(self, memoria):
        pipeline, channel, engine = montar(memoria)
        conversar(pipeline, "oi", "1", "1", "meu peito está muito dolorido, o que eu faço?")
        assert engine.calls, "a dúvida deveria ir ao modelo"
        assert pipeline.questionario.ativo(NUMERO)

    def test_emergencia_no_meio_vem_antes(self, memoria):
        pipeline, channel, _ = montar(memoria)
        conversar(pipeline, "oi", "1", "1", "meu bebê está roxo")
        assert "192" in channel.sent[-1][1]

    def test_perfil_sem_autorizacao_pede_a_autorizacao(self, memoria):
        pipeline, channel, _ = montar(memoria)
        memoria.decidir(NUMERO, False, "w0")
        conversar(pipeline, "PERFIL")
        assert "Posso guardar" in channel.sent[-1][1]

    def test_meus_dados_mostra_o_perfil(self, memoria):
        pipeline, channel, _ = montar(memoria)
        conversar(pipeline, "oi", "1", "1", "2", "15/09/2026", "2", "pular", "pular", "pular", "pular")
        conversar(pipeline, "meus dados")
        assert "Perfil:" in channel.sent[-1][1]
        assert "menino" in channel.sent[-1][1]

    def test_apagar_leva_o_perfil(self, memoria):
        pipeline, _, _ = montar(memoria)
        conversar(pipeline, "oi", "1", "1", "2", "15/09/2026", "parar")
        conversar(pipeline, "APAGAR", "CONFIRMAR")
        assert memoria.perfil(NUMERO) is None


def test_descrever_perfil_vazio():
    assert descrever_perfil(None, HOJE) == ""


class TestInformacaoAvulsa:
    @pytest.mark.parametrize("texto,esperado", [
        ("tenho 33 anos", ("idade_mae", 33)),
        ("Eu tenho 28 anos", ("idade_mae", 28)),
        ("ela pesou 5,2 kg ontem", ("peso_g", 5200)),
        ("está com 6 quilos", ("peso_g", 6000)),
        ("pesou 5200 g na consulta", ("peso_g", 5200)),
        ("ele mede 60 cm", ("altura_cm", 60)),
        ("na consulta mediu 58,5 cm", ("altura_cm", 58.5)),
    ])
    def test_reconhece(self, texto, esperado):
        from app.questionario import ler_informacao_avulsa

        assert ler_informacao_avulsa(texto) == esperado

    @pytest.mark.parametrize("texto", [
        "ela pesa 5 kg, é normal?",
        "tenho 2 filhos",
        "quantos quilos ela deveria ter com 3 meses",
        "como melhorar a pega?",
    ])
    def test_ignora(self, texto):
        from app.questionario import ler_informacao_avulsa

        assert ler_informacao_avulsa(texto) is None

    def _com_perfil(self, memoria):
        pipeline, channel, engine = montar(memoria)
        conversar(pipeline, "oi", "1", "1", "2", "15/09/2026", "parar")
        return pipeline, channel, engine

    def test_tenho_33_anos_pergunta_antes_e_grava_com_1(self, memoria):
        pipeline, channel, engine = self._com_perfil(memoria)
        chamadas = len(engine.calls)
        conversar(pipeline, "tenho 33 anos")
        assert "Quer que eu atualize o seu perfil?" in channel.sent[-1][1]
        assert "33 anos" in channel.sent[-1][1]
        assert len(engine.calls) == chamadas  # não foi ao modelo
        assert memoria.perfil(NUMERO)["ano_nascimento_mae"] is None
        conversar(pipeline, "1")
        assert "Atualizado" in channel.sent[-1][1]
        assert memoria.perfil(NUMERO)["ano_nascimento_mae"] == 1993

    def test_peso_acumula_no_diario(self, memoria):
        pipeline, channel, _ = self._com_perfil(memoria)
        conversar(pipeline, "ela pesou 5,2 kg", "1")
        assert memoria.perfil(NUMERO)["bebes"][0]["peso_g"][0] == 5200

    def test_2_nao_altera(self, memoria):
        pipeline, channel, _ = self._com_perfil(memoria)
        conversar(pipeline, "tenho 33 anos", "2")
        assert "não alterei" in channel.sent[-1][1]
        assert memoria.perfil(NUMERO)["ano_nascimento_mae"] is None

    def test_pai_dizendo_a_propria_idade_nao_vira_idade_da_mae(self, memoria):
        pipeline, channel, engine = montar(memoria)
        conversar(pipeline, "oi", "1", "2", "2", "15/09/2026", "parar")
        chamadas = len(engine.calls)
        conversar(pipeline, "tenho 35 anos")
        assert len(engine.calls) == chamadas + 1  # segue para o modelo, sem oferta

    def test_sem_autorizacao_nao_oferece(self, memoria):
        memoria.decidir(NUMERO, False, "w0")
        pipeline, channel, engine = montar(memoria)
        conversar(pipeline, "tenho 33 anos")
        assert "Quer que eu atualize" not in channel.bodies


def test_prompt_proibe_fingir_que_gravou():
    from app.llm import SYSTEM_PROMPT

    assert "Nunca diga \"registrei\"" in SYSTEM_PROMPT
    assert "PERFIL" in SYSTEM_PROMPT
