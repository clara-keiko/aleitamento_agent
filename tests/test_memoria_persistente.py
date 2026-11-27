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

    def test_guardar_mostra_a_pergunta_e_so_o_1_aceita(self, memoria):
        """GUARDAR sozinho não é consentimento informado."""
        memoria.decidir(NUMERO, False, "wamid.0")
        pipeline, channel, _ = montar(memoria)
        assert pipeline.handle(text_message("GUARDAR", "m1", NUMERO)) == "pergunta_de_memoria"
        assert memoria.decisao(NUMERO) is False
        pergunta = channel.sent[-1][1]
        assert "Posso guardar" in pergunta and "/privacidade" in pergunta
        assert pipeline.handle(text_message("1", "m2", NUMERO)) == "memoria_aceita"
        assert memoria.decisao(NUMERO) is True

    def test_meus_dados_e_depois_guardar(self, memoria):
        """O caminho do teste real: MEUS DADOS, GUARDAR, 1."""
        pipeline, channel, _ = montar(memoria)
        assert pipeline.handle(text_message("meus dados", "m1", NUMERO)) == "meus_dados"
        assert pipeline.handle(text_message("guardar", "m2", NUMERO)) == "pergunta_de_memoria"
        assert memoria.decisao(NUMERO) is None
        assert pipeline.handle(text_message("1", "m3", NUMERO)) == "memoria_aceita"

    def test_guardar_quando_ja_guarda(self, memoria):
        memoria.decidir(NUMERO, True, "wamid.0")
        pipeline, _, _ = montar(memoria)
        assert pipeline.handle(text_message("GUARDAR", "m1", NUMERO)) == "memoria_ja_aceita"

    def test_aceite_grava_a_versao_do_texto(self, memoria, banco):
        from app.memoria_persistente import VERSAO_MEMORIA

        pipeline, _, _ = montar(memoria)
        pipeline.handle(text_message("como melhorar a pega?", "m1", NUMERO))
        pipeline.handle(text_message("1", "m2", NUMERO))
        with psycopg.connect(banco) as con:
            versao = con.execute("SELECT versao_texto FROM consentimentos").fetchone()[0]
        assert versao == VERSAO_MEMORIA == "memoria-v2"

    @pytest.mark.parametrize("pedido,confirmacao", [
        ("sair", "CONFIRMAR"),
        ("APAGAR", "confirmar"),
        ("apagar tudo", "APAGAR"),
    ])
    def test_apagar_so_depois_de_confirmar(self, memoria, pedido, confirmacao):
        memoria.decidir(NUMERO, True, "wamid.0")
        memoria.registrar_troca(NUMERO, "p", "r")
        pipeline, channel, _ = montar(memoria)
        assert pipeline.handle(text_message(pedido, "m1", NUMERO)) == "pedido_de_exclusao"
        assert memoria.tem_dados(NUMERO)
        assert "CONFIRMAR" in channel.sent[-1][1]
        assert pipeline.handle(text_message(confirmacao, "m2", NUMERO)) == "apagou_tudo"
        assert not memoria.tem_dados(NUMERO)

    def test_so_autorizou_tambem_confirma_e_apagar_revoga(self, memoria):
        """O teste real: GUARDAR + 1, nenhuma conversa ainda, depois APAGAR."""
        memoria.decidir(NUMERO, True, "wamid.0")
        assert not memoria.tem_dados(NUMERO)
        pipeline, channel, _ = montar(memoria)
        assert pipeline.handle(text_message("APAGAR", "m1", NUMERO)) == "pedido_de_exclusao"
        assert "CONFIRMAR" in channel.sent[-1][1]
        assert memoria.decisao(NUMERO) is True
        assert pipeline.handle(text_message("CONFIRMAR", "m2", NUMERO)) == "apagou_tudo"
        assert memoria.decisao(NUMERO) is False

    def test_confirmar_sem_pedido_nao_apaga(self, memoria):
        memoria.decidir(NUMERO, True, "wamid.0")
        memoria.registrar_troca(NUMERO, "p", "r")
        pipeline, _, _ = montar(memoria)
        pipeline.handle(text_message("confirmar", "m1", NUMERO))
        assert memoria.tem_dados(NUMERO)

    def test_pedido_expira(self, memoria, monkeypatch):
        import app.pipeline as modulo

        memoria.decidir(NUMERO, True, "wamid.0")
        memoria.registrar_troca(NUMERO, "p", "r")
        pipeline, _, _ = montar(memoria)
        pipeline.handle(text_message("APAGAR", "m1", NUMERO))
        monkeypatch.setattr(modulo, "CONFIRMACAO_VALE_S", 0)
        assert pipeline.handle(text_message("confirmar", "m2", NUMERO)) != "apagou_tudo"
        assert memoria.tem_dados(NUMERO)

    def test_sem_dados_guardados_apagar_e_imediato(self, memoria):
        pipeline, _, _ = montar(memoria)
        assert pipeline.handle(text_message("APAGAR", "m1", NUMERO)) == "apagou_tudo"

    def test_pergunta_explica_a_vantagem_e_que_nao_e_obrigatorio(self):
        from app import guardrails

        texto = guardrails.memory_consent_question()
        assert "ajudar melhor" in texto
        assert "tudo bem" in texto

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


class TestNovaClienteComecaPelaPergunta:
    def test_primeira_mensagem_ja_traz_a_pergunta_antes_da_resposta(self, memoria):
        pipeline, channel, _ = montar(memoria)
        pipeline.handle(text_message("oi", "m1", NUMERO))
        primeira = channel.sent[0][1]
        assert "assistente educativo" in primeira
        assert "Posso guardar nossas conversas?" in primeira

    def test_resposta_da_primeira_pergunta_nao_e_guardada_sem_o_1(self, memoria):
        pipeline, _, _ = montar(memoria)
        pipeline.handle(text_message("como melhorar a pega?", "m1", NUMERO))
        assert not memoria.tem_dados(NUMERO)

    def test_convite_do_piloto_ja_traz_a_pergunta(self, memoria):
        from app.piloto import RegistroDoPiloto, ler_convites

        settings = Settings(
            openai_api_key="k", vector_store_id="vs", memoria_ativa=True,
            modo_acesso="convite",
        )
        channel = FakeChannel()
        piloto = RegistroDoPiloto(":memory:", ler_convites("PILOTO30:40"))
        pipeline = MessagePipeline(settings, channel, FakeEngine(), piloto=piloto, memoria=memoria)
        frase = "Quero participar da fase de testes do LactAI PILOTO30"
        assert pipeline.handle(text_message(frase, "m1", NUMERO)) == "convite_aceito"
        assert "Posso guardar nossas conversas?" in channel.sent[0][1]

    def test_emergencia_na_primeira_mensagem_nao_mistura_a_pergunta(self, memoria):
        pipeline, channel, _ = montar(memoria)
        assert pipeline.handle(text_message("meu bebê não está respirando", "m1", NUMERO)) == "emergencia"
        assert "Posso guardar" not in channel.bodies
        assert "192" in channel.sent[-1][1]
        # Na próxima resposta comum, a pergunta aparece.
        pipeline.handle(text_message("como melhorar a pega?", "m2", NUMERO))
        assert "Posso guardar" in channel.sent[-1][1]


class TestBoasVindasDepoisDeReiniciar:
    """Cada deploy zera a lista de conhecidos; a apresentação não pode
    reaparecer no meio da conversa de quem já decidiu."""

    def test_quem_ja_decidiu_nao_recebe_boas_vindas_de_novo(self, memoria):
        memoria.decidir(NUMERO, True, "w0")
        pipeline, channel, _ = montar(memoria)  # processo novo
        pipeline.handle(text_message("como melhorar a pega?", "m1", NUMERO))
        assert "assistente educativo" not in channel.bodies

    def test_quem_recusou_tambem_nao(self, memoria):
        memoria.decidir(NUMERO, False, "w0")
        pipeline, channel, _ = montar(memoria)
        pipeline.handle(text_message("como melhorar a pega?", "m1", NUMERO))
        assert "assistente educativo" not in channel.bodies

    def test_questionario_depois_de_reiniciar_sem_boas_vindas(self, memoria):
        memoria.decidir(NUMERO, True, "w0")
        pipeline, channel, _ = montar(memoria)
        pipeline.handle(text_message("PERFIL", "m1", NUMERO))
        pipeline.handle(text_message("1", "m2", NUMERO))
        assert "assistente educativo" not in channel.bodies

    def test_testadora_do_piloto_nao_recebe_de_novo(self):
        from app.piloto import RegistroDoPiloto, ler_convites

        settings = Settings(openai_api_key="k", vector_store_id="vs", modo_acesso="convite")
        piloto = RegistroDoPiloto(":memory:", ler_convites("PILOTO30:40"))
        piloto.usar_convite(NUMERO, "PILOTO30")
        channel = FakeChannel()
        pipeline = MessagePipeline(settings, channel, FakeEngine(), piloto=piloto)
        pipeline.handle(text_message("como melhorar a pega?", "m1", NUMERO))
        assert "assistente educativo" not in channel.bodies


class TestCustoDeUso:
    def test_registra_e_resume(self, memoria):
        memoria.registrar_uso(NUMERO, "pergunta", 0.004, 3000, 400)
        memoria.registrar_uso(NUMERO, "pergunta", 0.006, 4000, 500)
        memoria.registrar_uso(NUMERO, "vacina", 0.02, 5000, 600, 1)
        memoria.registrar_uso("5521900000002", "pergunta", 0.005)
        memoria.registrar_uso(None, "pergunta", 0.003)  # protótipo web
        u = memoria.resumo_de_uso(30)
        assert u["chamadas"] == 5
        assert u["custo_total_usd"] == pytest.approx(0.038)
        vacina = next(t for t in u["por_tipo"] if t["tipo"] == "vacina")
        assert vacina["medio_usd"] == pytest.approx(0.02)
        assert u["pessoas"] == 2
        assert u["max_perguntas_pessoa"] == 3
        assert u["max_custo_pessoa_usd"] == pytest.approx(0.03)

    def test_fora_da_janela_nao_conta(self, memoria, relogio):
        memoria.registrar_uso(NUMERO, "pergunta", 0.01)
        relogio.t += dt.timedelta(days=31)
        assert memoria.resumo_de_uso(30)["chamadas"] == 0

    def test_nao_guarda_telefone(self, memoria, banco):
        memoria.registrar_uso(NUMERO, "pergunta", 0.01)
        with psycopg.connect(banco) as con:
            assert "81814170" not in str(con.execute("SELECT * FROM uso_do_modelo").fetchall())

    def test_apagar_leva_o_uso(self, memoria):
        memoria.decidir(NUMERO, True, "w0")
        memoria.registrar_uso(NUMERO, "pergunta", 0.01)
        memoria.apagar_tudo(NUMERO)
        assert memoria.resumo_de_uso(30)["chamadas"] == 0

    def test_pipeline_registra_o_tipo(self, memoria):
        from app.llm import Answer

        pipeline, _, engine = montar(memoria, ativa=False)
        engine.answer_value = Answer(text="ok", grounded=True, input_tokens=1000, output_tokens=100)
        pipeline.handle(text_message("como melhorar a pega?", "m1", NUMERO))
        pipeline.handle(text_message("quais vacinas aos 2 meses?", "m2", NUMERO))
        tipos = {t["tipo"]: t["chamadas"] for t in memoria.resumo_de_uso(30)["por_tipo"]}
        assert tipos == {"pergunta": 1, "vacina": 1}

    def test_painel_mostra_o_custo(self, memoria, monkeypatch):
        import main

        memoria.registrar_uso(NUMERO, "pergunta", 0.008)
        monkeypatch.setattr(main, "memoria", memoria)
        bloco = main._bloco_de_custo()
        assert "US$ 0.0080" in bloco
        assert "150 perguntas/mês = US$ 1.20" in bloco
