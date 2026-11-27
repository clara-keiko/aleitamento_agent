"""Lembrete de vacina no mesversário."""

import datetime as dt

import psycopg
import pytest

from app.config import Settings
from app.cripto import Cofre
from app.lembretes import LEMBRETES, LembretesDeVacina, idade_por_extenso, mesversario
from app.memoria_persistente import MemoriaPersistente
from app.piloto import BRASILIA
from app.pipeline import MessagePipeline
from tests.test_pipeline import FakeChannel, FakeEngine, text_message

NUMERO = "5521981814170"


class TestCalendario:
    @pytest.mark.parametrize("nascimento,hoje,meses", [
        (dt.date(2026, 9, 15), dt.date(2026, 11, 15), 2),
        (dt.date(2026, 9, 15), dt.date(2026, 11, 14), None),
        (dt.date(2026, 1, 31), dt.date(2026, 2, 28), 1),   # fevereiro curto
        (dt.date(2026, 1, 31), dt.date(2026, 3, 31), 2),
        (dt.date(2026, 11, 15), dt.date(2026, 11, 15), None),  # dia do nascimento
        (dt.date(2019, 1, 10), dt.date(2026, 1, 10), None),    # passou da idade
    ])
    def test_mesversario(self, nascimento, hoje, meses):
        assert mesversario(nascimento, hoje) == meses

    @pytest.mark.parametrize("meses,texto", [
        (1, "1 mês"), (2, "2 meses"), (12, "1 ano"), (15, "1 ano e 3 meses"), (48, "4 anos"),
    ])
    def test_idade_por_extenso(self, meses, texto):
        assert idade_por_extenso(meses) == texto


@pytest.fixture
def memoria(banco):
    return MemoriaPersistente(banco, Cofre("chave-de-teste"))


def preparar(memoria, numero=NUMERO, relacao="mae", nascimento=dt.date(2026, 9, 15), quer=True):
    memoria.decidir(numero, True, "w0")
    memoria.salvar_relacao(numero, relacao)
    bebe = memoria.salvar_bebe(numero, nascimento)
    memoria.salvar_sexo(bebe, "feminino")
    if quer:
        memoria.decidir(numero, True, "w1", LEMBRETES, "lembretes-v1")
        memoria.salvar_telefone_para_lembretes(numero)
    return bebe


HOJE = dt.date(2026, 11, 15)


class TestQuemRecebe:
    def test_mesversario_de_quem_pediu(self, memoria):
        bebe = preparar(memoria)
        itens = memoria.lembretes_do_dia(HOJE)
        assert itens == [{"bebe_id": bebe, "idade_meses": 2, "sexo": "feminino", "telefone": NUMERO}]

    def test_telefone_guardado_cifrado(self, memoria, banco):
        preparar(memoria)
        with psycopg.connect(banco) as con:
            assert "81814170" not in str(con.execute("SELECT * FROM cuidadores").fetchall())

    def test_nao_repete_o_mesmo_mes(self, memoria):
        bebe = preparar(memoria)
        memoria.registrar_lembrete(bebe, 2, "enviado")
        assert memoria.lembretes_do_dia(HOJE) == []

    def test_quem_nao_pediu_ou_parou(self, memoria):
        preparar(memoria, quer=False)
        assert memoria.lembretes_do_dia(HOJE) == []
        memoria.decidir(NUMERO, True, "w2", LEMBRETES, "lembretes-v1")
        memoria.salvar_telefone_para_lembretes(NUMERO)
        memoria.decidir(NUMERO, False, "w3", LEMBRETES, "lembretes-v1")
        assert memoria.lembretes_do_dia(HOJE) == []

    def test_outra_pessoa_nao_recebe(self, memoria):
        preparar(memoria, relacao="outro")
        assert memoria.lembretes_do_dia(HOJE) == []

    def test_apagar_leva_os_lembretes(self, memoria):
        preparar(memoria)
        memoria.apagar_tudo(NUMERO)
        assert memoria.lembretes_do_dia(HOJE) == []


class CanalComModelo(FakeChannel):
    name = "meta"

    def __init__(self, ok=True):
        super().__init__()
        self.ok = ok
        self.modelos = []

    def send_template(self, to, nome, idioma, parametros):
        self.modelos.append((to, nome, idioma, parametros))
        return self.ok


def servico(memoria, consultar, hora=10, ok=True):
    canal = CanalComModelo(ok)
    settings = Settings(openai_api_key="k", vector_store_id="vs")
    agora = dt.datetime(2026, 11, 15, hora, 0, tzinfo=BRASILIA)
    return LembretesDeVacina(memoria, None, canal, settings, agora=lambda: agora,
                             consultar=consultar, pausa_s=0), canal


class TestEnvio:
    def test_manda_o_modelo_aprovado_com_as_variaveis(self, memoria):
        preparar(memoria)
        lembretes, canal = servico(memoria, lambda meses: True)
        assert lembretes.rodar()["enviados"] == 1
        assert canal.modelos == [
            (NUMERO, "lembrete_vacina_mesversario", "pt_BR", ["sua bebê", "2 meses"]),
        ]
        assert lembretes.rodar()["enviados"] == 0  # não repete

    def test_mes_sem_vacina_nao_manda_e_nao_reconsulta(self, memoria):
        preparar(memoria)
        consultas = []
        lembretes, canal = servico(memoria, lambda meses: consultas.append(meses) or False)
        assert lembretes.rodar()["sem_vacina"] == 1
        lembretes.rodar()
        assert canal.modelos == [] and consultas == [2]

    def test_sem_calendario_adia(self, memoria):
        preparar(memoria)
        lembretes, canal = servico(memoria, lambda meses: None)
        assert lembretes.rodar()["adiados"] == 1
        assert memoria.lembretes_do_dia(HOJE)  # continua pendente

    def test_fora_do_horario_nao_manda(self, memoria):
        preparar(memoria)
        lembretes, canal = servico(memoria, lambda meses: True, hora=22)
        lembretes.rodar()
        assert canal.modelos == []

    def test_falha_no_envio_tenta_de_novo(self, memoria):
        preparar(memoria)
        lembretes, _ = servico(memoria, lambda meses: True, ok=False)
        assert lembretes.rodar()["falhas"] == 1
        assert memoria.lembretes_do_dia(HOJE)


class TestConversa:
    def montar(self, memoria, ativos=True):
        settings = Settings(openai_api_key="k", vector_store_id="vs",
                            memoria_ativa=True, lembretes_ativos=ativos)
        channel = FakeChannel()
        pipeline = MessagePipeline(settings, channel, FakeEngine(), memoria=memoria)
        pipeline.questionario._agora = lambda: dt.datetime(2026, 11, 26, 12, tzinfo=BRASILIA)
        return pipeline, channel

    def conversar(self, pipeline, *textos):
        for i, texto in enumerate(textos):
            pipeline.handle(text_message(texto, f"l{id(pipeline)}-{i}-{texto}", NUMERO))

    def test_questionario_termina_oferecendo_lembretes(self, memoria):
        pipeline, channel = self.montar(memoria)
        self.conversar(pipeline, "oi", "1", "1", "2", "15/09/2026", "1", "pular", "pular", "pular", "31")
        assert "Quer que eu te avise das vacinas?" in channel.sent[-1][1]
        self.conversar(pipeline, "1")
        assert "Combinado! 📅" in channel.sent[-1][1]
        assert memoria.decisao(NUMERO, LEMBRETES) is True
        assert memoria.lembretes_do_dia(dt.date(2026, 12, 15))[0]["telefone"] == NUMERO

    def test_parar_lembretes_revoga_e_apaga_o_telefone(self, memoria, banco):
        preparar(memoria)
        pipeline, channel = self.montar(memoria)
        self.conversar(pipeline, "PARAR LEMBRETES")
        assert memoria.decisao(NUMERO, LEMBRETES) is False
        with psycopg.connect(banco) as con:
            assert con.execute("SELECT telefone_cifrado FROM cuidadores").fetchone()[0] is None

    def test_lembretes_sem_perfil_pede_perfil(self, memoria):
        memoria.decidir(NUMERO, True, "w0")
        pipeline, channel = self.montar(memoria)
        self.conversar(pipeline, "LEMBRETES")
        assert "PERFIL" in channel.sent[-1][1]

    def test_desligado_nao_oferece(self, memoria):
        pipeline, channel = self.montar(memoria, ativos=False)
        self.conversar(pipeline, "oi", "1", "1", "2", "15/09/2026", "1", "pular", "pular", "pular", "31")
        assert "Quer que eu te avise das vacinas?" not in channel.bodies


def test_modelo_vai_no_formato_da_cloud_api(monkeypatch):
    from app.channels.meta_cloud import MetaCloudChannel

    chamadas = []

    class Resposta:
        status_code = 200
        text = ""

    monkeypatch.setattr(
        "app.channels.meta_cloud.post_with_retry",
        lambda url, **kw: chamadas.append((url, kw["json"])) or Resposta(),
    )
    canal = MetaCloudChannel(Settings(whatsapp_token="t", phone_number_id="123"))
    assert canal.send_template(NUMERO, "lembrete_vacina_mesversario", "pt_BR", ["sua bebê", "2 meses"])
    url, corpo = chamadas[0]
    assert url.endswith("/123/messages")
    assert corpo["type"] == "template"
    assert corpo["template"]["language"] == {"code": "pt_BR"}
    assert corpo["template"]["components"][0]["parameters"][1] == {"type": "text", "text": "2 meses"}
