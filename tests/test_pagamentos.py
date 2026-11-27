"""Cobrança do LactAI Plus pelo Asaas: link pessoal, ativação pelo aviso de
pagamento (uma vez por pagamento), renovação, estorno e cancelamento."""

import datetime as dt
import json

import pytest

from app import pagamentos
from app.config import Settings
from app.cripto import Cofre
from app.llm import Answer
from app.memoria_persistente import MemoriaPersistente
from app.pagamentos import Asaas, Cobranca, mais_um_mes, token_valido
from app.piloto import BRASILIA
from app.pipeline import MessagePipeline
from tests.test_pipeline import FakeChannel, FakeEngine, text_message

NUMERO = "5521981814170"
AGORA = dt.datetime(2026, 11, 27, 10, 0, tzinfo=BRASILIA)
HOJE = AGORA.date()


class Resposta:
    def __init__(self, status=200, corpo=None):
        self.status_code = status
        self._corpo = corpo or {}
        self.text = json.dumps(self._corpo) if corpo is not None else ""

    def json(self):
        return self._corpo


class AsaasFalso:
    """Grava as chamadas; devolve o que o Asaas devolveria."""

    def __init__(self):
        self.chamadas = []
        self.webhooks = []
        self.falhar = False

    def post(self, url, json=None, headers=None, timeout=None):
        self.chamadas.append(("POST", url, json, headers))
        if self.falhar:
            return Resposta(400, {"errors": [{"description": "erro"}]})
        if url.endswith("/paymentLinks"):
            n = sum(1 for c in self.chamadas if c[1].endswith("/paymentLinks"))
            return Resposta(200, {"id": f"lnk_{n}", "url": f"https://sandbox.asaas.com/c/lnk_{n}"})
        if url.endswith("/webhooks"):
            self.webhooks.append(json)
            return Resposta(200, {"id": "wh_1"})
        return Resposta(404, {})

    def get(self, url, headers=None, timeout=None):
        self.chamadas.append(("GET", url, None, headers))
        return Resposta(200, {"data": [{"url": w["url"]} for w in self.webhooks]})

    def delete(self, url, headers=None, timeout=None):
        self.chamadas.append(("DELETE", url, None, headers))
        return Resposta(200, {"deleted": True})


def configuracao(**extra):
    return Settings(
        openai_api_key="k", vector_store_id="vs", asaas_api_key="$aact_teste",
        asaas_ambiente="sandbox", asaas_webhook_token="segredo-do-aviso", **extra,
    )


@pytest.fixture
def memoria(banco):
    return MemoriaPersistente(banco, Cofre("chave-de-teste"), agora=lambda: AGORA)


@pytest.fixture
def falso():
    return AsaasFalso()


@pytest.fixture
def cobranca(memoria, falso):
    settings = configuracao()
    return Cobranca(settings, memoria, Asaas(settings, falso.post, falso.get, falso.delete))


def aviso(evento="PAYMENT_CONFIRMED", pagamento="pay_1", link="lnk_1",
          assinatura="sub_1", vencimento="2026-11-27"):
    return {"id": f"evt_{pagamento}_{evento}", "event": evento, "payment": {
        "id": pagamento, "paymentLink": link, "subscription": assinatura,
        "dueDate": vencimento, "value": 29.9, "status": "CONFIRMED",
    }}


class TestAuxiliares:
    @pytest.mark.parametrize("dia,esperado", [
        (dt.date(2026, 11, 27), dt.date(2026, 12, 27)),
        (dt.date(2026, 12, 15), dt.date(2027, 1, 15)),
        (dt.date(2027, 1, 31), dt.date(2027, 2, 28)),
    ])
    def test_mais_um_mes(self, dia, esperado):
        assert mais_um_mes(dia) == esperado

    def test_token(self):
        assert token_valido("abc", "abc")
        assert not token_valido("abd", "abc")
        assert not token_valido(None, "abc")
        assert not token_valido("", "")  # sem token configurado, nada passa

    def test_base_por_ambiente(self):
        assert "sandbox" in Asaas(configuracao()).base
        prod = Settings(openai_api_key="k", vector_store_id="vs", asaas_ambiente="producao")
        assert Asaas(prod).base == "https://api.asaas.com/v3"


class TestLink:
    def test_cria_link_recorrente_mensal_uma_vez(self, cobranca, falso):
        primeira = cobranca.assinar(NUMERO, HOJE)
        segunda = cobranca.assinar(NUMERO, HOJE)
        assert "https://sandbox.asaas.com/c/lnk_1" in primeira
        assert primeira == segunda
        criacoes = [c for c in falso.chamadas if c[1].endswith("/paymentLinks")]
        assert len(criacoes) == 1
        corpo, cabecalhos = criacoes[0][2], criacoes[0][3]
        assert corpo["chargeType"] == "RECURRENT"
        assert corpo["subscriptionCycle"] == "MONTHLY"
        assert corpo["value"] == pytest.approx(29.90)
        assert corpo["billingType"] == "UNDEFINED"
        assert cabecalhos["access_token"] == "$aact_teste"
        assert cabecalhos["User-Agent"]

    def test_falha_do_asaas_nao_grava_nada(self, cobranca, falso, memoria):
        falso.falhar = True
        assert cobranca.assinar(NUMERO, HOJE) == pagamentos.MENSAGEM_INDISPONIVEL
        assert memoria.link_de_pagamento(NUMERO) is None

    def test_quem_ja_e_plus_nao_recebe_link(self, cobranca, memoria, falso):
        memoria.definir_assinatura(NUMERO, "plus", HOJE, HOJE + dt.timedelta(days=30))
        assert "já é *LactAI Plus*" in cobranca.assinar(NUMERO, HOJE)
        assert not falso.chamadas


class TestAviso:
    def test_pagamento_ativa_o_plus_e_avisa(self, cobranca, memoria):
        cobranca.assinar(NUMERO, HOJE)
        recado = cobranca.processar_aviso(aviso(), HOJE)
        assert recado is not None
        numero, texto = recado
        assert numero == NUMERO
        assert "Pagamento confirmado" in texto
        atual = memoria.assinatura(NUMERO)
        assert atual["plano"] == "plus"
        assert atual["origem"] == "asaas"
        assert atual["inicio"] == HOJE
        assert atual["pago_ate"] == dt.date(2026, 12, 30)  # um mês + 3 dias

    def test_aviso_repetido_nao_estende_de_novo(self, cobranca, memoria):
        cobranca.assinar(NUMERO, HOJE)
        cobranca.processar_aviso(aviso(), HOJE)
        # Cartão: CONFIRMED e depois RECEIVED do mesmo pagamento; e reentrega.
        assert cobranca.processar_aviso(aviso("PAYMENT_RECEIVED"), HOJE) is None
        assert cobranca.processar_aviso(aviso(), HOJE) is None
        assert memoria.assinatura(NUMERO)["pago_ate"] == dt.date(2026, 12, 30)

    def test_renovacao_estende_sem_mensagem_e_mantem_inicio(self, cobranca, memoria):
        cobranca.assinar(NUMERO, HOJE)
        cobranca.processar_aviso(aviso(), HOJE)
        # Próxima cobrança da assinatura: pode vir sem o id do link.
        dia = dt.date(2026, 12, 27)
        renovacao = aviso(pagamento="pay_2", link=None, vencimento="2026-12-27")
        assert cobranca.processar_aviso(renovacao, dia) is None
        atual = memoria.assinatura(NUMERO)
        assert atual["pago_ate"] == dt.date(2027, 1, 30)
        assert atual["inicio"] == HOJE

    def test_link_desconhecido_e_ignorado(self, cobranca, memoria):
        assert cobranca.processar_aviso(aviso(link="lnk_de_outro", assinatura="sub_x"), HOJE) is None
        assert memoria.assinatura(NUMERO) is None

    def test_eventos_que_nao_sao_pagamento_sao_ignorados(self, cobranca, memoria):
        cobranca.assinar(NUMERO, HOJE)
        assert cobranca.processar_aviso(aviso("PAYMENT_CREATED"), HOJE) is None
        assert cobranca.processar_aviso({"event": "PAYMENT_CONFIRMED"}, HOJE) is None
        assert memoria.assinatura(NUMERO) is None

    def test_estorno_encerra_o_plus(self, cobranca, memoria):
        cobranca.assinar(NUMERO, HOJE)
        cobranca.processar_aviso(aviso(), HOJE)
        cobranca.processar_aviso(aviso("PAYMENT_REFUNDED"), HOJE)
        assert memoria.assinatura(NUMERO)["pago_ate"] < HOJE


class TestCancelar:
    def test_cancela_a_recorrencia_e_mantem_o_que_foi_pago(self, cobranca, memoria, falso):
        cobranca.assinar(NUMERO, HOJE)
        cobranca.processar_aviso(aviso(), HOJE)
        texto = cobranca.cancelar(NUMERO)
        assert "30/12/2026" in texto
        assert ("DELETE", "https://api-sandbox.asaas.com/v3/subscriptions/sub_1", None,
                falso.chamadas[-1][3]) == falso.chamadas[-1]
        assert memoria.assinatura(NUMERO)["pago_ate"] == dt.date(2026, 12, 30)
        assert cobranca.cancelar(NUMERO) == pagamentos.MENSAGEM_SEM_ASSINATURA

    def test_sem_assinatura(self, cobranca):
        assert cobranca.cancelar(NUMERO) == pagamentos.MENSAGEM_SEM_ASSINATURA


class TestWebhookDoAsaas:
    def test_cadastra_uma_vez(self, cobranca, falso):
        url = "https://lactai.example/pagamentos/asaas"
        assert cobranca.asaas.garantir_webhook(url, "tok", "a@b.c") == "criado"
        assert cobranca.asaas.garantir_webhook(url, "tok", "a@b.c") == "ja_existia"
        corpo = falso.webhooks[0]
        assert corpo["authToken"] == "tok"
        assert "PAYMENT_CONFIRMED" in corpo["events"] and "PAYMENT_RECEIVED" in corpo["events"]


def montar(memoria, falso):
    settings = configuracao(memoria_ativa=True)
    channel = FakeChannel()
    engine = FakeEngine(answer=Answer(text="Resposta.", grounded=True))
    cobranca = Cobranca(settings, memoria, Asaas(settings, falso.post, falso.get, falso.delete))
    pipeline = MessagePipeline(settings, channel, engine, memoria=memoria, cobranca=cobranca)
    pipeline.questionario._agora = lambda: AGORA
    return pipeline, channel, engine


def dizer(pipeline, texto):
    return pipeline.handle(text_message(texto, f"{texto}-{id(object())}", NUMERO))


class TestComandos:
    def test_assinar_manda_o_link_sem_gastar_o_modelo(self, memoria, falso):
        pipeline, channel, engine = montar(memoria, falso)
        assert dizer(pipeline, "ASSINAR") == "assinar"
        assert "https://sandbox.asaas.com/c/lnk_1" in channel.sent[-1][1]
        assert not engine.calls

    def test_cancelar_assinatura_nao_e_sair_do_servico(self, memoria, falso):
        pipeline, channel, _ = montar(memoria, falso)
        dizer(pipeline, "assinar")
        pipeline.cobranca.processar_aviso(aviso(), HOJE)
        assert dizer(pipeline, "Cancelar assinatura") == "cancelar_assinatura"
        assert "cancelei a renovação" in channel.sent[-1][1]

    def test_apagar_tudo_cancela_a_cobranca(self, memoria, falso):
        pipeline, _, _ = montar(memoria, falso)
        dizer(pipeline, "assinar")
        pipeline.cobranca.processar_aviso(aviso(), HOJE)
        memoria.decidir(NUMERO, True, "teste")
        dizer(pipeline, "APAGAR")
        dizer(pipeline, "CONFIRMAR")
        assert any(c[0] == "DELETE" for c in falso.chamadas)
        assert memoria.link_de_pagamento(NUMERO) is None

    def test_sem_chave_assinar_vai_para_o_modelo(self, memoria, falso):
        settings = Settings(openai_api_key="k", vector_store_id="vs")
        channel = FakeChannel()
        engine = FakeEngine(answer=Answer(text="Resposta.", grounded=True))
        pipeline = MessagePipeline(settings, channel, engine, memoria=memoria,
                                   cobranca=Cobranca(settings, memoria))
        assert dizer(pipeline, "assinar") != "assinar"


class TestRota:
    """POST /pagamentos/asaas: só aceita o token cadastrado."""

    @pytest.fixture
    def cliente(self, monkeypatch, cobranca, falso):
        from fastapi.testclient import TestClient

        import main

        cobranca.settings = configuracao()
        monkeypatch.setattr(main, "cobranca", cobranca)
        monkeypatch.setattr(main.settings, "asaas_webhook_token", "segredo-do-aviso")
        enviados = []
        destino = type("Canal", (), {"send_text": lambda self, n, t: enviados.append((n, t))})()
        monkeypatch.setattr(main, "canal_meta", destino)
        cobranca.assinar(NUMERO, HOJE)
        return TestClient(main.app), enviados

    def test_token_errado_e_recusado(self, cliente, memoria):
        client, enviados = cliente
        r = client.post("/pagamentos/asaas", json=aviso(),
                        headers={"asaas-access-token": "outro"})
        assert r.status_code == 401
        assert memoria.assinatura(NUMERO) is None

    def test_aviso_autentico_ativa_e_confirma_no_whatsapp(self, cliente, memoria):
        client, enviados = cliente
        r = client.post("/pagamentos/asaas", json=aviso(),
                        headers={"asaas-access-token": "segredo-do-aviso"})
        assert r.status_code == 200
        assert memoria.assinatura(NUMERO)["plano"] == "plus"
        assert enviados and enviados[0][0] == NUMERO
