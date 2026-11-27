"""Perguntas de vacina: SUS x particular, só gov.br e SBIm, e a foto."""

import datetime as dt
from types import SimpleNamespace

import pytest

from app import guardrails
from app.config import Settings
from app.cripto import Cofre
from app.llm import (
    DOMINIOS_VACINA,
    VACINA_PROMPT,
    Answer,
    AssistantEngine,
    count_url_citations,
    count_web_searches,
)
from app.memoria_persistente import MemoriaPersistente
from app.piloto import RegistroDoPiloto
from app.pipeline import MessagePipeline
from tests.test_pipeline import FakeChannel, FakeEngine, text_message

NUMERO = "5521981814170"


class TestDeteccao:
    @pytest.mark.parametrize("texto", [
        "quais vacinas ela toma aos 2 meses?",
        "a BCG deixa marca?",
        "pentavalente dá febre?",
        "posso tomar vacina da gripe amamentando?",
        "o calendário vacinal mudou?",
        "meningocócica B é no SUS?",
    ])
    def test_reconhece(self, texto):
        assert guardrails.is_vaccine_question(texto)

    @pytest.mark.parametrize("texto", [
        "tenho hepatite B, posso amamentar?",
        "como melhorar a pega?",
        "posso tomar dipirona?",
    ])
    def test_nao_confunde(self, texto):
        assert not guardrails.is_vaccine_question(texto)


class TestPrompt:
    def test_separa_sus_e_particular_nesta_ordem(self):
        assert VACINA_PROMPT.index("No SUS (gratuito)") < VACINA_PROMPT.index("Na rede particular")

    def test_ignora_a_base_e_usa_so_os_sites(self):
        assert "Calendário Nacional de Vacinação" in VACINA_PROMPT
        assert "sbim.org.br" in VACINA_PROMPT
        assert "Não use nenhum outro conhecimento" in VACINA_PROMPT


def resposta_com(urls, buscas=1, texto="*No SUS (gratuito):* ...\n*Na rede particular:* ..."):
    anotacoes = [SimpleNamespace(type="url_citation", url=u) for u in urls]
    return SimpleNamespace(
        output_text=texto,
        output=[SimpleNamespace(type="web_search_call")] * buscas
        + [SimpleNamespace(type="message", content=[SimpleNamespace(annotations=anotacoes)])],
        usage=SimpleNamespace(input_tokens=3000, output_tokens=400),
    )


class TestChamada:
    def engine(self, resposta):
        chamadas = []

        def create(**kw):
            chamadas.append(kw)
            return resposta

        settings = Settings(openai_api_key="k", vector_store_id="vs")
        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        return AssistantEngine(settings, client=client), chamadas

    def test_busca_so_nos_sites_oficiais_e_sem_a_base(self):
        engine, chamadas = self.engine(resposta_com(["https://www.gov.br/saude/pt-br/vacinacao/calendario"]))
        answer = engine.answer_vacina("quais vacinas aos 2 meses?")
        ferramentas = chamadas[0]["tools"]
        assert ferramentas == [{"type": "web_search", "filters": {"allowed_domains": DOMINIOS_VACINA}}]
        assert answer.grounded and answer.buscas_web == 1

    def test_sem_citacao_oficial_nao_e_fundamentada(self):
        engine, _ = self.engine(resposta_com(["https://blog-qualquer.com/vacinas"]))
        assert not engine.answer_vacina("quais vacinas?").grounded

    def test_conta_citacoes_de_subdominio_da_sbim(self):
        resposta = resposta_com([
            "https://familia.sbim.org.br/calendarios", "https://sbim.org.br/x", "https://outro.com",
        ], buscas=2)
        assert count_url_citations(resposta, DOMINIOS_VACINA) == 2
        assert count_web_searches(resposta) == 2

    def test_erro_vira_resposta_de_erro(self):
        def create(**kw):
            raise RuntimeError("fora do ar")

        settings = Settings(openai_api_key="k", vector_store_id="vs")
        engine = AssistantEngine(settings, client=SimpleNamespace(responses=SimpleNamespace(create=create)))
        assert engine.answer_vacina("vacina?").error


class TestNoPipeline:
    def montar(self, answer=None, memoria=None, ativa=False):
        settings = Settings(openai_api_key="k", vector_store_id="vs", memoria_ativa=ativa)
        channel = FakeChannel()
        engine = FakeEngine(answer=answer)
        return MessagePipeline(settings, channel, engine, memoria=memoria), channel, engine

    def test_pergunta_de_vacina_vai_para_a_busca_oficial(self):
        pipeline, channel, engine = self.montar()
        assert pipeline.handle(text_message("quais vacinas aos 2 meses?", "m1", NUMERO)) == "respondido"
        assert getattr(engine, "vacina_calls", 0) == 1

    def test_pergunta_comum_nao(self):
        pipeline, _, engine = self.montar()
        pipeline.handle(text_message("como melhorar a pega?", "m1", NUMERO))
        assert getattr(engine, "vacina_calls", 0) == 0

    def test_sem_fonte_oficial_manda_os_links(self):
        pipeline, channel, _ = self.montar(answer=Answer(text="x", grounded=False))
        assert pipeline.handle(text_message("quais vacinas?", "m1", NUMERO)) == "vacina_sem_fonte"
        assert "gov.br/saude" in channel.sent[-1][1] and "sbim.org.br" in channel.sent[-1][1]

    def test_busca_na_web_entra_no_gasto(self):
        answer = Answer(text="ok", grounded=True, input_tokens=1_000_000, output_tokens=0, buscas_web=2)
        pipeline, _, _ = self.montar(answer=answer)
        pipeline.piloto = RegistroDoPiloto(":memory:", {})
        pipeline.handle(text_message("quais vacinas?", "m1", NUMERO))
        # US$ 0,25 de tokens + 2 buscas a US$ 0,01 (sem a taxa do file_search)
        assert pipeline.piloto.gasto_do_mes() == pytest.approx(0.27)


class TestSugestaoDaFoto:
    @pytest.fixture
    def memoria(self, banco):
        return MemoriaPersistente(banco, Cofre("chave-de-teste"))

    def montar(self, memoria):
        settings = Settings(openai_api_key="k", vector_store_id="vs", memoria_ativa=True)
        channel = FakeChannel()
        return MessagePipeline(settings, channel, FakeEngine(), memoria=memoria), channel

    def test_com_bebe_e_sem_vacinas_sugere_a_foto(self, memoria):
        memoria.decidir(NUMERO, True, "w0")
        memoria.salvar_bebe(NUMERO, dt.date(2026, 9, 15))
        pipeline, channel = self.montar(memoria)
        pipeline.handle(text_message("quais vacinas aos 2 meses?", "m1", NUMERO))
        assert "foto da página de vacinas da caderneta" in channel.sent[-1][1]

    def test_sem_bebe_sugere_perfil_e_foto(self, memoria):
        memoria.decidir(NUMERO, True, "w0")
        pipeline, channel = self.montar(memoria)
        pipeline.handle(text_message("quais vacinas aos 2 meses?", "m1", NUMERO))
        assert "PERFIL" in channel.sent[-1][1]

    def test_com_vacinas_anotadas_nao_insiste(self, memoria):
        memoria.decidir(NUMERO, True, "w0")
        bebe = memoria.salvar_bebe(NUMERO, dt.date(2026, 9, 15))
        memoria.registrar_vacinas(bebe, [("BCG – dose única", dt.date(2026, 9, 15))])
        pipeline, channel = self.montar(memoria)
        pipeline.handle(text_message("quais vacinas aos 2 meses?", "m1", NUMERO))
        assert "📷" not in channel.sent[-1][1]

    def test_quem_nao_guarda_nao_recebe_a_sugestao(self, memoria):
        memoria.decidir(NUMERO, False, "w0")
        pipeline, channel = self.montar(memoria)
        pipeline.handle(text_message("quais vacinas aos 2 meses?", "m1", NUMERO))
        assert "📷" not in channel.sent[-1][1]


class TestCalendarioSUS:
    PDF_FALSO = None

    def resposta(self, texto="", tipo="text/html", status=200, conteudo=None):
        import requests

        r = SimpleNamespace(
            status_code=status, text=texto, headers={"content-type": tipo},
            content=conteudo if conteudo is not None else texto.encode(),
        )

        def raise_for_status():
            if status >= 400:
                raise requests.HTTPError(str(status))

        r.raise_for_status = raise_for_status
        return r

    def test_extrai_texto_do_html_e_guarda_a_fonte(self):
        from app.calendario_sus import baixar

        linhas = "".join(
            f"<tr><td>{idade}</td><td>vacina {nome}</td></tr>"
            for idade, nome in [("2 meses", "Pentavalente 1ª dose")] * 60
        )
        pagina = f"<html><script>x()</script><table>{linhas}</table></html>"
        calendario = baixar(["https://www.gov.br/x"], get=lambda url, **kw: self.resposta(pagina))
        assert calendario.url == "https://www.gov.br/x"
        assert "2 meses | vacina Pentavalente 1ª dose" in calendario.texto
        assert "x()" not in calendario.texto

    def test_pula_fonte_quebrada_e_usa_a_proxima(self):
        from app.calendario_sus import baixar

        pagina = "<p>vacina BCG ao nascer</p>" * 100

        def get(url, **kw):
            return self.resposta(status=500) if url.endswith("a") else self.resposta(pagina)

        assert baixar(["https://g/a", "https://g/b"], get=get).url == "https://g/b"

    def test_pagina_sem_calendario_nao_conta(self):
        from app.calendario_sus import baixar

        assert baixar(["https://g/a"], get=lambda url, **kw: self.resposta("<p>erro</p>")) is None

    def test_falha_nova_nao_apaga_o_calendario_anterior(self):
        from app.calendario_sus import Calendario, CalendarioSUS

        bom = Calendario("vacina BCG", "https://g", dt.datetime(2026, 11, 26, tzinfo=dt.UTC))
        respostas = iter([bom, None])
        calendario = CalendarioSUS(baixar_agora=lambda: next(respostas))
        calendario.atualizar()
        calendario.atualizar()
        assert calendario.atual is bom

    def test_calendario_vai_para_o_modelo_e_basta_como_fonte(self):
        from app.calendario_sus import Calendario, CalendarioSUS

        chamadas = []

        def create(**kw):
            chamadas.append(kw)
            return resposta_com([])  # a busca da SBIm não citou nada

        settings = Settings(openai_api_key="k", vector_store_id="vs")
        engine = AssistantEngine(settings, client=SimpleNamespace(responses=SimpleNamespace(create=create)))
        cal = Calendario("2 meses: Pentavalente, VIP, Pneumo 10, Rotavírus",
                         "https://www.gov.br/arquivo", dt.datetime(2026, 11, 26, tzinfo=dt.UTC))
        engine.calendario_sus = CalendarioSUS(baixar_agora=lambda: cal)
        engine.calendario_sus.atualizar()

        answer = engine.answer_vacina("quais vacinas aos 2 meses?")
        contexto = chamadas[0]["input"][1]["content"]
        assert "Calendário Nacional de Vacinação (SUS)" in contexto
        assert "26/11/2026" in contexto and "Pentavalente" in contexto
        assert "https://www.gov.br/saude/pt-br/vacinacao/calendario" in contexto
        assert answer.grounded


def test_prompt_pede_uma_vacina_por_linha_em_cada_parte():
    sus = VACINA_PROMPT.index("*No SUS (gratuito):*")
    particular = VACINA_PROMPT.index("*Na rede particular:*")
    assert "• Nome da vacina – dose" in VACINA_PROMPT[sus:particular]
    assert "• Nome da vacina – dose" in VACINA_PROMPT[particular:]
