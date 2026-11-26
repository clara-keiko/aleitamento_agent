"""Testes do motor de resposta.

Concentrados no que quebra ao trocar de família de modelo. O `gpt-5-mini`
raciocina antes de responder, e os tokens de raciocínio entram no mesmo
`max_output_tokens` da resposta visível — com o teto de um modelo comum,
ele gasta a cota pensando e devolve vazio.
"""

from app.config import Settings
from app.llm import (
    MAX_OUTPUT_PADRAO,
    MAX_OUTPUT_RACIOCINIO,
    AssistantEngine,
    is_reasoning_model,
    token_usage,
    truncation_reason,
)


def engine(**overrides):
    kwargs = {"openai_api_key": "k", "vector_store_id": "vs"}
    kwargs.update(overrides)
    return AssistantEngine(Settings(**kwargs))


class TestDeteccaoDeRaciocinio:
    def test_familia_gpt5_raciocina(self):
        for modelo in ["gpt-5", "gpt-5-mini", "gpt-5.4-mini", "GPT-5-Mini"]:
            assert is_reasoning_model(modelo) is True, modelo

    def test_familia_o_raciocina(self):
        for modelo in ["o1", "o3-mini", "o4-mini"]:
            assert is_reasoning_model(modelo) is True, modelo

    def test_gpt4_nao_raciocina(self):
        for modelo in ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"]:
            assert is_reasoning_model(modelo) is False, modelo

    def test_vazio_nao_quebra(self):
        assert is_reasoning_model("") is False


class TestTetoDeSaida:
    def test_raciocinio_ganha_teto_maior(self):
        """O teto de 600 do modelo comum deixaria o gpt-5-mini sem resposta."""
        assert engine(openai_model="gpt-5-mini").teto_de_saida == MAX_OUTPUT_RACIOCINIO

    def test_modelo_comum_usa_teto_menor(self):
        assert engine(openai_model="gpt-4o-mini").teto_de_saida == MAX_OUTPUT_PADRAO

    def test_teto_de_raciocinio_e_bem_maior(self):
        assert MAX_OUTPUT_RACIOCINIO >= MAX_OUTPUT_PADRAO * 4

    def test_valor_explicito_no_env_vence(self):
        motor = engine(openai_model="gpt-5-mini", max_output_tokens=1234)
        assert motor.teto_de_saida == 1234


class TestParametrosEnviados:
    """O `reasoning` só pode ir para quem raciocina — modelo comum rejeita."""

    def _capturar(self, **overrides):
        capturado = {}

        class ClienteFalso:
            class responses:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    capturado.update(kwargs)
                    raise RuntimeError("parar aqui")

        motor = engine(**overrides)
        motor._client = ClienteFalso()
        motor.answer("qual a pega correta?")
        return capturado

    def test_raciocinio_recebe_effort(self):
        enviado = self._capturar(openai_model="gpt-5-mini")
        assert enviado["reasoning"] == {"effort": "low"}
        assert enviado["max_output_tokens"] == MAX_OUTPUT_RACIOCINIO

    def test_modelo_comum_nao_recebe_effort(self):
        enviado = self._capturar(openai_model="gpt-4o-mini")
        assert "reasoning" not in enviado

    def test_effort_configuravel(self):
        enviado = self._capturar(openai_model="gpt-5-mini", reasoning_effort="medium")
        assert enviado["reasoning"] == {"effort": "medium"}

    def test_file_search_continua_configurado(self):
        enviado = self._capturar(openai_model="gpt-5-mini")
        ferramenta = enviado["tools"][0]
        assert ferramenta["type"] == "file_search"
        assert ferramenta["vector_store_ids"] == ["vs"]


class TestRespostaTruncada:
    def test_detecta_estouro_do_teto(self):
        class Detalhes:
            reason = "max_output_tokens"

        class Resposta:
            status = "incomplete"
            incomplete_details = Detalhes()

        assert truncation_reason(Resposta()) == "max_output_tokens"

    def test_aceita_detalhes_como_dict(self):
        class Resposta:
            status = "incomplete"
            incomplete_details = {"reason": "content_filter"}

        assert truncation_reason(Resposta()) == "content_filter"

    def test_resposta_completa_nao_tem_motivo(self):
        class Resposta:
            status = "completed"

        assert truncation_reason(Resposta()) is None

    def test_sem_status_nao_quebra(self):
        assert truncation_reason(object()) is None

    def test_incompleta_sem_detalhe(self):
        class Resposta:
            status = "incomplete"
            incomplete_details = None

        assert truncation_reason(Resposta()) == "desconhecido"


class TestUsoDeTokens:
    def test_le_nomes_da_responses_api(self):
        class Uso:
            input_tokens = 6000
            output_tokens = 250

        class Resposta:
            usage = Uso()

        assert token_usage(Resposta()) == (6000, 250)

    def test_aceita_nomes_antigos(self):
        class Uso:
            prompt_tokens = 100
            completion_tokens = 20

        class Resposta:
            usage = Uso()

        assert token_usage(Resposta()) == (100, 20)

    def test_sem_usage_devolve_zero(self):
        """Zero significa 'não medido', não 'foi de graça'."""
        assert token_usage(object()) == (0, 0)


class TestMarcasDeCitacao:
    """O modelo às vezes escreve a marca de citação no texto, e ela chegava
    à mãe no WhatsApp: "... (rodilhas) fileciteturn0file0L??-L??."."""

    def limpar(self, texto):
        from app.llm import limpar_marcas_de_citacao

        return limpar_marcas_de_citacao(texto)

    def test_marca_sem_caracteres_invisiveis(self):
        assert self.limpar(
            "use coxins de algodão fileciteturn0file0L??-L??."
        ) == "use coxins de algodão."

    def test_marca_com_caracteres_invisiveis(self):
        assert self.limpar(
            "Verifique a pega fileciteturn0file3."
        ) == "Verifique a pega."

    def test_formato_antigo(self):
        assert self.limpar("Ofereça o peito 【4:0†caderneta.pdf】.") == "Ofereça o peito."

    def test_espaco_antes_da_pontuacao(self):
        assert self.limpar("rachaduras .\nprocure avaliação profissional .") == (
            "rachaduras.\nprocure avaliação profissional."
        )

    def test_preserva_listas_e_paragrafos(self):
        texto = "Dicas:\n\n- Pega correta.\n- Leite no mamilo; deixe secar."
        assert self.limpar(texto) == texto

    def test_nao_mexe_em_palavras_parecidas(self):
        assert self.limpar("O arquivo citado (file) e o turno da noite.") == (
            "O arquivo citado (file) e o turno da noite."
        )


class TestFonteNaResposta:
    def test_nomes_legiveis(self):
        from app.guardrails import sources_footer

        rodape = sources_footer([
            "amamentacao_uso_medicamentos_outras_substancias_3ed.pdf",
            "AppAM_livreDemanda_I.docx",
            "AppAM_SONOdoBEBE.docx",
        ])
        assert "Amamentação e uso de medicamentos e outras substâncias (3ª ed., 2025)" in rodape
        assert rodape.count("Material educativo AppAM") == 1

    def test_sem_fonte_sem_rodape(self):
        from app.guardrails import sources_footer

        assert sources_footer([]) == ""

    def test_arquivos_citados_sem_repetir(self):
        from types import SimpleNamespace

        from app.llm import cited_files

        def cit(nome):
            return SimpleNamespace(type="file_citation", filename=nome)

        resposta = SimpleNamespace(output=[SimpleNamespace(content=[SimpleNamespace(
            annotations=[cit("a.pdf"), cit("b.docx"), cit("a.pdf")]
        )])])
        assert cited_files(resposta) == ["a.pdf", "b.docx"]

    def test_prompt_nao_diz_sim_para_remedio_nem_oferece_acoes(self):
        from app.llm import SYSTEM_PROMPT

        assert 'não responda "sim", "pode"' in SYSTEM_PROMPT
        assert "Não ofereça fazer nada além de responder" in SYSTEM_PROMPT
