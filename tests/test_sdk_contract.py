"""Verifica que o SDK instalado tem a superfície que o código usa.

Existe por causa de uma regressão real: um pin em `openai==1.59.x` instala um
SDK sem `client.responses` nem `client.vector_stores`. O código importa e os
testes com mock passam — a quebra só apareceria em produção, na primeira
mensagem de uma mãe. Estes testes falham no CI em vez disso.
"""

import openai
from openai import OpenAI


def client() -> OpenAI:
    return OpenAI(api_key="chave-de-teste")


class TestSuperficieDoSDK:
    def test_versao_e_2_ou_maior(self):
        major = int(openai.__version__.split(".")[0])
        assert major >= 2, (
            f"openai {openai.__version__} é antigo demais: a 1.x não tem "
            "client.responses nem client.vector_stores"
        )

    def test_responses_api_existe(self):
        """É a API que substitui a Assistants, desligada em 26/08/2026."""
        assert hasattr(client(), "responses")
        assert hasattr(client().responses, "create")

    def test_vector_stores_existe(self):
        assert hasattr(client(), "vector_stores")
        assert hasattr(client().vector_stores, "create")

    def test_upload_em_lote_existe(self):
        assert hasattr(client().vector_stores.file_batches, "upload_and_poll")

    def test_transcricao_existe(self):
        assert hasattr(client().audio.transcriptions, "create")


class TestContratoDoNossoCodigo:
    def test_engine_usa_responses(self):
        """Se o SDK mudar o nome, isto quebra aqui e não com a usuária."""
        from app.config import Settings
        from app.llm import AssistantEngine

        engine = AssistantEngine(Settings(openai_api_key="k", vector_store_id="vs"))
        assert hasattr(engine.client, "responses")

    def test_deteccao_de_citacao_aceita_objeto_e_dict(self):
        """A checagem de fundamentação não pode depender do formato exato."""
        from app.llm import has_file_citations

        class Anotacao:
            type = "file_citation"

        class Conteudo:
            annotations = [Anotacao()]

        class Item:
            content = [Conteudo()]

        class RespostaObjeto:
            output = [Item()]

        assert has_file_citations(RespostaObjeto()) is True

        class ConteudoDict:
            annotations = [{"type": "file_citation", "file_id": "f1"}]

        class ItemDict:
            content = [ConteudoDict()]

        class RespostaDict:
            output = [ItemDict()]

        assert has_file_citations(RespostaDict()) is True

    def test_sem_citacao_retorna_false(self):
        from app.llm import has_file_citations

        class Conteudo:
            annotations = []

        class Item:
            content = [Conteudo()]

        class Resposta:
            output = [Item()]

        assert has_file_citations(Resposta()) is False

    def test_resposta_vazia_nao_quebra(self):
        from app.llm import has_file_citations

        class Vazia:
            output = None

        assert has_file_citations(Vazia()) is False
        assert has_file_citations(object()) is False
