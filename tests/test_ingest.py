"""Testes da seleção de arquivos para a base de conhecimento.

Existe por causa de um erro real: `docs/GO_LIVE.md` foi criado e seria
indexado como conteúdo clínico, competindo com o material de amamentação na
recuperação. A exclusão era uma lista fixa de nomes, que envelheceu no dia
seguinte.
"""

from pathlib import Path

from ingest_openai_kb import e_documentacao_do_projeto, listar_arquivos


class TestSeparacaoDeDocumentacao:
    def test_docs_do_projeto_ficam_de_fora(self):
        for nome in ["OPERACAO.md", "GO_LIVE.md", "README.md", "CONTRIBUTING.md"]:
            assert e_documentacao_do_projeto(Path(f"docs/{nome}")) is True, nome

    def test_conteudo_clinico_entra(self):
        clinicos = [
            "vacinas_brasil_bebes_criancas_gestantes_rag.md",
            "AppAM_livreDemanda_I.docx",
            "Amamentacao-bases-cientificas-4ed-2016.pdf",
        ]
        for nome in clinicos:
            assert e_documentacao_do_projeto(Path(f"docs/{nome}")) is False, nome

    def test_regra_so_vale_para_markdown(self):
        """Um PDF com nome em caixa alta ainda é conteúdo."""
        assert e_documentacao_do_projeto(Path("docs/CADERNETA.pdf")) is False


class TestListagemReal:
    def test_nenhum_doc_do_projeto_na_base(self):
        nomes = {caminho.name for caminho in listar_arquivos()}
        assert "OPERACAO.md" not in nomes
        assert "GO_LIVE.md" not in nomes

    def test_material_clinico_presente(self):
        nomes = {caminho.name for caminho in listar_arquivos()}
        assert "vacinas_brasil_bebes_criancas_gestantes_rag.md" in nomes
        assert any(nome.startswith("AppAM_") for nome in nomes)

    def test_todos_tem_extensao_suportada(self):
        for caminho in listar_arquivos():
            assert caminho.suffix.lower() in {".pdf", ".txt", ".md", ".docx"}

    def test_arquivos_ocultos_ignorados(self):
        assert not any(c.name.startswith(".") for c in listar_arquivos())
