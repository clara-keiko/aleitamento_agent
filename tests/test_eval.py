"""Testes do harness de avaliação.

A comparação entre modelos vira decisão de orçamento, então a aritmética
precisa estar certa — em especial o custo, que agora usa o consumo real
de tokens em vez de estimativa.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

from run_eval import (  # noqa: E402
    Report,
    Result,
    carregar_casos,
    carregar_precos,
    custo_por_pergunta,
    triagem,
)


def resultado(**kwargs):
    base = dict(
        case_id="x", pergunta="p", esperado="respondido", obtido="respondido", ok=True
    )
    base.update(kwargs)
    return Result(**base)


class TestCusto:
    def test_calcula_com_tokens_reais(self):
        precos = {"m": {"entrada": 0.15, "saida": 0.60}}
        # 6000 × 0,15/1M = 0,0009 ; 250 × 0,60/1M = 0,00015
        assert custo_por_pergunta("m", 6000, 250, precos) == pytest_approx(0.00105)

    def test_modelo_caro_custa_mais(self):
        precos = {
            "barato": {"entrada": 0.15, "saida": 0.60},
            "caro": {"entrada": 1.25, "saida": 10.00},
        }
        b = custo_por_pergunta("barato", 6000, 250, precos)
        c = custo_por_pergunta("caro", 6000, 250, precos)
        assert c > b * 5

    def test_modelo_sem_preco_retorna_none(self):
        assert custo_por_pergunta("desconhecido", 100, 10, {}) is None

    def test_zero_tokens_custa_zero(self):
        precos = {"m": {"entrada": 1.0, "saida": 1.0}}
        assert custo_por_pergunta("m", 0, 0, precos) == 0


class TestMetricas:
    def test_tokens_medios_ignoram_quem_nao_foi_ao_modelo(self):
        """Emergência não chama o modelo; não pode diluir a média."""
        r = Report(modelo="m", resultados=[
            resultado(tokens_entrada=6000, tokens_saida=200, latencia_ms=2000),
            resultado(tokens_entrada=4000, tokens_saida=100, latencia_ms=1500),
            resultado(esperado="emergencia", obtido="emergencia"),  # sem tokens
        ])
        assert r.tokens_medios == (5000, 150)

    def test_taxa_de_fundamentacao(self):
        r = Report(modelo="m", resultados=[
            resultado(tokens_entrada=100, citacoes=3),
            resultado(tokens_entrada=100, citacoes=1),
            resultado(tokens_entrada=100, citacoes=0),
            resultado(esperado="social", obtido="social"),  # não foi ao modelo
        ])
        assert r.taxa_fundamentacao == pytest_approx(66.67, 0.1)

    def test_percentis(self):
        r = Report(modelo="m", resultados=[
            resultado(latencia_ms=ms, tokens_entrada=1) for ms in [100, 200, 300, 400, 5000]
        ])
        assert r.percentil(0.5) == 300
        assert r.percentil(0.95) == 5000

    def test_percentil_sem_chamadas_e_zero(self):
        assert Report(modelo="m", resultados=[resultado()]).percentil(0.5) == 0

    def test_taxa_de_acerto(self):
        r = Report(resultados=[resultado(ok=True), resultado(ok=True), resultado(ok=False)])
        assert r.acertos == 2
        assert r.taxa == pytest_approx(66.67, 0.1)

    def test_falha_critica_e_separada(self):
        r = Report(resultados=[
            resultado(esperado="emergencia", ok=False),
            resultado(esperado="respondido", ok=False),
        ])
        assert len(r.falhas) == 2
        assert len(r.falhas_criticas) == 1


class TestConjuntoDourado:
    def test_carrega(self):
        casos = carregar_casos()
        assert len(casos) >= 40
        for caso in casos:
            assert "id" in caso and "pergunta" in caso and "outcome" in caso

    def test_ids_unicos(self):
        ids = [c["id"] for c in carregar_casos()]
        assert len(ids) == len(set(ids))

    def test_tem_casos_de_emergencia(self):
        emergencias = [c for c in carregar_casos() if c["outcome"] == "emergencia"]
        assert len(emergencias) >= 5

    def test_triagem_bate_com_o_esperado_nos_casos_locais(self):
        """Os casos que não dependem do modelo têm que passar sem rede."""
        for caso in carregar_casos():
            if caso["outcome"] in {"respondido", "fora_de_escopo"}:
                continue
            obtido, _ = triagem(caso["pergunta"])
            assert obtido == caso["outcome"], f"{caso['id']}: {caso['pergunta']}"


class TestRelatorioCego:
    """O relatório cego existe para controlar viés. Se ele mesmo enviesar,
    é pior que não ter — dá falsa confiança ao resultado."""

    def _reports(self, n_casos: int):
        from run_eval import escrever_relatorio_cego  # noqa: F401

        reports = []
        for indice, modelo in enumerate(["modelo-a", "modelo-b"]):
            resultados = [
                resultado(
                    case_id=f"c{i:02d}",
                    pergunta=f"pergunta {i}",
                    # Texto neutro de propósito: o nome do modelo não pode
                    # entrar pela resposta, senão o teste de anonimato mede
                    # o próprio dado de teste em vez do relatório.
                    resposta=f"conteúdo {indice} para o caso {i}",
                    tokens_entrada=100,
                )
                for i in range(n_casos)
            ]
            reports.append(Report(modelo=modelo, resultados=resultados))
        return reports

    def test_posicao_A_fica_equilibrada(self, tmp_path):
        import json

        from run_eval import escrever_relatorio_cego

        destino = tmp_path / "revisao.md"
        escrever_relatorio_cego(self._reports(30), destino)

        chave = json.loads((tmp_path / "revisao.chave.json").read_text(encoding="utf-8"))
        como_a = [v["A"] for v in chave.values()]
        assert como_a.count("modelo-a") == 15
        assert como_a.count("modelo-b") == 15

    def test_numero_impar_de_casos_fica_quase_equilibrado(self, tmp_path):
        import json

        from run_eval import escrever_relatorio_cego

        destino = tmp_path / "revisao.md"
        escrever_relatorio_cego(self._reports(7), destino)

        chave = json.loads((tmp_path / "revisao.chave.json").read_text(encoding="utf-8"))
        como_a = [v["A"] for v in chave.values()]
        assert abs(como_a.count("modelo-a") - como_a.count("modelo-b")) <= 1

    def test_markdown_nao_revela_o_modelo(self, tmp_path):
        from run_eval import escrever_relatorio_cego

        destino = tmp_path / "revisao.md"
        escrever_relatorio_cego(self._reports(6), destino)

        texto = destino.read_text(encoding="utf-8")
        assert "modelo-a" not in texto
        assert "modelo-b" not in texto
        assert "Resposta A:" in texto
        assert "- [ ] A melhor" in texto

    def test_e_reproduzivel(self, tmp_path):
        import json

        from run_eval import escrever_relatorio_cego

        chaves = []
        for nome in ["um", "dois"]:
            destino = tmp_path / f"{nome}.md"
            escrever_relatorio_cego(self._reports(10), destino)
            chaves.append(
                json.loads((tmp_path / f"{nome}.chave.json").read_text(encoding="utf-8"))
            )
        assert chaves[0] == chaves[1]


class TestPrecos:
    def test_carrega_e_tem_o_modelo_atual(self):
        precos = carregar_precos()
        assert "gpt-4o-mini" in precos

    def test_todo_preco_tem_entrada_e_saida(self):
        for modelo, tabela in carregar_precos().items():
            assert "entrada" in tabela, modelo
            assert "saida" in tabela, modelo
            assert tabela["saida"] >= tabela["entrada"], modelo


def pytest_approx(valor, tolerancia=0.00001):
    import pytest

    return pytest.approx(valor, abs=tolerancia)
