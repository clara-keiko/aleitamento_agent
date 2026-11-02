"""Testes da apuração da revisão cega.

Esta apuração decide se vale trocar de modelo, então errar a contagem
significa gastar mais por token sem motivo — ou deixar de gastar quando
valia a pena.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

from apurar import apurar  # noqa: E402

CHAVE = {
    "edu-01": {"A": "gpt-4o-mini", "B": "gpt-5-mini"},
    "edu-02": {"A": "gpt-5-mini", "B": "gpt-4o-mini"},
    "med-01": {"A": "gpt-4o-mini", "B": "gpt-5-mini"},
}


def markdown(*blocos: str) -> str:
    return "\n".join(blocos)


def caso(case_id: str, marcacao: str) -> str:
    return f"## {case_id}\n\n**Pergunta:** qualquer\n\n{marcacao}\n"


class TestContagem:
    def test_conta_vitoria_respeitando_o_embaralhamento(self):
        """Em edu-02 o rótulo A é o gpt-5-mini, não o 4o-mini."""
        texto = markdown(
            caso("edu-01", "- [x] A melhor  - [ ] B melhor  - [ ] empate"),
            caso("edu-02", "- [x] A melhor  - [ ] B melhor  - [ ] empate"),
        )
        r = apurar(texto, CHAVE)
        assert r.vitorias == {"gpt-4o-mini": 1, "gpt-5-mini": 1}

    def test_conta_empate(self):
        texto = caso("edu-01", "- [ ] A melhor  - [ ] B melhor  - [x] empate")
        r = apurar(texto, CHAVE)
        assert r.empates == 1
        assert r.vitorias == {}

    def test_aceita_x_maiusculo(self):
        texto = caso("edu-01", "- [X] A melhor  - [ ] B melhor  - [ ] empate")
        assert apurar(texto, CHAVE).vitorias == {"gpt-4o-mini": 1}

    def test_soma_varios_casos(self):
        texto = markdown(
            caso("edu-01", "- [x] B melhor"),   # B = gpt-5-mini
            caso("edu-02", "- [x] A melhor"),   # A = gpt-5-mini
            caso("med-01", "- [x] B melhor"),   # B = gpt-5-mini
        )
        assert apurar(texto, CHAVE).vitorias == {"gpt-5-mini": 3}


class TestPreenchimentoIncompleto:
    def test_caso_sem_marcacao_e_sinalizado(self):
        texto = caso("edu-01", "- [ ] A melhor  - [ ] B melhor  - [ ] empate")
        r = apurar(texto, CHAVE)
        assert r.nao_preenchidos == ["edu-01"]
        assert r.julgados == 0

    def test_duas_marcacoes_e_ambiguo(self):
        texto = caso("edu-01", "- [x] A melhor  - [x] B melhor")
        r = apurar(texto, CHAVE)
        assert r.ambiguos == ["edu-01"]
        assert r.vitorias == {}

    def test_marcar_melhor_e_empate_e_ambiguo(self):
        texto = caso("edu-01", "- [x] A melhor  - [ ] B melhor  - [x] empate")
        assert apurar(texto, CHAVE).ambiguos == ["edu-01"]

    def test_caso_fora_da_chave_e_ambiguo(self):
        texto = caso("desconhecido", "- [x] A melhor")
        r = apurar(texto, CHAVE)
        assert r.ambiguos == ["desconhecido"]

    def test_arquivo_vazio(self):
        r = apurar("", CHAVE)
        assert r.julgados == 0
        assert r.vitorias == {}


class TestRobustez:
    def test_ignora_texto_antes_do_primeiro_caso(self):
        texto = "# Revisão cega\n\nInstruções.\n\n- [x] A melhor\n\n" + caso(
            "edu-01", "- [x] B melhor"
        )
        r = apurar(texto, CHAVE)
        assert r.vitorias == {"gpt-5-mini": 1}

    def test_comentario_da_consultora_nao_atrapalha(self):
        texto = caso(
            "edu-01",
            "- [x] A melhor  - [ ] B melhor  - [ ] empate\n\n"
            "Comentário: a B usa jargão demais para uma mãe.",
        )
        assert apurar(texto, CHAVE).vitorias == {"gpt-4o-mini": 1}

    def test_julgados_soma_vitorias_e_empates(self):
        texto = markdown(
            caso("edu-01", "- [x] A melhor"),
            caso("edu-02", "- [x] empate"),
            caso("med-01", "- [ ] A melhor"),
        )
        r = apurar(texto, CHAVE)
        assert r.julgados == 2
        assert r.nao_preenchidos == ["med-01"]
