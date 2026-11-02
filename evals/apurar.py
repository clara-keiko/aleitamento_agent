#!/usr/bin/env python3
"""Apura a revisão cega preenchida pela consultora.

    python evals/apurar.py revisao.md

Lê as marcações, cruza com a chave e revela qual modelo venceu — com uma
noção de quanto dá para confiar no resultado, porque preferência humana em
poucos casos é ruído com facilidade.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

CABECALHO = re.compile(r"^##\s+(?P<id>[\w-]+)\s*$")
MARCACAO = re.compile(r"- \[(?P<marca>[ xX])\]\s+(?P<rotulo>[A-D]) melhor")
EMPATE = re.compile(r"- \[(?P<marca>[ xX])\]\s+empate")


@dataclass
class Apuracao:
    vitorias: dict[str, int] = field(default_factory=dict)
    empates: int = 0
    nao_preenchidos: list[str] = field(default_factory=list)
    ambiguos: list[str] = field(default_factory=list)

    @property
    def julgados(self) -> int:
        return sum(self.vitorias.values()) + self.empates


def apurar(markdown: str, chave: dict[str, dict[str, str]]) -> Apuracao:
    resultado = Apuracao()
    caso_atual: str | None = None
    escolhas: list[str] = []
    marcou_empate = False

    def fechar() -> None:
        nonlocal escolhas, marcou_empate
        if caso_atual is None:
            return

        total = len(escolhas) + (1 if marcou_empate else 0)
        if total == 0:
            resultado.nao_preenchidos.append(caso_atual)
        elif total > 1:
            resultado.ambiguos.append(caso_atual)
        elif marcou_empate:
            resultado.empates += 1
        else:
            rotulo = escolhas[0]
            modelo = chave.get(caso_atual, {}).get(rotulo)
            if modelo is None:
                resultado.ambiguos.append(caso_atual)
            else:
                resultado.vitorias[modelo] = resultado.vitorias.get(modelo, 0) + 1

        escolhas = []
        marcou_empate = False

    for linha in markdown.splitlines():
        cabecalho = CABECALHO.match(linha)
        if cabecalho:
            fechar()
            caso_atual = cabecalho.group("id")
            continue

        if caso_atual is None:
            continue

        for m in MARCACAO.finditer(linha):
            if m.group("marca").lower() == "x":
                escolhas.append(m.group("rotulo"))

        empate = EMPATE.search(linha)
        if empate and empate.group("marca").lower() == "x":
            marcou_empate = True

    fechar()
    return resultado


def imprimir(resultado: Apuracao) -> None:
    print("\n" + "═" * 58)
    print("  RESULTADO DA REVISÃO CEGA")
    print("═" * 58 + "\n")

    if not resultado.julgados:
        print("  Nenhuma marcação encontrada. O arquivo foi preenchido?\n")
        return

    print(f"  Casos julgados: {resultado.julgados}")
    if resultado.empates:
        print(f"  Empates: {resultado.empates}")
    print()

    ordenado = sorted(resultado.vitorias.items(), key=lambda kv: -kv[1])
    for modelo, vitorias in ordenado:
        proporcao = 100 * vitorias / resultado.julgados
        barra = "█" * int(proporcao / 3)
        print(f"  {modelo:<18} {vitorias:>3} ({proporcao:>3.0f}%)  {barra}")

    print()
    _veredito(resultado, ordenado)

    if resultado.nao_preenchidos:
        print(f"  Sem marcação ({len(resultado.nao_preenchidos)}): "
              f"{', '.join(resultado.nao_preenchidos[:8])}"
              + ("…" if len(resultado.nao_preenchidos) > 8 else ""))
    if resultado.ambiguos:
        print(f"  Marcação ambígua ({len(resultado.ambiguos)}): "
              f"{', '.join(resultado.ambiguos[:8])}")
    print()


def _veredito(resultado: Apuracao, ordenado: list[tuple[str, int]]) -> None:
    if len(ordenado) < 2:
        if ordenado:
            print(f"  Só um modelo recebeu preferência: {ordenado[0][0]}.\n")
        return

    (vencedor, v1), (_, v2) = ordenado[0], ordenado[1]
    decisivos = v1 + v2

    # Sem estatística pesada: com poucos julgamentos, uma diferença pequena
    # é ruído. A regra de bolso abaixo evita decidir com base em duas
    # respostas de diferença.
    if decisivos < 10:
        print(f"  ⓘ Apenas {decisivos} casos decisivos — pouco para concluir.")
        print("    Amplie o conjunto dourado antes de trocar de modelo.\n")
        return

    margem = abs(v1 - v2)
    limiar = max(3, int(decisivos**0.5))

    if margem < limiar:
        print(f"  Empate técnico ({v1} × {v2}). A diferença de {margem} caso(s)")
        print(f"    está dentro do ruído para {decisivos} julgamentos.")
        print("    → Fique no modelo mais barato.\n")
    else:
        print(f"  {vencedor} preferido de forma consistente ({v1} × {v2}).")
        print("    → Vale trocar, se o custo adicional couber. Compare com a")
        print("      tabela de custo do run_eval.py --comparar.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arquivo", help="markdown da revisão cega, preenchido")
    parser.add_argument(
        "--chave", default="", help="arquivo de chave (padrão: <arquivo>.chave.json)"
    )
    args = parser.parse_args()

    caminho = Path(args.arquivo)
    if not caminho.exists():
        print(f"ERRO: não encontrei {caminho}")
        return 2

    caminho_chave = Path(args.chave) if args.chave else caminho.with_suffix(".chave.json")
    if not caminho_chave.exists():
        print(f"ERRO: não encontrei a chave em {caminho_chave}")
        print("Ela é gerada junto com o relatório cego.")
        return 2

    chave = json.loads(caminho_chave.read_text(encoding="utf-8"))
    resultado = apurar(caminho.read_text(encoding="utf-8"), chave)
    imprimir(resultado)

    return 0


if __name__ == "__main__":
    sys.exit(main())
