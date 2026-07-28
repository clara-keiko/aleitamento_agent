#!/usr/bin/env python3
"""Avaliação do agente contra o conjunto dourado.

Existe para responder com medição, não com opinião:

  1. A triagem clínica ainda está correta depois desta mudança?
  2. O modelo responde a partir da base ou está inventando?
  3. Trocar de modelo melhora alguma coisa, e a que custo?

Modos:

    python evals/run_eval.py
        Só a triagem (guardrails). Não chama a OpenAI, não custa nada,
        roda em CI. É a camada crítica de segurança.

    python evals/run_eval.py --live
        Chama o modelo configurado. Mede fundamentação, recusa fora de
        escopo, latência e o custo com o consumo real de tokens.

    python evals/run_eval.py --comparar gpt-4o-mini,gpt-5-mini
        Roda o mesmo conjunto em cada modelo e imprime a tabela lado a
        lado — qualidade, latência e projeção de custo mensal.

    python evals/run_eval.py --comparar a,b --relatorio comparacao.md
        O mesmo, e grava as respostas de cada modelo lado a lado num
        markdown, para alguém julgar a qualidade que número nenhum mede.

Sai com código 1 se qualquer caso de emergência falhar. Um falso negativo
de emergência é o pior erro que este sistema pode cometer, e nenhuma
melhora em outra métrica compensa.
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from app import guardrails  # noqa: E402
from app.config import Settings  # noqa: E402
from app.llm import AssistantEngine  # noqa: E402

AQUI = Path(__file__).parent
GOLDEN_SET = AQUI / "golden_set.yaml"
PRECOS = AQUI / "precos.yaml"

SAFETY_CRITICAL = {"emergencia"}

# Volume mensal usado para projetar custo. Cenário "Operação" de
# docs/OPERACAO.md: mil mães, doze interações por mês.
INTERACOES_MES_PADRAO = 12_000


@dataclass
class Result:
    case_id: str
    pergunta: str
    esperado: str
    obtido: str
    ok: bool
    detalhe: str = ""
    latencia_ms: int = 0
    tokens_entrada: int = 0
    tokens_saida: int = 0
    citacoes: int = 0
    resposta: str = ""
    # Caso que carrega `nao_deve_conter` no conjunto dourado: pedido de dose,
    # de prescrição. É onde um modelo mais forte costuma se diferenciar
    # depois que a fundamentação satura.
    testa_seguranca: bool = False
    vazou_conteudo: bool = False

    @property
    def foi_ao_modelo(self) -> bool:
        return self.tokens_entrada > 0 or self.latencia_ms > 0


@dataclass
class Report:
    modelo: str = ""
    resultados: list[Result] = field(default_factory=list)

    def por_categoria(self) -> dict[str, tuple[int, int]]:
        agregado: dict[str, list[int]] = {}
        for r in self.resultados:
            bucket = agregado.setdefault(r.esperado, [0, 0])
            bucket[1] += 1
            if r.ok:
                bucket[0] += 1
        return {k: (v[0], v[1]) for k, v in agregado.items()}

    @property
    def falhas(self) -> list[Result]:
        return [r for r in self.resultados if not r.ok]

    @property
    def falhas_criticas(self) -> list[Result]:
        return [r for r in self.falhas if r.esperado in SAFETY_CRITICAL]

    @property
    def acertos(self) -> int:
        return len(self.resultados) - len(self.falhas)

    @property
    def taxa(self) -> float:
        return 100 * self.acertos / len(self.resultados) if self.resultados else 0.0

    # ------------------------------------------------------------------
    # Métricas que só existem no modo --live
    # ------------------------------------------------------------------
    @property
    def chamadas(self) -> list[Result]:
        return [r for r in self.resultados if r.foi_ao_modelo]

    def percentil(self, p: float) -> int:
        latencias = sorted(r.latencia_ms for r in self.chamadas if r.latencia_ms)
        if not latencias:
            return 0
        indice = min(int(len(latencias) * p), len(latencias) - 1)
        return latencias[indice]

    @property
    def tokens_medios(self) -> tuple[int, int]:
        chamadas = self.chamadas
        if not chamadas:
            return 0, 0
        entrada = sum(r.tokens_entrada for r in chamadas) // len(chamadas)
        saida = sum(r.tokens_saida for r in chamadas) // len(chamadas)
        return entrada, saida

    @property
    def taxa_fundamentacao(self) -> float:
        """Das perguntas que chegaram ao modelo, quantas citaram a base."""
        chamadas = self.chamadas
        if not chamadas:
            return 0.0
        com_citacao = sum(1 for r in chamadas if r.citacoes > 0)
        return 100 * com_citacao / len(chamadas)

    @property
    def seguranca_conteudo(self) -> tuple[int, int]:
        """(passaram, total) nos casos que testam vazamento de dose.

        Quando a fundamentação satura — acima de ~95% —, esta é a métrica
        que ainda separa um modelo do outro: seguir "não indique dose" é
        obediência a instrução, não recuperação.
        """
        casos = [r for r in self.resultados if r.testa_seguranca]
        if not casos:
            return 0, 0
        return sum(1 for r in casos if not r.vazou_conteudo), len(casos)

    @property
    def citacoes_medias(self) -> float:
        """Quantos trechos a resposta cita, em média.

        Fundamentação diz *se* citou; isto diz *quanto*. Com fundamentação
        saturada, mais citações costuma indicar resposta mais apoiada.
        """
        com_citacao = [r for r in self.chamadas if r.citacoes > 0]
        if not com_citacao:
            return 0.0
        return sum(r.citacoes for r in com_citacao) / len(com_citacao)


# ----------------------------------------------------------------------
# Carregamento
# ----------------------------------------------------------------------

def carregar_casos() -> list[dict]:
    with open(GOLDEN_SET, encoding="utf-8") as f:
        return yaml.safe_load(f)


def carregar_precos() -> dict[str, dict]:
    if not PRECOS.exists():
        return {}
    with open(PRECOS, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def custo_por_pergunta(modelo: str, entrada: int, saida: int, precos: dict) -> float | None:
    tabela = precos.get(modelo)
    if not tabela:
        return None
    return (entrada * tabela["entrada"] + saida * tabela["saida"]) / 1_000_000


# ----------------------------------------------------------------------
# Execução
# ----------------------------------------------------------------------

def triagem(pergunta: str) -> tuple[str, bool]:
    """Reproduz a ordem de decisão do pipeline, sem chamar o modelo."""
    risco = guardrails.classify_risk(pergunta)

    if risco.level == guardrails.EMERGENCY_NOW:
        return "emergencia", False
    if risco.level == guardrails.REFER_MEDICAL_CARE:
        return "encaminhamento", False
    if guardrails.is_small_talk(pergunta):
        return "social", False
    return "modelo", risco.needs_safety_note


def _avaliar_triagem(caso: dict, obtido: str, nota: bool) -> Result:
    esperado = caso["outcome"]
    ok = obtido == esperado
    detalhe = "" if ok else f"esperado '{esperado}', obtido '{obtido}'"
    return Result(caso["id"], caso["pergunta"], esperado, obtido, ok, detalhe)


def avaliar_offline(casos: list[dict]) -> Report:
    """Só a triagem. Casos que dependem do modelo são parcialmente checados."""
    report = Report()

    for caso in casos:
        esperado = caso["outcome"]
        obtido, nota = triagem(caso["pergunta"])

        if esperado in {"respondido", "fora_de_escopo"}:
            # Sem chamar o modelo, o máximo que dá para exigir é que a
            # triagem tenha deixado a pergunta passar.
            ok = obtido == "modelo"
            detalhe = "" if ok else f"triagem interceptou como '{obtido}'"
            if ok and caso.get("nota_de_seguranca") and not nota:
                ok, detalhe = False, "faltou a nota de segurança"
            report.resultados.append(
                Result(caso["id"], caso["pergunta"], esperado, obtido, ok, detalhe)
            )
        else:
            report.resultados.append(_avaliar_triagem(caso, obtido, nota))

    return report


def avaliar_live(casos: list[dict], modelo: str, silencioso: bool = False) -> Report:
    settings = Settings(openai_model=modelo)
    faltando = settings.missing_core()
    if faltando:
        print(f"ERRO: defina {', '.join(faltando)} para rodar com --live.")
        sys.exit(2)

    engine = AssistantEngine(settings)
    report = Report(modelo=modelo)

    for caso in casos:
        esperado = caso["outcome"]
        obtido, nota = triagem(caso["pergunta"])

        if obtido != "modelo":
            resultado = _avaliar_triagem(caso, obtido, nota)
            report.resultados.append(resultado)
            if not silencioso:
                print(f"  {'✓' if resultado.ok else '✗'} {caso['id']:<8} {obtido}")
            continue

        inicio = time.monotonic()
        resposta = engine.answer(caso["pergunta"])
        latencia = int((time.monotonic() - inicio) * 1000)

        if resposta.error:
            obtido, detalhe, ok = "erro", "chamada ao modelo falhou", False
        elif resposta.grounded:
            obtido = "respondido"
            ok = esperado == "respondido"
            detalhe = "" if ok else "respondeu algo que deveria recusar"
        else:
            obtido = "fora_de_escopo"
            ok = esperado == "fora_de_escopo"
            detalhe = "" if ok else "não citou a base — recusou algo que deveria responder"

        # Conteúdo proibido (ex.: dose de medicamento). Avaliado sempre que o
        # caso pedir, mesmo que o resultado já esteja marcado como falha —
        # senão a métrica de segurança fica com buraco.
        testa_seguranca = bool(caso.get("nao_deve_conter"))
        vazou = False
        if testa_seguranca:
            texto = guardrails.normalize(resposta.text)
            achados = [
                termo
                for termo in caso["nao_deve_conter"]
                if guardrails.normalize(termo) in texto
            ]
            if achados:
                vazou = True
                if ok:
                    ok, detalhe = False, f"conteúdo proibido: {', '.join(achados)}"

        if ok and caso.get("nota_de_seguranca") and not nota:
            ok, detalhe = False, "faltou a nota de segurança"

        resultado = Result(
            caso["id"], caso["pergunta"], esperado, obtido, ok, detalhe,
            latencia_ms=latencia,
            tokens_entrada=resposta.input_tokens,
            tokens_saida=resposta.output_tokens,
            citacoes=resposta.citations,
            resposta=resposta.text,
            testa_seguranca=testa_seguranca,
            vazou_conteudo=vazou,
        )
        report.resultados.append(resultado)

        if not silencioso:
            print(
                f"  {'✓' if ok else '✗'} {caso['id']:<8} {obtido:<16} "
                f"{latencia:>6} ms  {resposta.input_tokens:>6}→{resposta.output_tokens} tok"
            )

    return report


# ----------------------------------------------------------------------
# Saída
# ----------------------------------------------------------------------

def imprimir(report: Report, live: bool, precos: dict, interacoes: int) -> None:
    print("\n" + "=" * 70)
    print(f"  {'AVALIAÇÃO COMPLETA' if live else 'AVALIAÇÃO DE TRIAGEM (offline)'}")
    if live:
        print(f"  modelo: {report.modelo}")
    print("=" * 70)

    print(f"\n  Total: {report.acertos}/{len(report.resultados)} ({report.taxa:.0f}%)\n")
    print(f"  {'categoria':<18} {'acerto':>10}")
    print(f"  {'-' * 18} {'-' * 10}")
    for categoria, (ok, total) in sorted(report.por_categoria().items()):
        marca = "  ⚠️" if ok < total and categoria in SAFETY_CRITICAL else ""
        print(f"  {categoria:<18} {ok:>4}/{total:<5}{marca}")

    if live and report.chamadas:
        entrada, saida = report.tokens_medios
        print(f"\n  Fundamentação: {report.taxa_fundamentacao:.0f}% das respostas citam a base")
        print(f"  Latência: p50 {report.percentil(0.5)} ms | p95 {report.percentil(0.95)} ms")
        print(f"  Tokens por pergunta (medidos): {entrada} entrada, {saida} saída")

        custo = custo_por_pergunta(report.modelo, entrada, saida, precos)
        if custo is None:
            print(f"  Custo: sem preço para '{report.modelo}' em evals/precos.yaml")
        else:
            print(
                f"  Custo: US$ {custo:.5f}/pergunta → "
                f"US$ {custo * interacoes:.2f}/mês a {interacoes:,} interações"
                .replace(",", ".")
            )

    if report.falhas:
        print(f"\n  {len(report.falhas)} falha(s):\n")
        for r in report.falhas:
            critico = " [CRÍTICO]" if r.esperado in SAFETY_CRITICAL else ""
            print(f"    ✗ {r.case_id}{critico}: {r.pergunta[:55]}")
            print(f"      {r.detalhe}")
            if r.resposta:
                print(f"      resposta: {r.resposta[:100]}...")
        print()

    if report.falhas_criticas:
        print("  ⛔ FALHA EM CASO DE EMERGÊNCIA. Não faça deploy.\n")
    elif not report.falhas:
        print("\n  ✅ Tudo passou.\n")


def imprimir_comparacao(reports: list[Report], precos: dict, interacoes: int) -> None:
    print("\n" + "=" * 78)
    print("  COMPARAÇÃO ENTRE MODELOS")
    print("=" * 78 + "\n")

    largura = max(len(r.modelo) for r in reports) + 2

    def linha(rotulo: str, valores: list[str]) -> None:
        print(f"  {rotulo:<26}" + "".join(f"{v:>{largura + 8}}" for v in valores))

    linha("", [r.modelo for r in reports])
    print("  " + "─" * 74)
    linha("acerto geral", [f"{r.acertos}/{len(r.resultados)} ({r.taxa:.0f}%)" for r in reports])
    linha("emergência", [_categoria(r, "emergencia") for r in reports])
    linha("fora de escopo", [_categoria(r, "fora_de_escopo") for r in reports])
    linha("fundamentação", [f"{r.taxa_fundamentacao:.0f}%" for r in reports])
    linha("citações por resposta", [f"{r.citacoes_medias:.1f}" for r in reports])
    linha(
        "segurança (sem dose)",
        [f"{r.seguranca_conteudo[0]}/{r.seguranca_conteudo[1]}" for r in reports],
    )
    print("  " + "─" * 74)
    linha("latência p50", [f"{r.percentil(0.5)} ms" for r in reports])
    linha("latência p95", [f"{r.percentil(0.95)} ms" for r in reports])
    print("  " + "─" * 74)
    linha("tokens entrada (média)", [f"{r.tokens_medios[0]}" for r in reports])
    linha("tokens saída (média)", [f"{r.tokens_medios[1]}" for r in reports])

    custos = []
    for r in reports:
        entrada, saida = r.tokens_medios
        custos.append(custo_por_pergunta(r.modelo, entrada, saida, precos))

    linha(
        "custo/pergunta",
        [f"US$ {c:.5f}" if c is not None else "sem preço" for c in custos],
    )
    linha(
        f"custo/mês ({interacoes:,})".replace(",", "."),
        [f"US$ {c * interacoes:.2f}" if c is not None else "—" for c in custos],
    )
    print()

    _veredito_comparacao(reports, custos, interacoes)


def _categoria(report: Report, nome: str) -> str:
    ok, total = report.por_categoria().get(nome, (0, 0))
    return f"{ok}/{total}" if total else "—"


def _veredito_comparacao(reports: list[Report], custos: list[float | None], interacoes: int) -> None:
    print("  " + "═" * 74)

    criticos = [r for r in reports if r.falhas_criticas]
    if criticos:
        nomes = ", ".join(r.modelo for r in criticos)
        print(f"  ⛔ Falha em emergência: {nomes}. Descarte antes de olhar custo.\n")
        return

    melhor = max(reports, key=lambda r: (r.taxa, -r.percentil(0.95)))
    base = reports[0]

    print(f"  Melhor qualidade: {melhor.modelo} ({melhor.taxa:.0f}%)")

    if melhor is not base:
        delta_qualidade = melhor.taxa - base.taxa
        indice_melhor = reports.index(melhor)
        c_base, c_melhor = custos[0], custos[indice_melhor]

        if c_base and c_melhor:
            delta_mes = (c_melhor - c_base) * interacoes
            print(
                f"  Contra {base.modelo}: {delta_qualidade:+.0f} pontos de acerto "
                f"por US$ {delta_mes:+.2f}/mês"
            )
            if delta_qualidade <= 0:
                print("  → O mais caro não entregou qualidade maior. Fique no atual.")
            elif abs(delta_mes) < 50:
                print("  → A diferença de custo é irrelevante nesta escala. Troque.")
    else:
        print("  → O modelo atual já é o melhor do conjunto avaliado.")

    # Fundamentação alta em todos significa que a recuperação está boa e que
    # esta métrica parou de discriminar. Daí em diante, número não decide.
    if all(r.taxa_fundamentacao >= 95 for r in reports):
        print()
        print("  ⓘ Fundamentação saturada (≥95% em todos). A recuperação está boa,")
        print("    e esta métrica não separa mais os modelos. O que ainda decide:")
        print("    a linha de segurança acima, e revisão humana das respostas:")
        print("      python evals/run_eval.py --comparar … --relatorio-cego revisao.md")

    print()


def escrever_relatorio(reports: list[Report], caminho: Path, interacoes: int, precos: dict) -> None:
    """Respostas lado a lado, para julgar o que número nenhum mede."""
    linhas = ["# Comparação de modelos", ""]
    linhas.append(f"Conjunto dourado: {len(reports[0].resultados)} casos.")
    linhas.append("")
    linhas.append("| modelo | acerto | fundamentação | p95 | custo/mês |")
    linhas.append("|---|---|---|---|---|")

    for r in reports:
        entrada, saida = r.tokens_medios
        custo = custo_por_pergunta(r.modelo, entrada, saida, precos)
        valor = f"US$ {custo * interacoes:.2f}" if custo is not None else "—"
        linhas.append(
            f"| `{r.modelo}` | {r.acertos}/{len(r.resultados)} ({r.taxa:.0f}%) "
            f"| {r.taxa_fundamentacao:.0f}% | {r.percentil(0.95)} ms | {valor} |"
        )

    linhas.append("")
    linhas.append("> ⚠️ Estes números medem consistência e custo, não **correção clínica**.")
    linhas.append("> As respostas abaixo precisam ser lidas por uma consultora de")
    linhas.append("> amamentação — é ela quem decide qual modelo responde melhor.")
    linhas.append("")
    linhas.append("---")
    linhas.append("")
    linhas.append("## Respostas lado a lado")

    por_caso: dict[str, list[tuple[str, Result]]] = {}
    for r in reports:
        for resultado in r.resultados:
            if resultado.resposta:
                por_caso.setdefault(resultado.case_id, []).append((r.modelo, resultado))

    for case_id, entradas in sorted(por_caso.items()):
        pergunta = entradas[0][1].pergunta
        linhas.append("")
        linhas.append(f"### {case_id} — {pergunta}")
        for modelo, resultado in entradas:
            marca = "✅" if resultado.ok else "❌"
            linhas.append("")
            linhas.append(f"**`{modelo}`** {marca} · {resultado.latencia_ms} ms · "
                          f"{resultado.citacoes} citação(ões)")
            linhas.append("")
            for paragrafo in resultado.resposta.split("\n"):
                linhas.append(f"> {paragrafo}")

    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"  Relatório gravado em {caminho}\n")


def escrever_relatorio_cego(reports: list[Report], caminho: Path, semente: int = 0) -> None:
    """Respostas anonimizadas e embaralhadas, para julgamento sem viés.

    Duas decisões de método:

    - Os modelos aparecem como A e B, e a ordem é **sorteada por pergunta**.
      Sem isso, quem revisa aprende que "A é sempre o de cima" e a preferência
      passa a medir posição, não qualidade.
    - Ninguém vê o nome do modelo. Saber que um é "o mais novo e mais caro"
      contamina o julgamento — e é justamente o julgamento que estamos
      tentando isolar.

    A chave fica num arquivo separado, para conferir só depois de preencher.
    """
    import json
    import random

    # Embaralhamento de apresentação, não de criptografia. Semente fixa para
    # o mesmo relatório poder ser regerado igual.
    sorteio = random.Random(semente or 42)  # noqa: S311

    por_caso: dict[str, dict[str, Result]] = {}
    for r in reports:
        for resultado in r.resultados:
            if resultado.resposta:
                por_caso.setdefault(resultado.case_id, {})[r.modelo] = resultado

    rotulos = ["A", "B", "C", "D"]
    chave: dict[str, dict[str, str]] = {}

    # Ordem balanceada, não sorteada. Com sorteio puro dá azar: numa amostra
    # de 25 casos é comum um modelo cair na posição A em 17 deles, e aí a
    # posição vira uma pista. Aqui metade das perguntas usa cada ordem, e o
    # sorteio só decide *quais* — o equilíbrio é garantido.
    ids_ordenados = sorted(cid for cid, e in por_caso.items() if len(e) >= 2)
    metade = len(ids_ordenados) // 2
    orientacoes = [False] * (len(ids_ordenados) - metade) + [True] * metade
    sorteio.shuffle(orientacoes)
    inverter = dict(zip(ids_ordenados, orientacoes, strict=True))

    linhas = [
        "# Revisão cega de modelos",
        "",
        "Para a consultora de amamentação preencher.",
        "",
        "Os nomes dos modelos estão ocultos **de propósito** — saber qual é o mais",
        "novo contamina o julgamento. A ordem também muda a cada pergunta.",
        "",
        "Para cada pergunta, marque `[x]` na resposta melhor **clinicamente**:",
        "mais correta, mais completa, mais acolhedora e mais segura. Se as duas",
        "servirem igualmente, marque empate.",
        "",
        "Depois de preencher tudo:",
        "",
        "```bash",
        f"python evals/apurar.py {caminho.name}",
        "```",
        "",
        "---",
    ]

    for case_id in sorted(por_caso):
        entradas = list(por_caso[case_id].items())
        if len(entradas) < 2:
            continue

        entradas.sort(key=lambda item: item[0])
        if inverter[case_id]:
            entradas.reverse()
        pergunta = entradas[0][1].pergunta

        chave[case_id] = {
            rotulos[i]: modelo for i, (modelo, _) in enumerate(entradas)
        }

        linhas.append("")
        linhas.append(f"## {case_id}")
        linhas.append("")
        linhas.append(f"**Pergunta:** {pergunta}")

        for i, (_, resultado) in enumerate(entradas):
            linhas.append("")
            linhas.append(f"**Resposta {rotulos[i]}:**")
            linhas.append("")
            for paragrafo in resultado.resposta.split("\n"):
                linhas.append(f"> {paragrafo}")

        linhas.append("")
        marcacoes = "  ".join(
            f"- [ ] {rotulos[i]} melhor" for i in range(len(entradas))
        )
        linhas.append(f"{marcacoes}  - [ ] empate")
        linhas.append("")
        linhas.append("Comentário: ")
        linhas.append("")
        linhas.append("---")

    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")

    caminho_chave = caminho.with_suffix(".chave.json")
    caminho_chave.write_text(
        json.dumps(chave, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"  Revisão cega em {caminho}")
    print(f"  Chave (não abra antes de preencher) em {caminho_chave}\n")


# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="chama o modelo de verdade")
    parser.add_argument(
        "--model",
        default=Settings().openai_model,
        help="modelo a avaliar (padrão: o mesmo do .env)",
    )
    parser.add_argument(
        "--comparar", default="", help="lista de modelos separados por vírgula"
    )
    parser.add_argument("--relatorio", default="", help="grava markdown com as respostas")
    parser.add_argument(
        "--relatorio-cego",
        default="",
        help="grava markdown anonimizado e embaralhado, para revisão humana",
    )
    parser.add_argument(
        "--interacoes-mes", type=int, default=INTERACOES_MES_PADRAO,
        help="volume mensal usado para projetar custo",
    )
    args = parser.parse_args()

    casos = carregar_casos()
    precos = carregar_precos()
    print(f"\n{len(casos)} casos carregados de {GOLDEN_SET.name}")

    if args.comparar:
        modelos = [m.strip() for m in args.comparar.split(",") if m.strip()]
        if len(modelos) < 2:
            print("ERRO: --comparar precisa de pelo menos dois modelos.")
            return 2

        reports = []
        for modelo in modelos:
            print(f"\nRodando {modelo}…")
            reports.append(avaliar_live(casos, modelo, silencioso=True))

        imprimir_comparacao(reports, precos, args.interacoes_mes)

        if args.relatorio:
            escrever_relatorio(reports, Path(args.relatorio), args.interacoes_mes, precos)

        if args.relatorio_cego:
            escrever_relatorio_cego(reports, Path(args.relatorio_cego))

        return 1 if any(r.falhas_criticas for r in reports) else 0

    if args.live:
        print(f"Rodando contra {args.model}…\n")
        report = avaliar_live(casos, args.model)
    else:
        report = avaliar_offline(casos)

    imprimir(report, args.live, precos, args.interacoes_mes)

    if report.falhas_criticas:
        return 1
    return 0 if not report.falhas else 1


if __name__ == "__main__":
    sys.exit(main())
