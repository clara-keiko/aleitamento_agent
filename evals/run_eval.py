#!/usr/bin/env python3
"""Avaliação do agente contra o conjunto dourado.

Existe para responder com medição, não com opinião, três perguntas:

  1. A triagem clínica ainda está correta depois desta mudança?
  2. O modelo responde a partir da base ou está inventando?
  3. Trocar de modelo melhora alguma coisa, e a que custo?

Dois modos:

    python evals/run_eval.py
        Só a triagem (guardrails). Não chama a OpenAI, não custa nada,
        roda em CI. É a camada crítica de segurança.

    python evals/run_eval.py --live
        Chama o modelo de verdade. Mede fundamentação, recusa fora de
        escopo, latência e custo por pergunta.

    python evals/run_eval.py --live --model gpt-5-mini
        O mesmo, com outro modelo. Rode duas vezes e compare as tabelas —
        é assim que a escolha de modelo vira decisão baseada em dado.

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

GOLDEN_SET = Path(__file__).parent / "golden_set.yaml"

# Preço por 1M de tokens (USD). Confirme antes de citar em proposta —
# tabela de julho de 2026.
PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5.4-mini": (0.75, 4.50),
}

SAFETY_CRITICAL = {"emergencia"}


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
    resposta: str = ""


@dataclass
class Report:
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


def carregar_casos() -> list[dict]:
    with open(GOLDEN_SET, encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def avaliar_offline(casos: list[dict]) -> Report:
    """Só a triagem. Casos que dependem do modelo são pulados."""
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
        else:
            ok = obtido == esperado
            detalhe = "" if ok else f"esperado '{esperado}', obtido '{obtido}'"

        report.resultados.append(
            Result(caso["id"], caso["pergunta"], esperado, obtido, ok, detalhe)
        )

    return report


def avaliar_live(casos: list[dict], model: str) -> Report:
    settings = Settings(openai_model=model)
    faltando = [v for v in ("OPENAI_API_KEY", "VECTOR_STORE_ID") if v in settings.missing()]
    if faltando:
        print(f"ERRO: defina {', '.join(faltando)} para rodar com --live.")
        sys.exit(2)

    engine = AssistantEngine(settings)
    report = Report()

    for caso in casos:
        esperado = caso["outcome"]
        obtido, nota = triagem(caso["pergunta"])

        if obtido != "modelo":
            ok = obtido == esperado
            detalhe = "" if ok else f"esperado '{esperado}', obtido '{obtido}'"
            report.resultados.append(
                Result(caso["id"], caso["pergunta"], esperado, obtido, ok, detalhe)
            )
            print(f"  {'✓' if ok else '✗'} {caso['id']:<8} {obtido}")
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

        # Checagem de conteúdo proibido (ex.: dose de medicamento).
        if ok and caso.get("nao_deve_conter"):
            texto = guardrails.normalize(resposta.text)
            achados = [
                termo
                for termo in caso["nao_deve_conter"]
                if guardrails.normalize(termo) in texto
            ]
            if achados:
                ok, detalhe = False, f"conteúdo proibido: {', '.join(achados)}"

        if ok and caso.get("nota_de_seguranca") and not nota:
            ok, detalhe = False, "faltou a nota de segurança"

        resultado = Result(
            caso["id"], caso["pergunta"], esperado, obtido, ok, detalhe,
            latencia_ms=latencia, resposta=resposta.text,
        )
        report.resultados.append(resultado)
        print(f"  {'✓' if ok else '✗'} {caso['id']:<8} {obtido:<16} {latencia:>6} ms")

    return report


def imprimir(report: Report, model: str, live: bool) -> None:
    total = len(report.resultados)
    acertos = total - len(report.falhas)

    print("\n" + "=" * 70)
    print(f"  {'AVALIAÇÃO COMPLETA' if live else 'AVALIAÇÃO DE TRIAGEM (offline)'}")
    if live:
        print(f"  modelo: {model}")
    print("=" * 70)

    print(f"\n  Total: {acertos}/{total} ({100 * acertos / total:.0f}%)\n")

    print(f"  {'categoria':<18} {'acerto':>10}")
    print(f"  {'-' * 18} {'-' * 10}")
    for categoria, (ok, tot) in sorted(report.por_categoria().items()):
        marca = "  ⚠️" if ok < tot and categoria in SAFETY_CRITICAL else ""
        print(f"  {categoria:<18} {ok:>4}/{tot:<5}{marca}")

    if live:
        latencias = sorted(r.latencia_ms for r in report.resultados if r.latencia_ms)
        if latencias:
            p50 = latencias[len(latencias) // 2]
            p95 = latencias[int(len(latencias) * 0.95) - 1]
            print(f"\n  Latência: p50 {p50} ms | p95 {p95} ms | máx {latencias[-1]} ms")

            if model in PRICING:
                entrada, saida = PRICING[model]
                # Estimativa com o perfil de uso descrito em docs/OPERACAO.md.
                custo = (6000 * entrada + 250 * saida) / 1_000_000
                print(f"  Custo estimado: US$ {custo:.5f}/pergunta "
                      f"→ US$ {custo * 12000:.2f}/mês a 12 mil perguntas")

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="chama o modelo de verdade")
    parser.add_argument("--model", default="gpt-4o-mini", help="modelo a avaliar")
    args = parser.parse_args()

    casos = carregar_casos()
    print(f"\n{len(casos)} casos carregados de {GOLDEN_SET.name}")

    if args.live:
        print(f"Rodando contra {args.model}...\n")
        report = avaliar_live(casos, args.model)
    else:
        report = avaliar_offline(casos)

    imprimir(report, args.model, args.live)

    if report.falhas_criticas:
        return 1
    return 0 if not report.falhas else 1


if __name__ == "__main__":
    sys.exit(main())
