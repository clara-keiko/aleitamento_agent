#!/usr/bin/env python3
"""Calculadora de custo do agente, por canal e por escala.

Substitui a tabela fixa do OPERACAO.md §3, que envelhece a cada mudança de
preço ou de câmbio. Aqui as premissas são argumentos: você refaz a conta em
vez de reescrever o documento.

Duas correções em relação àquela tabela:

1. **gpt-5-mini raciocina antes de responder.** Os tokens de raciocínio são
   cobrados como saída e não aparecem na resposta. A conta antiga, feita para
   gpt-4o-mini, subestima a saída em ~2×.

2. **BSP cobra por mensagem enviada, e uma resposta vira várias mensagens.**
   O pipeline quebra resposta longa em balões (`split_message`). Com a Meta
   direta isso é irrelevante — mensagem de serviço é grátis. Com BSP,
   multiplica a fatura pelo número de balões.

    python3 scripts/custos.py
    python3 scripts/custos.py --maes 500 --canal twilio
    python3 scripts/custos.py --maes 1000 --plano 29.90 --conversao 0.05

⚠️ Os preços de BSP e a cotação do dólar vêm de fontes secundárias e não
foram verificados. Confirme antes de usar em proposta. Os preços de modelo
saem de evals/precos.yaml, que tem o mesmo aviso.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PRECOS = RAIZ / "evals" / "precos.yaml"

# ── premissas de consumo ──────────────────────────────────────────────────
# Estimativas. Para medir de verdade:  python3 evals/run_eval.py --live
TOKENS_ENTRADA = 6_000      # sistema + histórico + trechos do file_search
TOKENS_VISIVEIS = 250       # a resposta que a mãe lê
TOKENS_RACIOCINIO = 300     # gpt-5-mini com effort=low; cobrado como saída
CUSTO_TRANSCRICAO_MIN = 0.003   # ⚠️ US$/min, gpt-4o-mini-transcribe
DURACAO_AUDIO_MIN = 0.5         # nota de voz típica: 30 s

# ── preços de canal, US$ por mensagem enviada ─────────────────────────────
# ⚠️ Todos de memória. Confirme nas páginas dos fornecedores.
CANAIS = {
    "meta": {
        "rotulo": "Meta Cloud API (direto)",
        "por_mensagem": 0.0,
        "mensal_fixo": 0.0,
        "nota": "Mensagem de serviço (resposta em até 24h) é grátis desde jul/2025.",
    },
    "twilio-sandbox": {
        "rotulo": "Twilio Sandbox",
        "por_mensagem": 0.005,
        "mensal_fixo": 0.0,
        "nota": "Sem número próprio; a mãe manda um código de entrada antes.",
    },
    "twilio": {
        "rotulo": "Twilio (número próprio)",
        "por_mensagem": 0.005,
        "mensal_fixo": 1.15,
        "nota": "Taxa da Twilio por mensagem, além do que a Meta cobrar.",
    },
    "360dialog": {
        "rotulo": "360dialog",
        "por_mensagem": 0.0,
        "mensal_fixo": 53.0,
        "nota": "Assinatura mensal, sem taxa por mensagem. Vira barato em volume.",
    },
}

HOSPEDAGEM = {"piloto": 7.0, "operacao": 7.0, "escala": 25.0}


def carregar_precos() -> dict:
    """Lê evals/precos.yaml sem exigir PyYAML — o formato é plano."""
    if not PRECOS.exists():
        sys.exit(f"Não achei {PRECOS}")

    precos: dict[str, dict[str, float]] = {}
    modelo_atual = None
    for linha in PRECOS.read_text(encoding="utf-8").splitlines():
        sem_comentario = linha.split("#")[0].rstrip()
        if not sem_comentario.strip():
            continue
        if not sem_comentario.startswith(" "):
            modelo_atual = sem_comentario.rstrip(":").strip()
            precos[modelo_atual] = {}
        elif modelo_atual:
            chave, _, valor = sem_comentario.partition(":")
            try:
                precos[modelo_atual][chave.strip()] = float(valor)
            except ValueError:
                continue
    return precos


def custo_openai(modelo: str, precos: dict, com_raciocinio: bool) -> float:
    """US$ por interação respondida."""
    if modelo not in precos:
        sys.exit(f"Modelo {modelo!r} não está em {PRECOS.name}. "
                 f"Disponíveis: {', '.join(sorted(precos))}")
    tabela = precos[modelo]
    saida = TOKENS_VISIVEIS + (TOKENS_RACIOCINIO if com_raciocinio else 0)
    return (TOKENS_ENTRADA / 1e6 * tabela["entrada"]
            + saida / 1e6 * tabela["saida"])


def _linha(rotulo: str, valor: str, largura: int = 34) -> str:
    return f"  {rotulo:.<{largura}} {valor:>14}"


def analisar(args: argparse.Namespace, precos: dict) -> None:
    canal = CANAIS[args.canal]
    interacoes = args.maes * args.interacoes_por_mae
    raciocina = args.modelo.startswith(("gpt-5", "o1", "o3", "o4"))

    por_interacao = custo_openai(args.modelo, precos, raciocina)
    texto = por_interacao * interacoes
    transcricao = (interacoes * args.audio * DURACAO_AUDIO_MIN
                   * CUSTO_TRANSCRICAO_MIN)
    mensagens = interacoes * args.mensagens_por_resposta
    canal_custo = mensagens * canal["por_mensagem"] + canal["mensal_fixo"]

    if args.maes <= 200:
        hospedagem = HOSPEDAGEM["piloto"]
    elif args.maes <= 5000:
        hospedagem = HOSPEDAGEM["operacao"]
    else:
        hospedagem = HOSPEDAGEM["escala"]

    total = texto + transcricao + canal_custo + hospedagem
    reais = total * args.cambio

    print(f"\n{'═' * 52}")
    print(f"  {args.maes:,} mães · {interacoes:,} interações/mês".replace(",", "."))
    print(f"  modelo {args.modelo} · canal {canal['rotulo']}")
    print(f"{'═' * 52}")

    print("\nPor interação")
    print(_linha("entrada (6.000 tokens)", f"US$ {TOKENS_ENTRADA / 1e6 * precos[args.modelo]['entrada']:.5f}"))
    visivel = TOKENS_VISIVEIS / 1e6 * precos[args.modelo]["saida"]
    print(_linha(f"saída visível ({TOKENS_VISIVEIS} tokens)", f"US$ {visivel:.5f}"))
    if raciocina:
        oculto = TOKENS_RACIOCINIO / 1e6 * precos[args.modelo]["saida"]
        print(_linha(f"raciocínio ({TOKENS_RACIOCINIO} tokens, oculto)",
                     f"US$ {oculto:.5f}"))
    print(_linha("TOTAL por pergunta", f"US$ {por_interacao:.5f}"))

    print("\nMensal")
    print(_linha("OpenAI — texto", f"US$ {texto:.2f}"))
    print(_linha(f"OpenAI — transcrição ({args.audio:.0%} em áudio)",
                 f"US$ {transcricao:.2f}"))
    rotulo_canal = "canal — grátis" if canal_custo == 0 else "canal"
    print(_linha(rotulo_canal, f"US$ {canal_custo:.2f}"))
    if canal["por_mensagem"]:
        print(f"      ({mensagens:,} mensagens = {interacoes:,} respostas × "
              f"{args.mensagens_por_resposta} balões)".replace(",", "."))
    print(_linha("hospedagem (Render)", f"US$ {hospedagem:.2f}"))
    print(_linha("base vetorial", "US$ 0.00"))
    print(_linha("TOTAL", f"US$ {total:.2f}"))
    print(_linha(f"em reais (câmbio {args.cambio:.2f})", f"R$ {reais:.2f}"))
    print(_linha("por mãe/mês", f"R$ {reais / args.maes:.2f}"))
    print(f"\n  {canal['nota']}")


def comparar_canais(args: argparse.Namespace, precos: dict) -> None:
    interacoes = args.maes * args.interacoes_por_mae
    mensagens = interacoes * args.mensagens_por_resposta

    print(f"\n{'─' * 52}\n  Canais, mesmo volume\n{'─' * 52}")
    for chave, canal in CANAIS.items():
        custo = mensagens * canal["por_mensagem"] + canal["mensal_fixo"]
        marca = " ←" if chave == args.canal else ""
        print(f"  {canal['rotulo']:<28} US$ {custo:>8.2f}/mês{marca}")
    print("\n  ⚠️ Preços de BSP não verificados — confirme antes de decidir.")


def unit_economics(args: argparse.Namespace, precos: dict) -> None:
    """Quanto sobra de um plano pago, e quantos assinantes pagam a conta."""
    if not args.plano:
        return

    raciocina = args.modelo.startswith(("gpt-5", "o1", "o3", "o4"))
    por_interacao = custo_openai(args.modelo, precos, raciocina)
    custo_mae_mes = (por_interacao * args.interacoes_por_mae
                     + args.interacoes_por_mae * args.audio
                     * DURACAO_AUDIO_MIN * CUSTO_TRANSCRICAO_MIN) * args.cambio

    print(f"\n{'─' * 52}\n  Plano de R$ {args.plano:.2f}/mês\n{'─' * 52}")
    print(_linha("receita por assinante", f"R$ {args.plano:.2f}"))
    print(_linha("custo de IA por assinante", f"R$ {custo_mae_mes:.2f}"))
    margem_tecnica = args.plano - custo_mae_mes
    print(_linha("margem antes de custo humano", f"R$ {margem_tecnica:.2f}"))
    print(f"\n  A IA consome {custo_mae_mes / args.plano:.1%} da mensalidade.")
    print("  O que decide o negócio é o custo humano, não o de inferência.")

    if args.hora_consultora:
        horas = margem_tecnica / args.hora_consultora
        print(f"\n  Com consultora a R$ {args.hora_consultora:.2f}/h, a margem")
        print(f"  paga {horas:.2f} h por assinante/mês "
              f"({horas * 60:.0f} min).")
        if horas * 60 < 20:
            print("  ⚠️ Menos de 20 min. Uma consulta mensal de verdade não cabe")
            print("     nesse preço — reveja o plano ou o que ele promete.")

    if args.conversao:
        pagantes = args.maes * args.conversao
        receita = pagantes * args.plano
        gratuitas = args.maes - pagantes
        custo_gratuitas = (custo_mae_mes * gratuitas
                           + HOSPEDAGEM["operacao"] * args.cambio)
        print(f"\n  Com {args.conversao:.0%} de conversão em {args.maes:,} mães:"
              .replace(",", "."))
        print(_linha("assinantes pagantes", f"{pagantes:,.0f}".replace(",", ".")))
        print(_linha("receita", f"R$ {receita:.2f}"))
        print(_linha("custo das não-pagantes", f"R$ {custo_gratuitas:.2f}"))
        print(_linha("sobra para custo humano", f"R$ {receita - custo_gratuitas:.2f}"))


def main() -> int:
    analisador = argparse.ArgumentParser(
        description="Calcula o custo do agente por canal e escala.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    analisador.add_argument("--maes", type=int, default=1000)
    analisador.add_argument("--interacoes-por-mae", type=int, default=12,
                            help="perguntas por mãe por mês (padrão 12)")
    analisador.add_argument("--audio", type=float, default=0.40,
                            help="fração das mensagens em áudio (padrão 0.40)")
    analisador.add_argument("--mensagens-por-resposta", type=int, default=2,
                            help="balões por resposta; só afeta BSP (padrão 2)")
    analisador.add_argument("--modelo", default="gpt-5-mini")
    analisador.add_argument("--canal", default="meta", choices=sorted(CANAIS))
    analisador.add_argument("--cambio", type=float, default=5.40,
                            help="⚠️ R$ por US$ — confirme a cotação do dia")
    analisador.add_argument("--plano", type=float,
                            help="mensalidade do plano pago, para unit economics")
    analisador.add_argument("--conversao", type=float,
                            help="fração das mães que assina (ex.: 0.05)")
    analisador.add_argument("--hora-consultora", type=float, default=150.0,
                            help="⚠️ custo da hora de consultora (padrão R$ 150)")
    args = analisador.parse_args()

    precos = carregar_precos()
    analisar(args, precos)
    comparar_canais(args, precos)
    unit_economics(args, precos)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
