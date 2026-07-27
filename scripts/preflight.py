#!/usr/bin/env python3
"""Pré-voo: roda isto ANTES da demo.

Verifica de ponta a ponta o que costuma falhar na hora errada — chave
inválida, vector store apagada, base vazia, latência alta — e termina com um
veredito GO / NO-GO.

    python scripts/preflight.py

Gasta poucos centavos: faz três consultas reais ao modelo.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI, OpenAIError  # noqa: E402

from app import guardrails  # noqa: E402
from app.config import Settings  # noqa: E402
from app.llm import AssistantEngine  # noqa: E402

VERDE, VERMELHO, AMARELO, CINZA, FIM = "\033[92m", "\033[91m", "\033[93m", "\033[90m", "\033[0m"
OK, FALHA, AVISO = f"{VERDE}✓{FIM}", f"{VERMELHO}✗{FIM}", f"{AMARELO}!{FIM}"

# Acima disto, a plateia percebe a espera.
LATENCIA_BOA_MS = 5000
LATENCIA_LIMITE_MS = 10000

PERGUNTAS_DE_TESTE = [
    "Como sei se a pega está correta?",
    "Preciso dar mamada de 3 em 3 horas?",
    "Como aumentar a produção de leite?",
]


class Resultado:
    def __init__(self):
        self.bloqueios: list[str] = []
        self.avisos: list[str] = []

    def bloqueio(self, msg: str) -> None:
        self.bloqueios.append(msg)

    def aviso(self, msg: str) -> None:
        self.avisos.append(msg)


def secao(titulo: str) -> None:
    print(f"\n{titulo}")
    print(CINZA + "─" * 58 + FIM)


def checar_config(r: Resultado) -> Settings | None:
    secao("1. Configuração")
    settings = Settings()

    for nome, valor in [
        ("OPENAI_API_KEY", settings.openai_api_key),
        ("VECTOR_STORE_ID", settings.vector_store_id),
    ]:
        if valor:
            print(f"  {OK} {nome} definida")
        else:
            print(f"  {FALHA} {nome} AUSENTE")
            r.bloqueio(f"{nome} não está definida")

    print(f"  {OK} modelo: {settings.openai_model}")
    print(f"  {OK} protótipo web: {'ligado' if settings.enable_web else 'DESLIGADO'}")
    if not settings.enable_web:
        r.bloqueio("ENABLE_WEB está desligado — /chat não vai abrir")

    if settings.web_access_code:
        print(f"  {OK} código de acesso definido")
    else:
        print(f"  {AVISO} sem WEB_ACCESS_CODE — ok para demo local, "
              "arriscado numa URL pública")
        r.aviso("defina WEB_ACCESS_CODE se o link for ficar exposto")

    return settings if not r.bloqueios else None


def checar_base(settings: Settings, r: Resultado) -> None:
    secao("2. Base de conhecimento")
    client = OpenAI(api_key=settings.openai_api_key, timeout=30)

    try:
        store = client.vector_stores.retrieve(vector_store_id=settings.vector_store_id)
    except OpenAIError as exc:
        print(f"  {FALHA} não consegui abrir a vector store: {exc}")
        r.bloqueio("VECTOR_STORE_ID inválido — rode: python ingest_openai_kb.py")
        return

    contagem = store.file_counts
    print(f"  {OK} store '{store.name}' encontrada")
    print(f"     {contagem.completed} arquivo(s) prontos, "
          f"{contagem.failed} com falha, {contagem.in_progress} processando")

    if contagem.completed == 0:
        print(f"  {FALHA} base VAZIA — o agente vai recusar tudo")
        r.bloqueio("base vazia — rode: python ingest_openai_kb.py")
    if contagem.failed:
        r.aviso(f"{contagem.failed} arquivo(s) falharam; a base está incompleta")
    if contagem.in_progress:
        r.aviso(f"{contagem.in_progress} arquivo(s) ainda processando; espere terminar")


def checar_guardrails(r: Resultado) -> None:
    secao("3. Triagem clínica (sem custo)")

    casos = [
        ("meu bebe nao respira", guardrails.EMERGENCY_NOW, "emergência"),
        ("meu bebê está com febre desde ontem", guardrails.REFER_MEDICAL_CARE,
         "encaminhamento"),
        ("posso amamentar com febre?", guardrails.EDUCATIONAL_OK, "educativo + nota"),
    ]

    for texto, esperado, rotulo in casos:
        risco = guardrails.classify_risk(texto)
        if risco.level == esperado:
            print(f"  {OK} {rotulo}")
        else:
            print(f"  {FALHA} {rotulo}: esperado {esperado}, obtido {risco.level}")
            r.bloqueio(f"guardrail de {rotulo} quebrado")

    if guardrails.is_small_talk("obrigada"):
        print(f"  {OK} saudação")
    else:
        print(f"  {FALHA} saudação não reconhecida")
        r.aviso("small talk quebrado — 'obrigada' vai receber recusa")


def checar_respostas(settings: Settings, r: Resultado) -> None:
    secao("4. Respostas reais (consome cota)")
    engine = AssistantEngine(settings)
    latencias = []

    for pergunta in PERGUNTAS_DE_TESTE:
        inicio = time.monotonic()
        resposta = engine.answer(pergunta)
        ms = int((time.monotonic() - inicio) * 1000)
        latencias.append(ms)

        curta = pergunta[:38] + ("…" if len(pergunta) > 38 else "")

        if resposta.error:
            print(f"  {FALHA} {curta:<40} ERRO na chamada")
            r.bloqueio("chamada ao modelo falhou — verifique a chave e a cota")
        elif not resposta.grounded:
            print(f"  {FALHA} {curta:<40} sem citação da base")
            r.bloqueio(f"'{pergunta}' seria recusada na demo — recuperação ruim")
        else:
            cor = VERDE if ms < LATENCIA_BOA_MS else AMARELO
            print(f"  {OK} {curta:<40} {cor}{ms:>6} ms{FIM}")

    if not latencias:
        return

    pior = max(latencias)
    media = sum(latencias) // len(latencias)
    print(f"\n     média {media} ms · pior {pior} ms")

    if pior > LATENCIA_LIMITE_MS:
        print(f"  {FALHA} lento demais para demo ao vivo")
        r.bloqueio(f"latência de {pior} ms — reduza MAX_RETRIEVAL_RESULTS")
    elif pior > LATENCIA_BOA_MS:
        print(f"  {AVISO} perceptível; avise a plateia que está consultando a base")
        r.aviso(f"pior latência {pior} ms")


def veredito(r: Resultado) -> int:
    print("\n" + "═" * 58)
    if r.bloqueios:
        print(f"  {VERMELHO}NO-GO — resolva antes da demo:{FIM}\n")
        for item in r.bloqueios:
            print(f"    ✗ {item}")
        print()
        return 1

    print(f"  {VERDE}GO — pronto para a demo.{FIM}")
    if r.avisos:
        print(f"\n  {AMARELO}Atenção:{FIM}")
        for item in r.avisos:
            print(f"    ! {item}")
    print("\n  Lembretes:")
    print("    · Abra /chat e mande uma mensagem 10 min antes (aquece o serviço)")
    print("    · Deixe o roteiro aberto: docs/DEMO.md")
    print("    · Plano B: rodar local com uvicorn, sem depender do Render")
    print()
    return 0


def main() -> int:
    print("\n" + "═" * 58)
    print("  PRÉ-VOO DA DEMO")
    print("═" * 58)

    r = Resultado()
    settings = checar_config(r)

    if settings is None:
        return veredito(r)

    checar_base(settings, r)
    checar_guardrails(r)

    if os.getenv("PREFLIGHT_SKIP_LIVE", "").lower() not in {"1", "true"}:
        checar_respostas(settings, r)
    else:
        secao("4. Respostas reais")
        print(f"  {AVISO} pulado por PREFLIGHT_SKIP_LIVE")
        r.aviso("as respostas reais não foram testadas")

    return veredito(r)


if __name__ == "__main__":
    sys.exit(main())
