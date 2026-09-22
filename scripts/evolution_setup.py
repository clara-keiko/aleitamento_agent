#!/usr/bin/env python3
"""Prepara a instância da Evolution e pareia o número, por comando.

O console da Evolution é enxuto e a ordem das coisas importa: a instância
precisa existir, o webhook precisa apontar para o agente, e só então o QR vale
a pena. Este script faz os três na ordem e mostra o estado entre eles.

Biblioteca padrão apenas, Python 3.9+. Roda na sua máquina.

    export EVOLUTION_BASE_URL='https://sua-evolution.onrender.com'
    export EVOLUTION_API_KEY='a-chave-que-voce-gerou'
    export EVOLUTION_INSTANCE='lactai'
    export AGENT_WEBHOOK_URL='https://aleitamento-agent-cmb6.onrender.com/webhook'

    python3 scripts/evolution_setup.py criar
    python3 scripts/evolution_setup.py qr        # salva e abre qr.png
    python3 scripts/evolution_setup.py estado
    python3 scripts/evolution_setup.py testar 5521999999999

⚠️ As rotas abaixo vêm do que se conhece da v2 e não foram conferidas contra a
documentação corrente. Erro da Evolution é impresso como veio — se uma rota
mudou, a mensagem dela diz qual.

⚠️ Pareie um número que NÃO esteja registrado na Cloud API. Um número é ou
conta comum, ou Cloud API — nunca os dois.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request

QR_PADRAO = "qr.png"


class ErroDaEvolution(Exception):
    """Erro devolvido pela Evolution, já legível."""


def _exigir(nome: str) -> str:
    valor = os.environ.get(nome, "").strip()
    if not valor:
        sys.exit(f"Falta {nome} no ambiente. Veja o cabeçalho deste arquivo.")
    return valor


def _chamar(metodo: str, caminho: str, corpo: dict | None = None) -> dict:
    base = _exigir("EVOLUTION_BASE_URL").rstrip("/")
    chave = _exigir("EVOLUTION_API_KEY")
    url = f"{base}/{caminho.lstrip('/')}"

    if not url.startswith(("https://", "http://")):
        raise ErroDaEvolution(f"Destino inesperado: {url}")

    dados = json.dumps(corpo).encode() if corpo is not None else None
    pedido = urllib.request.Request(  # noqa: S310 - esquema conferido acima
        url,
        data=dados,
        method=metodo,
        headers={"apikey": chave, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(pedido, timeout=60) as resposta:  # noqa: S310
            texto = resposta.read().decode()
            return json.loads(texto) if texto.strip() else {}
    except urllib.error.HTTPError as erro:
        detalhe = erro.read().decode(errors="replace")[:800]
        dica = ""
        if erro.code in (401, 403):
            dica = "\n  → EVOLUTION_API_KEY não confere com a chave do serviço."
        if erro.code == 404:
            dica = ("\n  → Rota ou instância inexistente. Confira o nome em"
                    " EVOLUTION_INSTANCE, ou rode 'criar' antes.")
        raise ErroDaEvolution(f"HTTP {erro.code} em {caminho}\n  {detalhe}{dica}") from erro
    except urllib.error.URLError as erro:
        raise ErroDaEvolution(
            f"Não consegui falar com a Evolution: {erro.reason}\n"
            f"  → A URL {url} está certa e o serviço está no ar?"
        ) from erro


def _mostrar(dados: dict) -> None:
    print(json.dumps(dados, indent=2, ensure_ascii=False))


# ── comandos ──────────────────────────────────────────────────────────────


def cmd_criar(args: argparse.Namespace) -> int:
    instancia = _exigir("EVOLUTION_INSTANCE")
    webhook = _exigir("AGENT_WEBHOOK_URL")

    corpo = {
        "instanceName": instancia,
        "qrcode": True,
        "integration": "WHATSAPP-BAILEYS",
        "webhook": {
            "url": webhook,
            "byEvents": False,
            "base64": False,
            # Só o evento de mensagem nova. Assinar tudo faria o agente
            # receber status de entrega e presença, que ele descarta — mas
            # descartar também custa requisição.
            "events": ["MESSAGES_UPSERT"],
        },
    }

    try:
        resposta = _chamar("POST", "/instance/create", corpo)
    except ErroDaEvolution as erro:
        print(erro)
        print("\nSe disser que a instância já existe, siga para:")
        print("  python3 scripts/evolution_setup.py webhook")
        print("  python3 scripts/evolution_setup.py qr")
        return 1

    _mostrar(resposta)
    print("\nInstância criada. Agora: python3 scripts/evolution_setup.py qr")
    return 0


def cmd_webhook(args: argparse.Namespace) -> int:
    """Reaponta o webhook de uma instância que já existe."""
    instancia = _exigir("EVOLUTION_INSTANCE")
    webhook = _exigir("AGENT_WEBHOOK_URL")

    corpo = {
        "webhook": {
            "enabled": True,
            "url": webhook,
            "byEvents": False,
            "base64": False,
            "events": ["MESSAGES_UPSERT"],
        }
    }
    try:
        _mostrar(_chamar("POST", f"/webhook/set/{instancia}", corpo))
    except ErroDaEvolution as erro:
        print(erro)
        return 1

    print(f"\nWebhook apontado para {webhook}")
    return 0


def cmd_qr(args: argparse.Namespace) -> int:
    instancia = _exigir("EVOLUTION_INSTANCE")
    try:
        resposta = _chamar("GET", f"/instance/connect/{instancia}")
    except ErroDaEvolution as erro:
        print(erro)
        return 1

    bruto = resposta.get("base64") or resposta.get("qrcode", {}).get("base64") or ""
    if not bruto:
        _mostrar(resposta)
        estado = resposta.get("instance", {}).get("state")
        if estado == "open":
            print("\nJá está pareada — não há QR a mostrar.")
            return 0
        print("\nResposta sem QR. Veja acima o que veio.")
        return 1

    imagem = base64.b64decode(bruto.split(",")[-1])
    destino = pathlib.Path(args.arquivo)
    destino.write_bytes(imagem)
    print(f"QR salvo em {destino}")

    if sys.platform == "darwin":
        subprocess.run(["open", str(destino)], check=False)  # noqa: S603, S607

    print("\nNo celular do número que vai atender:")
    print("  WhatsApp → Aparelhos conectados → Conectar um aparelho")
    print("\nDepois confirme com: python3 scripts/evolution_setup.py estado")
    print("O QR expira em cerca de 1 minuto; rode 'qr' de novo se demorar.")
    return 0


def cmd_estado(args: argparse.Namespace) -> int:
    instancia = _exigir("EVOLUTION_INSTANCE")
    try:
        resposta = _chamar("GET", f"/instance/connectionState/{instancia}")
    except ErroDaEvolution as erro:
        print(erro)
        return 1

    _mostrar(resposta)
    estado = json.dumps(resposta)
    if '"open"' in estado:
        print("\n✅ Pareada e conectada.")
        print("Agora defina no agente (painel do Render):")
        print("  WHATSAPP_PROVIDER  = evolution")
        print(f"  EVOLUTION_BASE_URL = {os.environ.get('EVOLUTION_BASE_URL', '')}")
        print(f"  EVOLUTION_INSTANCE = {instancia}")
        print("  EVOLUTION_API_KEY  = a mesma chave do serviço")
    elif '"connecting"' in estado:
        print("\n⏳ Conectando. Rode 'qr' e escaneie.")
    else:
        print("\n❌ Desconectada. Rode 'qr' para parear de novo.")
    return 0


def cmd_testar(args: argparse.Namespace) -> int:
    """Manda uma mensagem, para provar o envio antes de envolver o agente."""
    instancia = _exigir("EVOLUTION_INSTANCE")
    try:
        resposta = _chamar(
            "POST",
            f"/message/sendText/{instancia}",
            {"number": args.numero, "text": args.texto},
        )
    except ErroDaEvolution as erro:
        print(erro)
        return 1

    _mostrar(resposta)
    print(f"\nSe chegou no {args.numero}, o envio está funcionando.")
    print("O que falta então é só apontar o agente para esta instância.")
    return 0


def main() -> int:
    analisador = argparse.ArgumentParser(
        description="Prepara a instância da Evolution e pareia o número.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = analisador.add_subparsers(dest="comando", required=True)

    sub.add_parser("criar", help="cria a instância já com o webhook do agente")
    sub.add_parser("webhook", help="reaponta o webhook de uma instância existente")

    p_qr = sub.add_parser("qr", help="baixa o QR de pareamento como imagem")
    p_qr.add_argument("--arquivo", default=QR_PADRAO)

    sub.add_parser("estado", help="mostra se a instância está conectada")

    p_teste = sub.add_parser("testar", help="envia uma mensagem de teste")
    p_teste.add_argument("numero", help="destino, só dígitos: 5521999999999")
    p_teste.add_argument("--texto", default="Teste do agente. Pode ignorar.")

    args = analisador.parse_args()
    comandos = {
        "criar": cmd_criar,
        "webhook": cmd_webhook,
        "qr": cmd_qr,
        "estado": cmd_estado,
        "testar": cmd_testar,
    }
    return comandos[args.comando](args)


if __name__ == "__main__":
    sys.exit(main())
