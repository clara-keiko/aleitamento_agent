#!/usr/bin/env python3
"""Administra a conta do WhatsApp pela Graph API, sem passar pela UI da Meta.

Por que existe: a interface do Gerenciador esconde o motivo das coisas. Quando
um nome de exibição é recusado, ela diz "não aprovado" e ponto. A API devolve
`error_user_msg`, que costuma dizer o que faltou. Este script existe para
mostrar essa mensagem.

Roda com a biblioteca padrão do Python 3.9+ — nada de `pip install`. É para ser
usado no seu Mac, não no servidor.

    export META_TOKEN='EAA...'           # System User, permissão de business
    export PHONE_NUMBER_ID='1303115646210874'
    export WABA_ID='...'                 # opcional
    export BUSINESS_ID='...'             # opcional, para os comandos de domínio

    python3 scripts/meta_admin.py status
    python3 scripts/meta_admin.py nome "LactAI"           # só simula
    python3 scripts/meta_admin.py nome "LactAI" --confirmar
    python3 scripts/meta_admin.py dominios
    python3 scripts/meta_admin.py dominio-add aleitamento.com.br

⚠️ O campo *Site* do portfólio (Informações da empresa) provavelmente não é
gravável pela Graph API — até onde sei, só pela UI. É um campo de texto só,
o mais fácil da interface inteira. Este script cobre o resto.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

VERSAO = os.environ.get("META_API_VERSION", "v25.0")
BASE = f"https://graph.facebook.com/{VERSAO}"

# O que cada estado de nome significa, e o que fazer a seguir. A UI mostra o
# código; o que falta é a segunda coluna.
ESTADOS_DE_NOME = {
    "APPROVED": "aprovado e em uso.",
    "AVAILABLE_WITHOUT_REVIEW": "liberado sem análise — já pode usar.",
    "PENDING_REVIEW": "em análise. Costuma levar de algumas horas a 2 dias.",
    "DECLINED": "RECUSADO. Veja o motivo no Gerenciador; em geral é a marca não"
                " aparecer publicamente ligada à razão social.",
    "EXPIRED": "expirou sem análise concluída — reenvie.",
    "NONE": "nenhum nome enviado para análise.",
}


class ErroDaMeta(Exception):
    """Erro devolvido pela Graph API, já legível."""


def _requisitar(metodo: str, caminho: str, params: dict | None = None) -> dict:
    token = os.environ.get("META_TOKEN", "").strip()
    if not token:
        sys.exit("Falta META_TOKEN no ambiente. Veja o cabeçalho deste arquivo.")

    params = dict(params or {})
    url = f"{BASE}/{caminho.lstrip('/')}"
    dados = None

    if metodo == "GET":
        params["access_token"] = token
        url = f"{url}?{urllib.parse.urlencode(params)}"
    else:
        params["access_token"] = token
        dados = urllib.parse.urlencode(params).encode()

    # O caminho vem de variáveis de ambiente; o prefixo é constante. A checagem
    # garante que continua sendo https para a Graph API e nada mais — o token
    # vai no corpo da requisição e não pode vazar para outro destino.
    if not url.startswith(BASE + "/"):
        raise ErroDaMeta(f"Destino inesperado: {url}")

    pedido = urllib.request.Request(url, data=dados, method=metodo)  # noqa: S310
    try:
        with urllib.request.urlopen(pedido, timeout=30) as resposta:  # noqa: S310
            return json.loads(resposta.read().decode())
    except urllib.error.HTTPError as erro:
        corpo = erro.read().decode(errors="replace")
        try:
            detalhe = json.loads(corpo).get("error", {})
        except json.JSONDecodeError:
            raise ErroDaMeta(f"HTTP {erro.code}: {corpo[:500]}") from erro
        raise ErroDaMeta(_formatar_erro(erro.code, detalhe)) from erro
    except urllib.error.URLError as erro:
        raise ErroDaMeta(f"Não consegui falar com a Graph API: {erro.reason}") from erro


def _formatar_erro(http: int, detalhe: dict) -> str:
    """A mensagem útil da Meta quase sempre está em error_user_msg."""
    linhas = [f"HTTP {http} · código {detalhe.get('code')}"]
    if detalhe.get("error_subcode"):
        linhas[0] += f" · subcódigo {detalhe['error_subcode']}"
    for campo in ("error_user_title", "error_user_msg", "message"):
        if detalhe.get(campo):
            linhas.append(f"  {detalhe[campo]}")
    if detalhe.get("code") == 190:
        linhas.append("  → Token inválido ou expirado. Se for o token temporário"
                      " de 24h, gere um de System User.")
    if detalhe.get("code") in (200, 10):
        linhas.append("  → O token não tem a permissão necessária"
                      " (whatsapp_business_management / business_management).")
    return "\n".join(linhas)


def _exigir(nome: str) -> str:
    valor = os.environ.get(nome, "").strip()
    if not valor:
        sys.exit(f"Falta {nome} no ambiente.")
    return valor


def _titulo(texto: str) -> None:
    print(f"\n{texto}\n{'─' * len(texto)}")


# ── comandos ──────────────────────────────────────────────────────────────


def cmd_status(args: argparse.Namespace) -> int:
    numero = _exigir("PHONE_NUMBER_ID")
    campos = ",".join([
        "verified_name", "display_phone_number", "name_status",
        "new_name_status", "quality_rating", "code_verification_status",
        "platform_type", "throughput",
    ])

    _titulo("Número")
    try:
        dados = _requisitar("GET", numero, {"fields": campos})
    except ErroDaMeta as erro:
        print(erro)
        return 1

    if args.json:
        print(json.dumps(dados, indent=2, ensure_ascii=False))
    else:
        print(f"  telefone       {dados.get('display_phone_number', '—')}")
        print(f"  nome atual     {dados.get('verified_name', '—')}")
        estado = dados.get("name_status", "NONE")
        print(f"  estado do nome {estado} — {ESTADOS_DE_NOME.get(estado, '?')}")
        if dados.get("new_name_status"):
            novo = dados["new_name_status"]
            print(f"  nome pendente  {novo} — {ESTADOS_DE_NOME.get(novo, '?')}")
        print(f"  qualidade      {dados.get('quality_rating', '—')}")

    if os.environ.get("WABA_ID"):
        _titulo("Conta comercial (WABA)")
        try:
            waba = _requisitar("GET", os.environ["WABA_ID"], {
                "fields": "name,business_verification_status,"
                          "owner_business_info,on_behalf_of_business_info",
            })
            print(json.dumps(waba, indent=2, ensure_ascii=False))
        except ErroDaMeta as erro:
            print(erro)

    return 0


def cmd_nome(args: argparse.Namespace) -> int:
    numero = _exigir("PHONE_NUMBER_ID")

    if not args.confirmar:
        print(f"Simulação. Enviaria o nome de exibição: {args.nome!r}")
        print("\nA Meta limita quantas vezes o nome pode ser reenviado, e cada")
        print("tentativa recusada custa uma delas. Confira a landing antes:")
        print("  · a marca aparece na página?")
        print("  · a razão social e o CNPJ estão no rodapé?")
        print("  · o campo *Site* do portfólio aponta para essa página?")
        print("\nQuando as três respostas forem sim, repita com --confirmar.")
        return 0

    try:
        resposta = _requisitar("POST", numero, {"new_display_name": args.nome})
    except ErroDaMeta as erro:
        print("Não foi aceito:\n")
        print(erro)
        return 1

    print(json.dumps(resposta, indent=2, ensure_ascii=False))
    print("\nEnviado. Acompanhe com: python3 scripts/meta_admin.py status")
    return 0


def cmd_descobrir(args: argparse.Namespace) -> int:
    """Descobre os IDs da conta e imprime o link direto de cada tela.

    Existe porque a navegação do Gerenciador é o ponto onde todo mundo trava:
    os nomes das telas mudam, e o mesmo item aparece em interfaces diferentes.
    Com os IDs em mãos, dá para pular o menu e ir pela URL.

    Cada etapa falha em silêncio parcial — imprime o erro e segue. Um token
    sem permissão de business ainda descobre o app e o número.
    """
    token = os.environ["META_TOKEN"].strip()
    app_id = None
    negocios: list[dict] = []

    _titulo("Token")
    try:
        info = _requisitar("GET", "debug_token", {"input_token": token}).get("data", {})
        app_id = info.get("app_id")
        print(f"  app_id     {app_id or '—'}")
        print(f"  tipo       {info.get('type', '—')}")
        expira = info.get("expires_at")
        if expira == 0:
            print("  validade   permanente (System User) ✅")
        elif expira:
            quando = datetime.datetime.fromtimestamp(expira)
            restante = quando - datetime.datetime.now()
            horas = restante.total_seconds() / 3600
            marca = "⚠️  TEMPORÁRIO" if horas < 48 else ""
            print(f"  validade   expira em {quando:%d/%m %H:%M} "
                  f"({horas:.0f}h) {marca}")
            if horas < 48:
                print("             → Gere um token de System User antes de"
                      " configurar o Render, ou o bot para em algumas horas.")
        escopos = info.get("scopes") or []
        print(f"  permissões {', '.join(escopos) if escopos else '—'}")
        for necessaria in ("whatsapp_business_management", "business_management"):
            if necessaria not in escopos:
                print(f"             ⚠️  falta {necessaria}")
    except ErroDaMeta as erro:
        print(erro)

    _titulo("Portfólios empresariais")
    try:
        negocios = _requisitar("GET", "me/businesses", {"fields": "id,name"}).get("data", [])
        for negocio in negocios:
            print(f"  {negocio.get('name')}  (id {negocio.get('id')})")
        if not negocios:
            print("  nenhum visível para este token")
    except ErroDaMeta as erro:
        print(erro)

    for negocio in negocios:
        try:
            wabas = _requisitar(
                "GET", f"{negocio['id']}/owned_whatsapp_business_accounts",
                {"fields": "id,name"},
            ).get("data", [])
        except ErroDaMeta as erro:
            print(erro)
            continue

        for waba in wabas:
            _titulo(f"WABA {waba.get('name')} (id {waba.get('id')})")
            try:
                numeros = _requisitar(
                    "GET", f"{waba['id']}/phone_numbers",
                    {"fields": "id,display_phone_number,verified_name,name_status"},
                ).get("data", [])
                for numero in numeros:
                    print(f"  {numero.get('display_phone_number')} — "
                          f"{numero.get('verified_name')} "
                          f"[{numero.get('name_status')}]")
                    print(f"     PHONE_NUMBER_ID={numero.get('id')}")
            except ErroDaMeta as erro:
                print(erro)

    _titulo("Links diretos")
    if app_id:
        base = f"https://developers.facebook.com/apps/{app_id}"
        print(f"  APP_SECRET        {base}/settings/basic/")
        print(f"  Webhook/WhatsApp  {base}/whatsapp-business/wa-settings/")
    else:
        print("  (sem app_id — abra https://developers.facebook.com/apps)")
    for negocio in negocios:
        bid = negocio["id"]
        print(f"  Info da empresa   https://business.facebook.com/settings/info"
              f"?business_id={bid}")
        print(f"  WhatsApp Manager  https://business.facebook.com/wa/manage"
              f"/phone-numbers/?business_id={bid}")
    return 0


PERFIL = "whatsapp_business_profile"
CAMPOS_DO_PERFIL = "about,address,description,email,vertical,websites"


def cmd_perfil(args: argparse.Namespace) -> int:
    """Mostra o perfil comercial — o cartão que a mãe vê ao abrir o contato."""
    numero = _exigir("PHONE_NUMBER_ID")
    try:
        dados = _requisitar("GET", f"{numero}/{PERFIL}", {"fields": CAMPOS_DO_PERFIL})
    except ErroDaMeta as erro:
        print(erro)
        return 1

    if args.json:
        print(json.dumps(dados, indent=2, ensure_ascii=False))
        return 0

    perfil = (dados.get("data") or [{}])[0]
    _titulo("Perfil comercial do número")
    for rotulo, chave in [
        ("descrição", "description"), ("sobre", "about"), ("e-mail", "email"),
        ("categoria", "vertical"), ("sites", "websites"),
    ]:
        valor = perfil.get(chave) or "—"
        if isinstance(valor, list):
            valor = ", ".join(valor) or "—"
        print(f"  {rotulo:<10} {valor}")
    return 0


def cmd_perfil_site(args: argparse.Namespace) -> int:
    """Grava o site do perfil comercial.

    Não substitui o campo *Site* do portfólio — são coisas diferentes. Mas é
    um sinal público a mais ligando o número à página do produto, e este dá
    para gravar por API.
    """
    numero = _exigir("PHONE_NUMBER_ID")
    if not args.url.startswith("https://"):
        sys.exit("A Meta só aceita site com https.")

    campos = {"messaging_product": "whatsapp", "websites": json.dumps([args.url])}
    if args.descricao:
        campos["description"] = args.descricao
    if args.email:
        campos["email"] = args.email

    try:
        resposta = _requisitar("POST", f"{numero}/{PERFIL}", campos)
    except ErroDaMeta as erro:
        print(erro)
        return 1

    print(json.dumps(resposta, indent=2, ensure_ascii=False))
    print("\nConfira com: python3 scripts/meta_admin.py perfil")
    return 0


def cmd_dominios(args: argparse.Namespace) -> int:
    negocio = _exigir("BUSINESS_ID")
    try:
        dados = _requisitar("GET", f"{negocio}/owned_domains", {
            "fields": "id,domain_name,is_verified,verification_code",
        })
    except ErroDaMeta as erro:
        print(erro)
        return 1

    if args.json:
        print(json.dumps(dados, indent=2, ensure_ascii=False))
        return 0

    dominios = dados.get("data", [])
    if not dominios:
        print("Nenhum domínio no portfólio.")
        return 0

    _titulo("Domínios do portfólio")
    for dominio in dominios:
        marca = "✅" if dominio.get("is_verified") else "⏳"
        print(f"  {marca} {dominio.get('domain_name')}  (id {dominio.get('id')})")
        if not dominio.get("is_verified") and dominio.get("verification_code"):
            print(f"     TXT: {dominio['verification_code']}")
    return 0


def cmd_dominio_add(args: argparse.Namespace) -> int:
    negocio = _exigir("BUSINESS_ID")
    try:
        resposta = _requisitar("POST", f"{negocio}/owned_domains", {
            "domain_name": args.dominio,
        })
    except ErroDaMeta as erro:
        print(erro)
        print("\nSe disser que o domínio já pertence a outro portfólio: um domínio")
        print("só pode ser verificado em um lugar. Nesse caso o caminho é o campo")
        print("*Site* do portfólio, não a verificação de domínio.")
        return 1

    print(json.dumps(resposta, indent=2, ensure_ascii=False))
    print("\nAgora peça o registro TXT com:")
    print("  python3 scripts/meta_admin.py dominios")
    print("Publique no DNS de", args.dominio, "e aguarde a propagação.")
    return 0


def main() -> int:
    analisador = argparse.ArgumentParser(
        description="Administra a conta do WhatsApp pela Graph API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    analisador.add_argument("--json", action="store_true",
                            help="imprime a resposta crua da Meta")
    sub = analisador.add_subparsers(dest="comando", required=True)

    sub.add_parser("descobrir",
                   help="descobre IDs e imprime o link direto de cada tela")

    sub.add_parser("status", help="mostra o estado do número e do nome")

    p_nome = sub.add_parser("nome", help="envia um nome de exibição para análise")
    p_nome.add_argument("nome")
    p_nome.add_argument("--confirmar", action="store_true",
                        help="envia de verdade (sem isso, só simula)")

    sub.add_parser("perfil", help="mostra o perfil comercial do número")

    p_site = sub.add_parser("perfil-site", help="grava o site do perfil comercial")
    p_site.add_argument("url")
    p_site.add_argument("--descricao", help="descrição curta do serviço")
    p_site.add_argument("--email", help="e-mail de contato público")

    sub.add_parser("dominios", help="lista os domínios do portfólio")

    p_add = sub.add_parser("dominio-add", help="acrescenta um domínio ao portfólio")
    p_add.add_argument("dominio")

    args = analisador.parse_args()

    # Antes de qualquer cabeçalho na tela: sem token nada funciona, e o erro
    # tem que ser a primeira coisa que aparece.
    if not os.environ.get("META_TOKEN", "").strip():
        sys.exit("Falta META_TOKEN no ambiente. Veja o cabeçalho deste arquivo.")

    comandos = {
        "descobrir": cmd_descobrir,
        "status": cmd_status,
        "nome": cmd_nome,
        "perfil": cmd_perfil,
        "perfil-site": cmd_perfil_site,
        "dominios": cmd_dominios,
        "dominio-add": cmd_dominio_add,
    }
    return comandos[args.comando](args)


if __name__ == "__main__":
    sys.exit(main())
