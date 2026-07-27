#!/usr/bin/env python3
"""Indexa docs/ numa vector store da OpenAI para o file_search.

A versão anterior criava uma vector store **nova a cada execução**. Rodar o
script duas vezes deixava duas bases pagas no ar, e a segunda tinha o mesmo
conteúdo da primeira. Também não avisava quando um arquivo falhava no
processamento — a base ficava incompleta em silêncio, e o agente passava a
responder "não encontrei isso no material" sem ninguém entender por quê.

Agora:

    python ingest_openai_kb.py
        Reusa a store apontada por VECTOR_STORE_ID (ou acha pelo nome) e
        envia só os arquivos que ainda não estão lá.

    python ingest_openai_kb.py --recreate
        Descarta e reindexa tudo do zero.

    python ingest_openai_kb.py --dry-run
        Mostra o que faria, sem gastar nada.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

DOCS_DIR = Path(__file__).parent / "docs"
STORE_NAME = "puericultura_amamentacao"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def e_documentacao_do_projeto(caminho: Path) -> bool:
    """True para docs do projeto, que não podem entrar na base clínica.

    A convenção é o nome em CAIXA ALTA (OPERACAO.md, GO_LIVE.md). O material
    clínico usa nome normal (AppAM_*.docx, vacinas_*.md). Uma lista fixa de
    nomes já falhou uma vez: bastou criar um doc novo para ele ser indexado
    como se fosse conteúdo de saúde e passar a competir na recuperação.
    """
    if caminho.suffix.lower() != ".md":
        return False
    return caminho.stem.upper() == caminho.stem


def listar_arquivos() -> list[Path]:
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {DOCS_DIR}")

    arquivos = [
        caminho
        for caminho in DOCS_DIR.rglob("*")
        if caminho.is_file()
        and caminho.suffix.lower() in ALLOWED_EXTENSIONS
        and not e_documentacao_do_projeto(caminho)
        and not caminho.name.startswith(".")
    ]

    if not arquivos:
        raise ValueError(
            f"Nenhum arquivo compatível em {DOCS_DIR}. "
            f"Extensões aceitas: {sorted(ALLOWED_EXTENSIONS)}"
        )

    return sorted(arquivos)


def encontrar_store(client: OpenAI, store_id: str) -> str | None:
    """Devolve o id de uma store utilizável, ou None."""
    if store_id:
        try:
            store = client.vector_stores.retrieve(vector_store_id=store_id)
            print(f"Usando a store existente: {store.id} ({store.name})")
            return store.id
        except OpenAIError:
            print(f"VECTOR_STORE_ID={store_id} não existe mais; procurando pelo nome…")

    try:
        for store in client.vector_stores.list(limit=100).data:
            if store.name == STORE_NAME:
                print(f"Encontrei pelo nome: {store.id}")
                return store.id
    except OpenAIError as exc:
        print(f"Não consegui listar as stores: {exc}")

    return None


def nomes_ja_indexados(client: OpenAI, store_id: str) -> set[str]:
    """Nomes de arquivo já presentes na store, para não reenviar."""
    nomes: set[str] = set()
    try:
        for item in client.vector_stores.files.list(vector_store_id=store_id, limit=100):
            try:
                arquivo = client.files.retrieve(item.id)
                nomes.add(arquivo.filename)
            except OpenAIError:
                continue
    except OpenAIError as exc:
        print(f"Aviso: não consegui listar os arquivos da store ({exc}); "
              "vou tratar todos como novos.")
    return nomes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recreate", action="store_true", help="reindexa tudo do zero")
    parser.add_argument("--dry-run", action="store_true", help="não envia nada")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERRO: OPENAI_API_KEY não configurada.")
        return 2

    arquivos = listar_arquivos()
    print(f"\n{len(arquivos)} arquivo(s) em docs/:")
    for caminho in arquivos:
        print(f"  - {caminho.name} ({caminho.stat().st_size / 1024:.0f} KB)")

    if args.dry_run:
        print("\n--dry-run: nada foi enviado.")
        return 0

    client = OpenAI(api_key=api_key, timeout=600)

    store_id = None if args.recreate else encontrar_store(
        client, os.getenv("VECTOR_STORE_ID", "").strip()
    )

    if store_id is None:
        store = client.vector_stores.create(name=STORE_NAME)
        store_id = store.id
        print(f"\nStore criada: {store_id}")
        pendentes = arquivos
    else:
        indexados = nomes_ja_indexados(client, store_id)
        pendentes = [c for c in arquivos if c.name not in indexados]
        print(f"\n{len(indexados)} já indexado(s), {len(pendentes)} a enviar.")

    if not pendentes:
        print("\nNada a fazer. A base já está atualizada.")
        print(f"\nVECTOR_STORE_ID={store_id}")
        return 0

    print(f"\nEnviando {len(pendentes)} arquivo(s)… (pode demorar em PDFs grandes)")

    abertos = []
    try:
        abertos = [caminho.open("rb") for caminho in pendentes]
        lote = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=store_id, files=abertos
        )
    except OpenAIError as exc:
        print(f"\nERRO no envio: {exc}")
        return 1
    finally:
        for arquivo in abertos:
            arquivo.close()

    contagem = lote.file_counts
    print(f"\nStatus do lote: {lote.status}")
    print(f"  concluídos: {contagem.completed}")
    print(f"  falharam:   {contagem.failed}")
    print(f"  cancelados: {contagem.cancelled}")

    if contagem.failed:
        # Falha silenciosa aqui vira "não encontrei isso no material" para a
        # mãe, sem ninguém saber que faltou conteúdo. Falhar visivelmente.
        print("\n⚠️  Arquivos falharam no processamento. A base está INCOMPLETA.")
        try:
            for item in client.vector_stores.files.list(vector_store_id=store_id, limit=100):
                if item.status == "failed":
                    erro = getattr(item, "last_error", None)
                    print(f"    ✗ {item.id}: {getattr(erro, 'message', 'motivo desconhecido')}")
        except OpenAIError:
            pass
        print(f"\nVECTOR_STORE_ID={store_id}")
        return 1

    print("\n✅ Base indexada.")
    print(f"\nColoque no seu .env e no painel do Render:\n\nVECTOR_STORE_ID={store_id}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
