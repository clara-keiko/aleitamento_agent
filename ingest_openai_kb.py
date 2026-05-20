import os
import time
from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DOCS_DIR = Path("docs")
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

def list_files_from_docs():
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {DOCS_DIR}")

    files = [
        path for path in DOCS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    ]

    if not files:
        raise ValueError(
            f"Nenhum arquivo compatível encontrado em {DOCS_DIR}. "
            f"Extensões aceitas: {sorted(ALLOWED_EXTENSIONS)}"
        )

    return sorted(files)

def main():
    file_paths = list_files_from_docs()

    print("Arquivos encontrados:")
    for path in file_paths:
        print(f" - {path}")

    vector_store = client.vector_stores.create(
        name="puericultura_amamentacao_mvp"
    )
    print(f"\nVECTOR_STORE_ID: {vector_store.id}\n")

    attached_file_ids = []

    for path in file_paths:
        print(f"Enviando arquivo: {path}")
        with open(path, "rb") as f:
            uploaded = client.files.create(
                file=f,
                purpose="assistants"
            )

        attached = client.vector_stores.files.create(
            vector_store_id=vector_store.id,
            file_id=uploaded.id
        )

        attached_file_ids.append(attached.id)
        print(f"Arquivo enviado e anexado: {path} -> {attached.id}")

    pending = set(attached_file_ids)

    while pending:
        completed_now = []

        for file_id in pending:
            status_obj = client.vector_stores.files.retrieve(
                vector_store_id=vector_store.id,
                file_id=file_id
            )

            print(f"{file_id}: {status_obj.status}")

            if status_obj.status in {"completed", "failed", "cancelled"}:
                completed_now.append(file_id)

        for file_id in completed_now:
            pending.remove(file_id)

        if pending:
            time.sleep(5)

    print("\nTodos os arquivos foram processados.")
    print(f"Use este VECTOR_STORE_ID no Render: {vector_store.id}")

if __name__ == "__main__":
    main()