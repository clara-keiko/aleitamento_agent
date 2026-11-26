"""Aplica as migrações de migrations/ em ordem, cada uma uma vez só.

Roda no deploy, antes do código novo subir (preDeployCommand no render.yaml):
se uma migração falhar, o Render interrompe o deploy e a versão anterior
continua no ar, com o banco como estava — cada migração roda numa transação.

    python -m app.migrar
"""

import os
import sys
from pathlib import Path

import psycopg

PASTA = Path(__file__).resolve().parent.parent / "migrations"
# Dois deploys simultâneos não podem aplicar a mesma migração duas vezes.
_TRAVA = 727_274


def aplicar(dsn: str, pasta: Path = PASTA) -> list[str]:
    aplicadas: list[str] = []
    with psycopg.connect(dsn, autocommit=True) as con:
        con.execute("SELECT pg_advisory_lock(%s)", (_TRAVA,))
        try:
            con.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " versao text PRIMARY KEY,"
                " aplicada_em timestamptz NOT NULL DEFAULT now())"
            )
            feitas = {linha[0] for linha in con.execute("SELECT versao FROM schema_migrations")}
            for arquivo in sorted(pasta.glob("*.sql")):
                if arquivo.stem in feitas:
                    continue
                with con.transaction():
                    con.execute(arquivo.read_text(encoding="utf-8"))
                    con.execute(
                        "INSERT INTO schema_migrations (versao) VALUES (%s)", (arquivo.stem,)
                    )
                aplicadas.append(arquivo.stem)
        finally:
            con.execute("SELECT pg_advisory_unlock(%s)", (_TRAVA,))
    return aplicadas


def main() -> int:
    dsn = os.getenv("DATABASE_URL", "").strip()
    if not dsn:
        # Sem banco configurado, o agente funciona como antes (memória curta).
        print("DATABASE_URL vazia; nenhuma migração aplicada.")
        return 0
    aplicadas = aplicar(dsn)
    print("migrações aplicadas:", ", ".join(aplicadas) if aplicadas else "nenhuma (já em dia)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
