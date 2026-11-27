import os

# Importar main abre o banco do piloto. Em memória, os testes não criam
# data/lactai.db no repositório nem herdam contadores de uma rodada anterior.
os.environ.setdefault("DB_PATH", ":memory:")
# Teste não baixa o calendário do gov.br.
os.environ.setdefault("CALENDARIO_SUS_ATIVO", "false")


import pytest  # noqa: E402


@pytest.fixture
def banco():
    """Postgres de teste, zerado e migrado a cada teste.

    Usa TEST_DATABASE_URL (o CI sobe um Postgres; na sua máquina, aponte para
    um banco descartável). Sem ela, os testes de banco são pulados.
    """
    dsn = os.environ.get("TEST_DATABASE_URL", "").strip()
    if not dsn:
        pytest.skip("TEST_DATABASE_URL não definida")

    import psycopg

    from app.migrar import aplicar

    with psycopg.connect(dsn, autocommit=True) as con:
        con.execute("DROP SCHEMA public CASCADE")
        con.execute("CREATE SCHEMA public")
    aplicar(dsn)
    return dsn
