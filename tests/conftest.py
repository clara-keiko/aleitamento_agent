import os

# Importar main abre o banco do piloto. Em memória, os testes não criam
# data/lactai.db no repositório nem herdam contadores de uma rodada anterior.
os.environ.setdefault("DB_PATH", ":memory:")
