"""Fase de testes: quem pode conversar e quanto pode gastar.

Três perguntas, respondidas aqui e consultadas pelo pipeline:

1. **Este número foi convidado?** A pessoa recebe um link que abre o WhatsApp
   com uma frase pronta contendo um código. Ao enviar, o próprio WhatsApp diz
   de qual número veio — então autorizar é só anotar o remetente. Ninguém
   digita número, ninguém confirma nada à mão. Cada código tem um teto de
   usos: se o link circular além do previsto, as vagas acabam sozinhas.

2. **Esta pessoa já perguntou demais hoje?** Contagem por dia, no fuso de
   Brasília, só das perguntas que chamam o modelo.

3. **O piloto já gastou o que podia neste mês?** Soma do custo real de cada
   resposta, calculado com os tokens que a API informa.

O telefone nunca é gravado: guardamos o HMAC dele. Para reconhecer quem
voltou, basta calcular o mesmo HMAC de quem escreve.

SQLite porque o volume é pequeno (dezenas de pessoas), porque não exige
serviço a mais nem dependência nova, e porque os testes rodam sem nada
instalado. Em produção o arquivo mora num disco persistente; num caminho
efêmero, um deploy apagaria convites e contadores.
"""

import datetime as dt
import re
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

from app.logging_utils import get_logger, hash_estavel

log = get_logger(__name__)

AUTORIZADO = "autorizado"
JA_AUTORIZADO = "ja_autorizado"
ESGOTADO = "esgotado"
CODIGO_INVALIDO = "codigo_invalido"

# O Brasil não tem horário de verão desde 2019. Offset fixo dispensa a base de
# fusos, que nem toda imagem slim traz.
BRASILIA = dt.timezone(dt.timedelta(hours=-3))

_ESQUEMA = """
CREATE TABLE IF NOT EXISTS convites (
    codigo  TEXT PRIMARY KEY,
    limite  INTEGER NOT NULL,      -- 0 = sem teto
    usados  INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS testadores (
    numero_hash   TEXT PRIMARY KEY,
    codigo        TEXT NOT NULL,
    autorizado_em TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS uso_diario (
    dia         TEXT NOT NULL,
    numero_hash TEXT NOT NULL,
    perguntas   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (dia, numero_hash)
);
CREATE TABLE IF NOT EXISTS gasto_mensal (
    mes TEXT PRIMARY KEY,
    usd REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS recusas (
    numero_hash TEXT PRIMARY KEY,
    avisado_em  TEXT NOT NULL
);
"""


def numero_canonico(numero: str) -> str:
    """Uma forma só para o mesmo telefone.

    O WhatsApp entrega muitos celulares brasileiros sem o 9 depois do DDD.
    Sem normalizar, a mesma pessoa viraria duas e gastaria duas vagas. Só
    celular ganha o 9: número de 8 dígitos começando em 6 a 9; fixo, que
    começa de 2 a 5, fica como está.
    """
    digitos = "".join(c for c in numero or "" if c.isdigit())
    if digitos.startswith("55") and len(digitos) == 12 and digitos[4] in "6789":
        return digitos[:4] + "9" + digitos[4:]
    return digitos


def ler_convites(bruto: str) -> dict[str, int]:
    """'PILOTO:30, EQUIPE:5' -> {'PILOTO': 30, 'EQUIPE': 5}. Sem número = sem teto."""
    convites: dict[str, int] = {}
    for item in (bruto or "").split(","):
        codigo, _, limite = item.strip().partition(":")
        codigo = codigo.strip().upper()
        if not codigo:
            continue
        try:
            convites[codigo] = max(int(limite.strip() or 0), 0)
        except ValueError:
            log.warning("convite %r com limite inválido %r; ignorado", codigo, limite)
    return convites


def encontrar_codigo(texto: str, codigos: list[str]) -> str | None:
    """Código de convite presente na mensagem, como palavra inteira."""
    if not texto or not codigos:
        return None
    palavras = set(re.findall(r"[A-Z0-9_-]+", texto.upper()))
    for codigo in codigos:
        if codigo in palavras:
            return codigo
    return None


class RegistroDoPiloto:
    def __init__(
        self,
        caminho: str,
        convites: dict[str, int],
        agora: Callable[[], dt.datetime] | None = None,
    ):
        self.caminho = caminho
        self._agora = agora or (lambda: dt.datetime.now(BRASILIA))
        self._lock = threading.Lock()

        if caminho != ":memory:":
            Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        # Uma conexão compartilhada e um lock: o volume é baixo, e o pipeline
        # roda em threads do servidor.
        self._con = sqlite3.connect(caminho, check_same_thread=False, isolation_level=None)
        with self._lock:
            self._con.executescript(_ESQUEMA)
        self._sincronizar_convites(convites)

    @property
    def codigos(self) -> list[str]:
        with self._lock:
            return [linha[0] for linha in self._con.execute("SELECT codigo FROM convites")]

    # ------------------------------------------------------------------
    # Convites
    # ------------------------------------------------------------------
    def _sincronizar_convites(self, convites: dict[str, int]) -> None:
        """Os códigos vêm da configuração; os usos ficam no banco.

        Mudar o teto na configuração vale na hora sem zerar quem já entrou.
        Código removido da configuração deixa de aceitar gente nova, mas quem
        entrou por ele continua autorizado.
        """
        with self._lock:
            for codigo, limite in convites.items():
                self._con.execute(
                    "INSERT INTO convites (codigo, limite) VALUES (?, ?) "
                    "ON CONFLICT(codigo) DO UPDATE SET limite = excluded.limite",
                    (codigo, limite),
                )
            ativos = list(convites)
            if ativos:
                marcas = ",".join("?" * len(ativos))
                self._con.execute(f"DELETE FROM convites WHERE codigo NOT IN ({marcas})", ativos)  # noqa: S608
            else:
                self._con.execute("DELETE FROM convites")

    def autorizado(self, numero: str) -> bool:
        chave = hash_estavel(numero_canonico(numero))
        with self._lock:
            linha = self._con.execute(
                "SELECT 1 FROM testadores WHERE numero_hash = ?", (chave,)
            ).fetchone()
        return linha is not None

    def usar_convite(self, numero: str, codigo: str) -> str:
        chave = hash_estavel(numero_canonico(numero))
        agora = self._agora().isoformat(timespec="seconds")
        with self._lock:
            if self._con.execute(
                "SELECT 1 FROM testadores WHERE numero_hash = ?", (chave,)
            ).fetchone():
                return JA_AUTORIZADO

            convite = self._con.execute(
                "SELECT limite, usados FROM convites WHERE codigo = ?", (codigo,)
            ).fetchone()
            if convite is None:
                return CODIGO_INVALIDO
            limite, usados = convite
            if limite and usados >= limite:
                return ESGOTADO

            self._con.execute("BEGIN IMMEDIATE")
            try:
                self._con.execute(
                    "INSERT INTO testadores (numero_hash, codigo, autorizado_em) VALUES (?, ?, ?)",
                    (chave, codigo, agora),
                )
                self._con.execute(
                    "UPDATE convites SET usados = usados + 1 WHERE codigo = ?", (codigo,)
                )
                self._con.execute("COMMIT")
            except sqlite3.Error:
                self._con.execute("ROLLBACK")
                raise
        log.info("convite %s usado (%d de %s)", codigo, usados + 1, limite or "sem teto")
        return AUTORIZADO

    def deve_avisar_recusa(self, numero: str, intervalo_horas: int = 24) -> bool:
        """Quem não foi convidado recebe o aviso no máximo uma vez por dia.

        Responder a cada mensagem de um estranho seria insistente, e é o
        padrão de envio que mais chama atenção num número não oficial.
        """
        chave = hash_estavel(numero_canonico(numero))
        agora = self._agora()
        with self._lock:
            linha = self._con.execute(
                "SELECT avisado_em FROM recusas WHERE numero_hash = ?", (chave,)
            ).fetchone()
            if linha and agora - dt.datetime.fromisoformat(linha[0]) < dt.timedelta(
                hours=intervalo_horas
            ):
                return False
            self._con.execute(
                "INSERT INTO recusas (numero_hash, avisado_em) VALUES (?, ?) "
                "ON CONFLICT(numero_hash) DO UPDATE SET avisado_em = excluded.avisado_em",
                (chave, agora.isoformat(timespec="seconds")),
            )
        return True

    # ------------------------------------------------------------------
    # Uso e gasto
    # ------------------------------------------------------------------
    def _dia(self) -> str:
        return self._agora().astimezone(BRASILIA).date().isoformat()

    def _mes(self) -> str:
        return self._dia()[:7]

    def perguntas_hoje(self, numero: str) -> int:
        chave = hash_estavel(numero_canonico(numero))
        with self._lock:
            linha = self._con.execute(
                "SELECT perguntas FROM uso_diario WHERE dia = ? AND numero_hash = ?",
                (self._dia(), chave),
            ).fetchone()
        return linha[0] if linha else 0

    def registrar_pergunta(self, numero: str | None, custo_usd: float) -> None:
        """Soma uma pergunta ao dia da pessoa e o custo ao mês do piloto.

        `numero` vazio (protótipo web) conta só no gasto: lá o remetente é uma
        sessão do navegador, não uma pessoa identificável.
        """
        with self._lock:
            if numero:
                self._con.execute(
                    "INSERT INTO uso_diario (dia, numero_hash, perguntas) VALUES (?, ?, 1) "
                    "ON CONFLICT(dia, numero_hash) DO UPDATE SET perguntas = perguntas + 1",
                    (self._dia(), hash_estavel(numero_canonico(numero))),
                )
            self._con.execute(
                "INSERT INTO gasto_mensal (mes, usd) VALUES (?, ?) "
                "ON CONFLICT(mes) DO UPDATE SET usd = usd + excluded.usd",
                (self._mes(), max(custo_usd, 0.0)),
            )

    def gasto_do_mes(self) -> float:
        with self._lock:
            linha = self._con.execute(
                "SELECT usd FROM gasto_mensal WHERE mes = ?", (self._mes(),)
            ).fetchone()
        return linha[0] if linha else 0.0

    # ------------------------------------------------------------------
    # Painel
    # ------------------------------------------------------------------
    def resumo(self) -> dict:
        dia = self._dia()
        with self._lock:
            convites = [
                {"codigo": c, "limite": lim, "usados": usados}
                for c, lim, usados in self._con.execute(
                    "SELECT codigo, limite, usados FROM convites ORDER BY codigo"
                )
            ]
            testadores = self._con.execute("SELECT COUNT(*) FROM testadores").fetchone()[0]
            ativos_hoje, perguntas_hoje = self._con.execute(
                "SELECT COUNT(*), COALESCE(SUM(perguntas), 0) FROM uso_diario WHERE dia = ?",
                (dia,),
            ).fetchone()
        return {
            "dia": dia,
            "convites": convites,
            "testadores": testadores,
            "ativos_hoje": ativos_hoje,
            "perguntas_hoje": perguntas_hoje,
            "gasto_do_mes_usd": round(self.gasto_do_mes(), 4),
        }


def custo_da_resposta(
    tokens_entrada: int, tokens_saida: int, preco_entrada_1m: float, preco_saida_1m: float
) -> float:
    return (tokens_entrada * preco_entrada_1m + tokens_saida * preco_saida_1m) / 1_000_000
