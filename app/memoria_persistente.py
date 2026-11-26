"""Memória do cuidador(a), guardada no Postgres. Ver docs/MEMORIA_E_LEMBRETES.md.

Só existe para quem disse "sim" à pergunta de memória; para o resto, o
pipeline segue com a memória curta de sempre (app/memory.py).

O telefone nunca é gravado em claro: a conta é achada pelo HMAC do número
canônico, o mesmo usado no piloto. O conteúdo das mensagens vai cifrado
(app/cripto.py).
"""

import datetime as dt
from collections.abc import Callable

import psycopg

from app.cripto import Cofre
from app.logging_utils import get_logger, hash_estavel
from app.piloto import numero_canonico

log = get_logger(__name__)

MEMORIA = "memoria"
VERSAO_MEMORIA = "memoria-v1"

_PAPEL = {"cuidador": "user", "lactai": "assistant"}


def _chave(numero: str) -> str:
    return hash_estavel(numero_canonico(numero))


class MemoriaPersistente:
    def __init__(
        self,
        dsn: str,
        cofre: Cofre,
        agora: Callable[[], dt.datetime] | None = None,
    ):
        self.dsn = dsn
        self.cofre = cofre
        self._agora = agora or (lambda: dt.datetime.now(dt.UTC))

    def _conectar(self) -> psycopg.Connection:
        # Uma conexão por operação: o volume é de poucas mensagens por minuto,
        # e assim uma queda do banco não deixa conexão quebrada presa no pool.
        return psycopg.connect(self.dsn, autocommit=True, connect_timeout=5)

    # ------------------------------------------------------------------
    # Consentimento
    # ------------------------------------------------------------------
    def decisao(self, numero: str, finalidade: str = MEMORIA) -> bool | None:
        """True/False pela decisão vigente; None se nunca foi perguntado."""
        with self._conectar() as con:
            linha = con.execute(
                "SELECT aceito, revogado_em FROM consentimentos"
                " WHERE telefone_hash = %s AND finalidade = %s"
                " ORDER BY decidido_em DESC, id DESC LIMIT 1",
                (_chave(numero), finalidade),
            ).fetchone()
        if linha is None:
            return None
        aceito, revogado_em = linha
        return bool(aceito) and revogado_em is None

    def decidir(
        self, numero: str, aceito: bool, evidencia: str,
        finalidade: str = MEMORIA, versao: str = VERSAO_MEMORIA,
    ) -> None:
        with self._conectar() as con:
            con.execute(
                "INSERT INTO consentimentos"
                " (telefone_hash, finalidade, aceito, versao_texto, decidido_em, evidencia)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (_chave(numero), finalidade, aceito, versao, self._agora(), evidencia),
            )

    # ------------------------------------------------------------------
    # Conversa
    # ------------------------------------------------------------------
    def _garantir_cuidador(self, con: psycopg.Connection, numero: str) -> str:
        """Cria a conta ou a reativa; toda mensagem conta como uso."""
        return con.execute(
            "INSERT INTO cuidadores (telefone_hash, ultima_mensagem_em) VALUES (%s, %s)"
            " ON CONFLICT (telefone_hash) DO UPDATE SET"
            "  ultima_mensagem_em = excluded.ultima_mensagem_em,"
            "  status = 'ativa', inativa_desde = NULL, motivo_inativa = NULL"
            " RETURNING id",
            (_chave(numero), self._agora()),
        ).fetchone()[0]

    def tocar(self, numero: str) -> None:
        """Registra uso sem guardar conteúdo (cumprimento, comando)."""
        with self._conectar() as con:
            self._garantir_cuidador(con, numero)

    def registrar_troca(
        self, numero: str, pergunta: str, resposta: str,
        tipo: str = "texto", desfecho: str | None = None,
    ) -> None:
        agora = self._agora()
        with self._conectar() as con, con.transaction():
            cuidador = self._garantir_cuidador(con, numero)
            con.execute(
                "INSERT INTO mensagens (cuidador_id, autor, conteudo_cifrado, tipo, criada_em)"
                " VALUES (%s, 'cuidador', %s, %s, %s)",
                (cuidador, self.cofre.cifrar(pergunta), tipo, agora),
            )
            con.execute(
                "INSERT INTO mensagens"
                " (cuidador_id, autor, conteudo_cifrado, desfecho, criada_em)"
                " VALUES (%s, 'lactai', %s, %s, %s)",
                (cuidador, self.cofre.cifrar(resposta), desfecho, agora),
            )

    def historico(self, numero: str, maximo: int) -> list[dict]:
        """As últimas `maximo` mensagens, da mais antiga para a mais nova."""
        with self._conectar() as con:
            linhas = con.execute(
                "SELECT m.autor, m.conteudo_cifrado FROM mensagens m"
                " JOIN cuidadores c ON c.id = m.cuidador_id"
                " WHERE c.telefone_hash = %s ORDER BY m.id DESC LIMIT %s",
                (_chave(numero), maximo),
            ).fetchall()
        return [
            {"role": _PAPEL[autor], "content": self.cofre.decifrar(conteudo)}
            for autor, conteudo in reversed(linhas)
        ]

    # ------------------------------------------------------------------
    # Direitos da titular
    # ------------------------------------------------------------------
    def tem_dados(self, numero: str) -> bool:
        with self._conectar() as con:
            return con.execute(
                "SELECT 1 FROM cuidadores WHERE telefone_hash = %s", (_chave(numero),)
            ).fetchone() is not None

    def meus_dados(self, numero: str) -> dict:
        chave = _chave(numero)
        with self._conectar() as con:
            conta = con.execute(
                "SELECT c.criado_em,"
                " (SELECT count(*) FROM mensagens m WHERE m.cuidador_id = c.id),"
                " (SELECT count(*) FROM bebes b WHERE b.cuidador_id = c.id),"
                " (SELECT count(*) FROM registros_do_bebe r JOIN bebes b ON b.id = r.bebe_id"
                "   WHERE b.cuidador_id = c.id)"
                " FROM cuidadores c WHERE c.telefone_hash = %s",
                (chave,),
            ).fetchone()
            decisoes = con.execute(
                "SELECT DISTINCT ON (finalidade) finalidade, aceito, revogado_em, decidido_em"
                " FROM consentimentos WHERE telefone_hash = %s"
                " ORDER BY finalidade, decidido_em DESC, id DESC",
                (chave,),
            ).fetchall()
        return {
            "conta_desde": conta[0] if conta else None,
            "mensagens": conta[1] if conta else 0,
            "bebes": conta[2] if conta else 0,
            "registros_do_bebe": conta[3] if conta else 0,
            "consentimentos": {
                finalidade: {"aceito": bool(aceito) and revogado is None, "em": em}
                for finalidade, aceito, revogado, em in decisoes
            },
        }

    def apagar_tudo(self, numero: str) -> None:
        """Exclusão a pedido (art. 18, VI): conta, bebês, diário e conversa.

        Fica só a prova das decisões, sem conteúdo, com a revogação anotada.
        """
        chave = _chave(numero)
        agora = self._agora()
        with self._conectar() as con, con.transaction():
            con.execute("DELETE FROM cuidadores WHERE telefone_hash = %s", (chave,))
            con.execute(
                "UPDATE consentimentos SET revogado_em = %s"
                " WHERE telefone_hash = %s AND aceito AND revogado_em IS NULL",
                (agora, chave),
            )

    # ------------------------------------------------------------------
    # Retenção
    # ------------------------------------------------------------------
    def aplicar_retencao(self, meses_sem_uso: int, meses_inativa: int) -> tuple[int, int]:
        """Inativa quem sumiu e apaga quem está inativo há tempo demais.

        Inativar não apaga: quem volta a escrever reativa a conta com tudo.
        """
        agora = self._agora()
        with self._conectar() as con, con.transaction():
            inativadas = con.execute(
                "UPDATE cuidadores SET status = 'inativa', inativa_desde = %s,"
                " motivo_inativa = 'sem_uso'"
                " WHERE status = 'ativa' AND ultima_mensagem_em < %s - make_interval(months => %s)",
                (agora, agora, meses_sem_uso),
            ).rowcount
            apagadas = con.execute(
                "DELETE FROM cuidadores"
                " WHERE status = 'inativa' AND inativa_desde < %s - make_interval(months => %s)",
                (agora, meses_inativa),
            ).rowcount
        if inativadas or apagadas:
            log.info("retenção: %d conta(s) inativada(s), %d apagada(s)", inativadas, apagadas)
        return inativadas, apagadas
