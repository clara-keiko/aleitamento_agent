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
# v1: texto sem link da política, e GUARDAR aceitava sem mostrar a pergunta.
# v2: link da política no texto; aceite só respondendo à pergunta inteira.
VERSAO_MEMORIA = "memoria-v2"

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
    # Perfil e diário do bebê (questionário inicial)
    # ------------------------------------------------------------------
    def _atualizar_cuidador(self, numero: str, campos: str, valores: tuple) -> None:
        with self._conectar() as con:
            cuidador = self._garantir_cuidador(con, numero)
            con.execute(f"UPDATE cuidadores SET {campos} WHERE id = %s", (*valores, cuidador))  # noqa: S608

    def salvar_relacao(self, numero: str, relacao: str) -> None:
        self._atualizar_cuidador(numero, "relacao = %s", (relacao,))

    def salvar_gestacao(self, numero: str, data_provavel_parto: dt.date) -> None:
        self._atualizar_cuidador(
            numero, "gestante = true, data_provavel_parto = %s", (data_provavel_parto,)
        )

    def salvar_ano_nascimento_mae(self, numero: str, ano: int) -> None:
        self._atualizar_cuidador(numero, "ano_nascimento_mae = %s", (ano,))

    def salvar_bebe(self, numero: str, data_nascimento: dt.date) -> str:
        """O bebê com essa data de nascimento; cria se ainda não existe."""
        with self._conectar() as con, con.transaction():
            cuidador = self._garantir_cuidador(con, numero)
            # O bebê nasceu: a gestação registrada antes terminou.
            con.execute(
                "UPDATE cuidadores SET gestante = false WHERE id = %s", (cuidador,)
            )
            existente = con.execute(
                "SELECT id FROM bebes WHERE cuidador_id = %s AND data_nascimento = %s",
                (cuidador, data_nascimento),
            ).fetchone()
            if existente:
                return str(existente[0])
            return str(con.execute(
                "INSERT INTO bebes (cuidador_id, data_nascimento) VALUES (%s, %s) RETURNING id",
                (cuidador, data_nascimento),
            ).fetchone()[0])

    def salvar_sexo(self, bebe_id: str, sexo: str) -> None:
        with self._conectar() as con:
            con.execute("UPDATE bebes SET sexo = %s WHERE id = %s", (sexo, bebe_id))

    def registrar_do_bebe(
        self, bebe_id: str, tipo: str, referente_a: dt.date,
        numero_valor: float | None = None, texto_valor: str | None = None,
        origem: str = "pergunta_do_lactai",
    ) -> None:
        """Uma linha nova no diário; nunca sobrescreve as anteriores."""
        with self._conectar() as con:
            con.execute(
                "INSERT INTO registros_do_bebe"
                " (bebe_id, tipo, valor_numero, valor_texto, referente_a, origem)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (bebe_id, tipo, numero_valor, texto_valor, referente_a, origem),
            )

    def registrar_vacinas(self, bebe_id: str, vacinas: list[tuple[str, dt.date]]) -> tuple[int, int]:
        """(novas, já anotadas). A mesma foto mandada duas vezes não duplica."""
        novas = 0
        with self._conectar() as con, con.transaction():
            for rotulo, data in vacinas:
                novas += con.execute(
                    "INSERT INTO registros_do_bebe"
                    " (bebe_id, tipo, valor_texto, referente_a, origem)"
                    " VALUES (%s, 'vacina_tomada', %s, %s, 'foto_caderneta')"
                    " ON CONFLICT (bebe_id, valor_texto, referente_a)"
                    " WHERE tipo = 'vacina_tomada' DO NOTHING",
                    (bebe_id, rotulo, data),
                ).rowcount
        return novas, len(vacinas) - novas

    def perfil(self, numero: str) -> dict | None:
        """Situação atual: cuidador(a) e bebês com o último valor de cada tipo."""
        with self._conectar() as con:
            conta = con.execute(
                "SELECT id, relacao, ano_nascimento_mae, gestante, data_provavel_parto"
                " FROM cuidadores WHERE telefone_hash = %s",
                (_chave(numero),),
            ).fetchone()
            if conta is None:
                return None
            bebes = con.execute(
                "SELECT id, data_nascimento, sexo FROM bebes WHERE cuidador_id = %s"
                " ORDER BY data_nascimento DESC",
                (conta[0],),
            ).fetchall()
            vacinas = con.execute(
                "SELECT r.bebe_id, r.valor_texto, r.referente_a FROM registros_do_bebe r"
                " JOIN bebes b ON b.id = r.bebe_id"
                " WHERE b.cuidador_id = %s AND r.tipo = 'vacina_tomada'"
                " AND NOT EXISTS (SELECT 1 FROM registros_do_bebe c WHERE c.substitui_id = r.id)"
                " ORDER BY r.referente_a",
                (conta[0],),
            ).fetchall()
            situacao = con.execute(
                "SELECT s.bebe_id, s.tipo, s.valor_numero, s.valor_texto, s.referente_a"
                " FROM situacao_atual_do_bebe s JOIN bebes b ON b.id = s.bebe_id"
                " WHERE b.cuidador_id = %s",
                (conta[0],),
            ).fetchall()
        atuais: dict = {}
        por_bebe: dict = {}
        for bebe_id, rotulo, data in vacinas:
            por_bebe.setdefault(bebe_id, []).append((rotulo, data))
        for bebe_id, tipo, numero_valor, texto_valor, referente_a in situacao:
            atuais.setdefault(bebe_id, {})[tipo] = (
                float(numero_valor) if numero_valor is not None else texto_valor, referente_a
            )
        return {
            "relacao": conta[1],
            "ano_nascimento_mae": conta[2],
            "gestante": conta[3],
            "data_provavel_parto": conta[4],
            "bebes": [
                {
                    "id": str(bebe_id), "nascimento": nascimento, "sexo": sexo,
                    "vacinas": por_bebe.get(bebe_id, []), **atuais.get(bebe_id, {}),
                }
                for bebe_id, nascimento, sexo in bebes
            ],
        }

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
