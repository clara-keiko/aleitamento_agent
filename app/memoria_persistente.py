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

    def _familia(self, con, chave: str):
        """(família, hash do titular) de quem é titular ou membro; ou None."""
        linha = con.execute(
            "SELECT id, titular_hash FROM familias WHERE titular_hash = %s", (chave,)
        ).fetchone()
        if linha:
            return linha
        return con.execute(
            "SELECT f.id, f.titular_hash FROM membros_da_familia m"
            " JOIN familias f ON f.id = m.familia_id WHERE m.telefone_hash = %s",
            (chave,),
        ).fetchone()

    def _cuidador_dos_bebes(self, con, numero: str) -> str:
        """Os bebês moram na conta do titular; o membro da família vê e
        atualiza os dele. Fora de família, a própria conta."""
        familia = self._familia(con, _chave(numero))
        if familia and familia[1] != _chave(numero):
            linha = con.execute(
                "SELECT id FROM cuidadores WHERE telefone_hash = %s", (familia[1],)
            ).fetchone()
            if linha:
                return linha[0]
        return self._garantir_cuidador(con, numero)

    def salvar_bebe(self, numero: str, data_nascimento: dt.date) -> str:
        """O bebê com essa data de nascimento; cria se ainda não existe."""
        with self._conectar() as con, con.transaction():
            cuidador = self._cuidador_dos_bebes(con, numero)
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
            dono = self._cuidador_dos_bebes(con, numero)
            bebes = con.execute(
                "SELECT id, data_nascimento, sexo FROM bebes WHERE cuidador_id = %s"
                " ORDER BY data_nascimento DESC",
                (dono,),
            ).fetchall()
            vacinas = con.execute(
                "SELECT r.bebe_id, r.valor_texto, r.referente_a FROM registros_do_bebe r"
                " JOIN bebes b ON b.id = r.bebe_id"
                " WHERE b.cuidador_id = %s AND r.tipo = 'vacina_tomada'"
                " AND NOT EXISTS (SELECT 1 FROM registros_do_bebe c WHERE c.substitui_id = r.id)"
                " ORDER BY r.referente_a",
                (dono,),
            ).fetchall()
            situacao = con.execute(
                "SELECT s.bebe_id, s.tipo, s.valor_numero, s.valor_texto, s.referente_a"
                " FROM situacao_atual_do_bebe s JOIN bebes b ON b.id = s.bebe_id"
                " WHERE b.cuidador_id = %s",
                (dono,),
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
    # Lembretes de vacina
    # ------------------------------------------------------------------
    def salvar_telefone_para_lembretes(self, numero: str) -> None:
        """Telefone cifrado: só existe para quem pediu lembrete (precisa de
        destino). O número é guardado como o WhatsApp o entregou."""
        self._atualizar_cuidador(numero, "telefone_cifrado = %s", (self.cofre.cifrar(numero),))

    def apagar_telefone(self, numero: str) -> None:
        with self._conectar() as con:
            con.execute(
                "UPDATE cuidadores SET telefone_cifrado = NULL WHERE telefone_hash = %s",
                (_chave(numero),),
            )

    def lembretes_do_dia(self, hoje: dt.date) -> list[dict]:
        """Um item por pessoa a avisar: bebês com mesversário hoje, ainda não
        avisados nesta idade. Avisa o titular e os membros da família que
        pediram lembrete (cada um tem o próprio consentimento e telefone)."""
        from app.lembretes import mesversario

        with self._conectar() as con:
            bebes = con.execute(
                "SELECT b.id, b.data_nascimento, b.sexo, c.telefone_hash"
                " FROM bebes b JOIN cuidadores c ON c.id = b.cuidador_id"
                " WHERE c.status = 'ativa' AND c.relacao IN ('mae','pai','responsavel_legal')"
            ).fetchall()
            feitos = {
                (str(b), m) for b, m in con.execute(
                    "SELECT bebe_id, idade_meses FROM lembretes"
                ).fetchall()
            }
            pendentes = []
            for bebe_id, nascimento, sexo, dono_hash in bebes:
                meses = mesversario(nascimento, hoje)
                if meses is None or (str(bebe_id), meses) in feitos:
                    continue
                destinos = con.execute(
                    "SELECT c.telefone_cifrado FROM cuidadores c"
                    " WHERE c.status = 'ativa' AND c.telefone_cifrado IS NOT NULL"
                    " AND (c.telefone_hash = %s OR c.telefone_hash IN ("
                    "      SELECT m.telefone_hash FROM membros_da_familia m"
                    "      JOIN familias f ON f.id = m.familia_id WHERE f.titular_hash = %s))"
                    " AND (SELECT k.aceito AND k.revogado_em IS NULL FROM consentimentos k"
                    "      WHERE k.telefone_hash = c.telefone_hash AND k.finalidade = 'lembretes'"
                    "      ORDER BY k.decidido_em DESC, k.id DESC LIMIT 1)",
                    (dono_hash, dono_hash),
                ).fetchall()
                for (telefone,) in destinos:
                    pendentes.append({
                        "bebe_id": str(bebe_id), "idade_meses": meses, "sexo": sexo,
                        "telefone": self.cofre.decifrar(telefone),
                    })
        return pendentes

    def registrar_lembrete(self, bebe_id: str, idade_meses: int, status: str) -> None:
        with self._conectar() as con:
            con.execute(
                "INSERT INTO lembretes (bebe_id, idade_meses, status) VALUES (%s, %s, %s)"
                " ON CONFLICT (bebe_id, idade_meses) DO NOTHING",
                (bebe_id, idade_meses, status),
            )

    # ------------------------------------------------------------------
    # Modo família (ver app/familia.py)
    # ------------------------------------------------------------------
    def familia(self, numero: str) -> dict | None:
        chave = _chave(numero)
        with self._conectar() as con:
            familia = self._familia(con, chave)
            if familia is None:
                return None
            membros = con.execute(
                "SELECT telefone_hash, telefone_cifrado FROM membros_da_familia"
                " WHERE familia_id = %s ORDER BY entrou_em",
                (familia[0],),
            ).fetchall()
        return {
            "id": str(familia[0]),
            "eh_titular": familia[1] == chave,
            "membros": [
                {"hash": h, "final": self.cofre.decifrar(t)[-4:]} for h, t in membros
            ],
        }

    def criar_convite_familia(self, numero: str, codigo: str, validade: dt.timedelta) -> None:
        """Cria a família do titular, se ainda não existe, e um convite."""
        agora = self._agora()
        with self._conectar() as con, con.transaction():
            familia = con.execute(
                "INSERT INTO familias (titular_hash) VALUES (%s)"
                " ON CONFLICT (titular_hash) DO UPDATE SET titular_hash = excluded.titular_hash"
                " RETURNING id",
                (_chave(numero),),
            ).fetchone()[0]
            con.execute(
                "INSERT INTO convites_de_familia (codigo, familia_id, criado_em, expira_em)"
                " VALUES (%s, %s, %s, %s)",
                (codigo, familia, agora, agora + validade),
            )

    def situacao_do_convite(self, codigo: str, numero: str, maximo_membros: int) -> str:
        """valido, invalido, expirado, usado, cheia, proprio ou ja_em_familia."""
        chave = _chave(numero)
        with self._conectar() as con:
            convite = con.execute(
                "SELECT c.familia_id, c.expira_em, c.usado_em, f.titular_hash"
                " FROM convites_de_familia c JOIN familias f ON f.id = c.familia_id"
                " WHERE c.codigo = %s",
                (codigo,),
            ).fetchone()
            if convite is None:
                return "invalido"
            familia_id, expira_em, usado_em, titular = convite
            if titular == chave:
                return "proprio"
            if self._familia(con, chave) is not None:
                return "ja_em_familia"
            if usado_em is not None:
                return "usado"
            if expira_em < self._agora():
                return "expirado"
            membros = con.execute(
                "SELECT count(*) FROM membros_da_familia WHERE familia_id = %s", (familia_id,)
            ).fetchone()[0]
        return "cheia" if membros >= maximo_membros else "valido"

    def entrar_na_familia(self, numero: str, codigo: str, maximo_membros: int) -> str:
        """Usa o convite. Devolve "entrou" ou o motivo de não ter entrado."""
        situacao = self.situacao_do_convite(codigo, numero, maximo_membros)
        if situacao != "valido":
            return situacao
        with self._conectar() as con, con.transaction():
            familia_id = con.execute(
                "UPDATE convites_de_familia SET usado_em = %s"
                " WHERE codigo = %s AND usado_em IS NULL RETURNING familia_id",
                (self._agora(), codigo),
            ).fetchone()
            if familia_id is None:
                return "usado"
            con.execute(
                "INSERT INTO membros_da_familia (telefone_hash, familia_id, telefone_cifrado)"
                " VALUES (%s, %s, %s)",
                (_chave(numero), familia_id[0], self.cofre.cifrar(numero)),
            )
            self._garantir_cuidador(con, numero)
        return "entrou"

    def sair_da_familia(self, numero: str) -> bool:
        with self._conectar() as con:
            return con.execute(
                "DELETE FROM membros_da_familia WHERE telefone_hash = %s", (_chave(numero),)
            ).rowcount > 0

    def remover_da_familia(self, titular: str, final: str) -> bool:
        """O titular remove o membro cujo número termina em `final`."""
        info = self.familia(titular)
        if not info or not info["eh_titular"]:
            return False
        alvo = [m["hash"] for m in info["membros"] if m["final"] == final]
        if len(alvo) != 1:
            return False
        with self._conectar() as con:
            con.execute("DELETE FROM membros_da_familia WHERE telefone_hash = %s", (alvo[0],))
        return True

    # ------------------------------------------------------------------
    # Custo de uso (calibrar os limites por plano)
    # ------------------------------------------------------------------
    def registrar_uso(
        self, numero: str | None, tipo: str, custo_usd: float,
        tokens_entrada: int = 0, tokens_saida: int = 0, buscas_web: int = 0,
    ) -> None:
        with self._conectar() as con:
            con.execute(
                "INSERT INTO uso_do_modelo (telefone_hash, tipo, custo_usd, tokens_entrada,"
                " tokens_saida, buscas_web, criado_em) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (_chave(numero) if numero else None, tipo, custo_usd,
                 tokens_entrada, tokens_saida, buscas_web, self._agora()),
            )

    def _hashes_do_plano(self, con, numero: str) -> list[str]:
        """Quem divide as perguntas: a família inteira, se o titular tem o
        Plus em dia; senão, só a própria pessoa."""
        chave = _chave(numero)
        familia = self._familia(con, chave)
        if familia is None:
            return [chave]
        hoje = self._agora().date()
        ativa = con.execute(
            "SELECT 1 FROM assinaturas WHERE telefone_hash = %s AND pago_ate >= %s",
            (familia[1], hoje),
        ).fetchone()
        if not ativa:
            return [chave]
        membros = [h for (h,) in con.execute(
            "SELECT telefone_hash FROM membros_da_familia WHERE familia_id = %s", (familia[0],)
        ).fetchall()]
        return [familia[1], *membros]

    def usos_entre(self, numero: str, inicio: dt.datetime, fim: dt.datetime) -> int:
        with self._conectar() as con:
            return con.execute(
                "SELECT count(*) FROM uso_do_modelo"
                " WHERE telefone_hash = ANY(%s) AND criado_em >= %s AND criado_em < %s",
                (self._hashes_do_plano(con, numero), inicio, fim),
            ).fetchone()[0]

    # ------------------------------------------------------------------
    # Assinaturas
    # ------------------------------------------------------------------
    def assinatura(self, numero: str) -> dict | None:
        """A assinatura que vale para a pessoa: a dela ou, se é membro de uma
        família, a do titular."""
        with self._conectar() as con:
            familia = self._familia(con, _chave(numero))
            dono = familia[1] if familia else _chave(numero)
            linha = con.execute(
                "SELECT plano, inicio, pago_ate, origem FROM assinaturas WHERE telefone_hash = %s",
                (dono,),
            ).fetchone()
        if linha is None:
            return None
        return {"plano": linha[0], "inicio": linha[1], "pago_ate": linha[2], "origem": linha[3]}

    def definir_assinatura(
        self, numero: str, plano: str, inicio: dt.date, pago_ate: dt.date, origem: str = "manual",
    ) -> None:
        """Ativa ou renova. Renovação mantém o dia de início (o ciclo)."""
        with self._conectar() as con:
            con.execute(
                "INSERT INTO assinaturas (telefone_hash, plano, inicio, pago_ate, origem)"
                " VALUES (%s, %s, %s, %s, %s)"
                " ON CONFLICT (telefone_hash) DO UPDATE SET plano = excluded.plano,"
                "  pago_ate = excluded.pago_ate, origem = excluded.origem, atualizado_em = now()",
                (_chave(numero), plano, inicio, pago_ate, origem),
            )

    def cancelar_assinatura(self, numero: str) -> None:
        with self._conectar() as con:
            con.execute("DELETE FROM assinaturas WHERE telefone_hash = %s", (_chave(numero),))

    # ------------------------------------------------------------------
    # Cobrança (Asaas)
    # ------------------------------------------------------------------
    def link_de_pagamento(self, numero: str) -> str | None:
        with self._conectar() as con:
            linha = con.execute(
                "SELECT url FROM links_de_pagamento WHERE telefone_hash = %s"
                " ORDER BY criado_em DESC LIMIT 1",
                (_chave(numero),),
            ).fetchone()
        return linha[0] if linha else None

    def guardar_link_de_pagamento(self, numero: str, link_id: str, url: str) -> None:
        with self._conectar() as con:
            con.execute(
                "INSERT INTO links_de_pagamento (link_id, telefone_hash, telefone_cifrado, url)"
                " VALUES (%s, %s, %s, %s) ON CONFLICT (link_id) DO NOTHING",
                (link_id, _chave(numero), self.cofre.cifrar(numero), url),
            )

    def assinatura_asaas(self, numero: str) -> str | None:
        """Id da assinatura recorrente da própria pessoa no Asaas."""
        with self._conectar() as con:
            linha = con.execute(
                "SELECT assinatura_asaas FROM links_de_pagamento"
                " WHERE telefone_hash = %s AND assinatura_asaas IS NOT NULL"
                " ORDER BY criado_em DESC LIMIT 1",
                (_chave(numero),),
            ).fetchone()
        return linha[0] if linha else None

    def esquecer_assinatura_asaas(self, assinatura_id: str) -> None:
        with self._conectar() as con:
            con.execute(
                "UPDATE links_de_pagamento SET assinatura_asaas = NULL"
                " WHERE assinatura_asaas = %s",
                (assinatura_id,),
            )

    def _dono_do_pagamento(self, con, link_id: str | None, assinatura_id: str | None):
        if link_id:
            linha = con.execute(
                "SELECT link_id, telefone_hash, telefone_cifrado FROM links_de_pagamento"
                " WHERE link_id = %s", (link_id,),
            ).fetchone()
            if linha:
                return linha
        if assinatura_id:
            return con.execute(
                "SELECT link_id, telefone_hash, telefone_cifrado FROM links_de_pagamento"
                " WHERE assinatura_asaas = %s", (assinatura_id,),
            ).fetchone()
        return None

    def aplicar_pagamento(
        self, pagamento_id: str, evento: str, link_id: str | None,
        assinatura_id: str | None, pago_ate: dt.date, hoje: dt.date,
    ) -> dict:
        """Um pagamento confirmado estende o LactAI Plus, uma vez só.

        Tudo numa transação: se algo falhar, o Asaas reenvia e nada fica
        marcado como feito sem ter sido feito. Status: "novo", "repetido" ou
        "desconhecido" (link que não é de ninguém aqui).
        """
        with self._conectar() as con, con.transaction():
            dono = self._dono_do_pagamento(con, link_id, assinatura_id)
            if dono is None:
                return {"status": "desconhecido"}
            novo = con.execute(
                "INSERT INTO pagamentos_processados (pagamento_id, evento) VALUES (%s, %s)"
                " ON CONFLICT (pagamento_id) DO NOTHING RETURNING pagamento_id",
                (pagamento_id, evento),
            ).fetchone()
            if novo is None:
                return {"status": "repetido"}
            link, chave, telefone = dono
            if assinatura_id:
                con.execute(
                    "UPDATE links_de_pagamento SET assinatura_asaas = %s WHERE link_id = %s",
                    (assinatura_id, link),
                )
            atual = con.execute(
                "SELECT inicio, pago_ate FROM assinaturas WHERE telefone_hash = %s", (chave,),
            ).fetchone()
            vigente = atual is not None and atual[1] >= hoje
            inicio = atual[0] if vigente else hoje
            ate = max(pago_ate, atual[1]) if vigente else pago_ate
            con.execute(
                "INSERT INTO assinaturas (telefone_hash, plano, inicio, pago_ate, origem)"
                " VALUES (%s, 'plus', %s, %s, 'asaas')"
                " ON CONFLICT (telefone_hash) DO UPDATE SET plano = 'plus', inicio = excluded.inicio,"
                "  pago_ate = excluded.pago_ate, origem = 'asaas', atualizado_em = now()",
                (chave, inicio, ate),
            )
        return {
            "status": "novo",
            "numero": self.cofre.decifrar(telefone),
            "pago_ate": ate,
            "primeiro": not vigente,
        }

    def estornar_pagamento(
        self, pagamento_id: str, link_id: str | None, assinatura_id: str | None, hoje: dt.date,
    ) -> str:
        """Estorno encerra o Plus hoje. Status como em aplicar_pagamento."""
        with self._conectar() as con, con.transaction():
            dono = self._dono_do_pagamento(con, link_id, assinatura_id)
            if dono is None:
                return "desconhecido"
            novo = con.execute(
                "INSERT INTO pagamentos_processados (pagamento_id, evento) VALUES (%s, 'PAYMENT_REFUNDED')"
                " ON CONFLICT (pagamento_id) DO NOTHING RETURNING pagamento_id",
                (f"{pagamento_id}:estorno",),
            ).fetchone()
            if novo is None:
                return "repetido"
            con.execute(
                "UPDATE assinaturas SET pago_ate = %s, atualizado_em = now()"
                " WHERE telefone_hash = %s AND pago_ate >= %s",
                (hoje - dt.timedelta(days=1), dono[1], hoje),
            )
        return "novo"

    def contar_assinaturas_ativas(self, hoje: dt.date) -> int:
        with self._conectar() as con:
            return con.execute(
                "SELECT count(*) FROM assinaturas WHERE pago_ate >= %s", (hoje,)
            ).fetchone()[0]

    def resumo_de_uso(self, dias: int = 30) -> dict:
        """Custo real nos últimos `dias`: por tipo e por pessoa."""
        desde = self._agora() - dt.timedelta(days=dias)
        with self._conectar() as con:
            por_tipo = con.execute(
                "SELECT tipo, count(*), coalesce(sum(custo_usd), 0), coalesce(avg(custo_usd), 0)"
                " FROM uso_do_modelo WHERE criado_em >= %s GROUP BY tipo ORDER BY tipo",
                (desde,),
            ).fetchall()
            pessoas = con.execute(
                "SELECT count(*), coalesce(avg(n), 0), coalesce(max(n), 0),"
                " coalesce(avg(custo), 0), coalesce(max(custo), 0)"
                " FROM (SELECT telefone_hash, count(*) AS n, sum(custo_usd) AS custo"
                "       FROM uso_do_modelo WHERE criado_em >= %s AND telefone_hash IS NOT NULL"
                "       GROUP BY telefone_hash) p",
                (desde,),
            ).fetchone()
        chamadas = sum(linha[1] for linha in por_tipo)
        total = sum(float(linha[2]) for linha in por_tipo)
        return {
            "dias": dias,
            "chamadas": chamadas,
            "custo_total_usd": total,
            "custo_medio_usd": total / chamadas if chamadas else 0.0,
            "por_tipo": [
                {"tipo": t, "chamadas": n, "custo_usd": float(s), "medio_usd": float(m)}
                for t, n, s, m in por_tipo
            ],
            "pessoas": pessoas[0],
            "perguntas_por_pessoa": float(pessoas[1]),
            "max_perguntas_pessoa": pessoas[2],
            "custo_por_pessoa_usd": float(pessoas[3]),
            "max_custo_pessoa_usd": float(pessoas[4]),
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
            con.execute("DELETE FROM uso_do_modelo WHERE telefone_hash = %s", (chave,))
            # O telefone cifrado do link de pagamento também é dado pessoal.
            con.execute("DELETE FROM links_de_pagamento WHERE telefone_hash = %s", (chave,))
            # Titular apagando desfaz a família; membro apagando sai dela.
            con.execute("DELETE FROM familias WHERE titular_hash = %s", (chave,))
            con.execute("DELETE FROM membros_da_familia WHERE telefone_hash = %s", (chave,))
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
            # Custo de uso só serve para calibrar planos: 13 meses bastam.
            con.execute(
                "DELETE FROM uso_do_modelo WHERE criado_em < %s - interval '13 months'", (agora,)
            )
            apagadas = con.execute(
                "DELETE FROM cuidadores"
                " WHERE status = 'inativa' AND inativa_desde < %s - make_interval(months => %s)",
                (agora, meses_inativa),
            ).rowcount
        if inativadas or apagadas:
            log.info("retenção: %d conta(s) inativada(s), %d apagada(s)", inativadas, apagadas)
        return inativadas, apagadas
