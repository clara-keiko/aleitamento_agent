"""Questionário inicial: quem cuida, o bebê e a mãe.

Roda logo depois do "pode guardar" (e de novo com PERFIL). Uma pergunta por
mensagem, com resposta curta por número ou valor; PULAR pula, PARAR encerra.
Cada resposta é gravada na hora — se o processo reiniciar no meio, o que já
foi respondido fica.

Dado do bebê só existe se quem responde é mãe, pai ou responsável legal
(LGPD art. 14): para os outros cuidadores o questionário termina na primeira
pergunta, e o atendimento segue igual.
"""

import datetime as dt
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from app.guardrails import normalize
from app.piloto import BRASILIA

PULAR = {"pular", "pula", "nao sei", "prefiro nao dizer", "prefiro nao responder"}
PARAR = {"parar questionario", "parar", "depois", "agora nao", "sair do questionario"}

RELACOES = {"1": "mae", "mae": "mae", "2": "pai", "pai": "pai",
            "3": "responsavel_legal", "responsavel": "responsavel_legal",
            "responsavel legal": "responsavel_legal",
            "4": "outro", "outro": "outro", "outra": "outro", "outra pessoa": "outro",
            "avo": "outro", "tia": "outro", "baba": "outro"}
FASES = {"1": "gestante", "gestante": "gestante", "gravida": "gestante",
         "ainda nao": "gestante", "2": "nasceu", "nasceu": "nasceu", "ja nasceu": "nasceu"}
SEXOS = {"1": "feminino", "menina": "feminino", "feminino": "feminino",
         "2": "masculino", "menino": "masculino", "masculino": "masculino"}
ALEITAMENTO = {"1": "exclusivo", "so peito": "exclusivo", "peito": "exclusivo",
               "2": "misto", "peito e formula": "misto", "misto": "misto",
               "3": "formula", "so formula": "formula", "formula": "formula",
               "4": "complementar", "complementar": "complementar"}
NOMES_ALEITAMENTO = {"exclusivo": "só peito", "misto": "peito e fórmula",
                     "formula": "só fórmula", "complementar": "leite e outros alimentos"}

TEXTO = {
    "relacao": (
        "Para eu te conhecer melhor, umas perguntas rápidas — leva menos de "
        "um minuto. Pode responder *PULAR* em qualquer uma, ou *PARAR* para "
        "continuar outra hora.\n\n"
        "*Você é:*\n1 – Mãe\n2 – Pai\n3 – Responsável legal\n4 – Outra pessoa\n\n"
        "_Se você não for mãe, pai ou responsável legal, continuo te atendendo "
        "normalmente, mas não guardo informações sobre o bebê: a lei de proteção "
        "de dados (LGPD) só permite isso com a autorização deles._"
    ),
    "fase": "*O bebê já nasceu?*\n1 – Ainda não, a gestação está em andamento\n2 – Já nasceu",
    "dpp": "*Qual a data prevista para o parto?* (ex.: 15/03/2027)",
    "nascimento": "*Qual a data de nascimento do bebê?* (ex.: 03/10/2026)",
    "sexo": "*É menina ou menino?*\n1 – Menina\n2 – Menino",
    "peso": "*Qual foi o último peso do bebê?* (ex.: 4,2 kg)",
    "comprimento": "*E o último comprimento (altura)?* (ex.: 55 cm)",
    "aleitamento": (
        "*Como o bebê está se alimentando agora?*\n1 – Só peito\n"
        "2 – Peito e fórmula\n3 – Só fórmula\n4 – Leite e outros alimentos"
    ),
    "idade_mae_mae": "*Por último: quantos anos você tem?*",
    "idade_mae_outro": "*Por último: quantos anos tem a mãe do bebê?*",
}

FIM_OUTRO = (
    "Obrigada! 💛 Como você não é mãe, pai ou responsável legal, eu não guardo "
    "informações sobre o bebê — é uma regra de proteção de dados de crianças. "
    "Continuo respondendo todas as suas dúvidas normalmente."
)

FIM_SEM_RELACAO = (
    "Tudo bem! Sem saber se você é mãe, pai ou responsável legal, eu não "
    "guardo informações sobre o bebê — é uma regra de proteção de dados de "
    "crianças. Se quiser responder depois, mande *PERFIL*. Qual é a sua dúvida?"
)


@dataclass
class _Estado:
    passo: str
    # Quem entrou pela família: o bebê já está no perfil; só falta saber
    # quem é a pessoa.
    somente_relacao: bool = False
    relacao: str | None = None
    bebe_id: str | None = None
    anotado: list[str] = field(default_factory=list)


def _limpo(texto: str) -> str:
    return normalize(texto).strip(" .!?")


def ler_data(texto: str, hoje: dt.date, futuro: bool) -> dt.date | None:
    """dd/mm/aaaa, dd/mm/aa ou dd/mm. Sem ano: a data mais próxima no
    passado (nascimento) ou no futuro (parto)."""
    achado = re.search(r"(\d{1,2})\s*[/.\-]\s*(\d{1,2})(?:\s*[/.\-]\s*(\d{2,4}))?", texto)
    if not achado:
        return None
    dia, mes, ano = int(achado[1]), int(achado[2]), achado[3]
    try:
        if ano:
            ano_n = int(ano) + (2000 if len(ano) == 2 else 0)
            return dt.date(ano_n, mes, dia)
        data = dt.date(hoje.year, mes, dia)
        if futuro and data < hoje:
            data = dt.date(hoje.year + 1, mes, dia)
        if not futuro and data > hoje:
            data = dt.date(hoje.year - 1, mes, dia)
        return data
    except ValueError:
        return None


def _numero(texto: str) -> float | None:
    achado = re.search(r"\d+(?:[.,]\d+)?", texto)
    return float(achado[0].replace(",", ".")) if achado else None


def ler_peso_g(texto: str) -> int | None:
    valor = _numero(texto)
    if valor is None:
        return None
    gramas = valor if valor > 100 or re.search(r"\d\s*g\b", normalize(texto)) else valor * 1000
    return int(round(gramas)) if 300 <= gramas <= 30000 else None


def ler_comprimento_cm(texto: str) -> float | None:
    valor = _numero(texto)
    if valor is None:
        return None
    cm = valor * 100 if valor < 2 else valor
    return round(cm, 1) if 25 <= cm <= 130 else None


def ler_idade(texto: str) -> int | None:
    valor = _numero(texto)
    return int(valor) if valor is not None and 12 <= valor <= 60 else None


class Questionario:
    def __init__(self, memoria, agora: Callable[[], dt.datetime] | None = None):
        self.memoria = memoria
        self._agora = agora or (lambda: dt.datetime.now(BRASILIA))
        self._estados: dict[str, _Estado] = {}

    def _hoje(self) -> dt.date:
        return self._agora().astimezone(BRASILIA).date()

    def e_parada(self, texto: str) -> bool:
        return _limpo(texto) in PARAR

    def ativo(self, numero: str) -> bool:
        return numero in self._estados

    def encerrar(self, numero: str) -> None:
        self._estados.pop(numero, None)

    def iniciar(self, numero: str, somente_relacao: bool = False) -> str:
        self._estados[numero] = _Estado("relacao", somente_relacao=somente_relacao)
        if somente_relacao:
            return "Para eu te conhecer: *você é:*\n1 – Mãe\n2 – Pai\n3 – Responsável legal\n4 – Outra pessoa"
        return TEXTO["relacao"]

    def _pergunta(self, estado: _Estado) -> str:
        if estado.passo == "idade_mae":
            return TEXTO["idade_mae_mae" if estado.relacao == "mae" else "idade_mae_outro"]
        return TEXTO[estado.passo]

    def pergunta_atual(self, numero: str) -> str:
        return self._pergunta(self._estados[numero])

    def responder(self, numero: str, texto: str) -> str | None:
        """Mensagem a enviar, ou None se o texto não é resposta (é uma dúvida)."""
        estado = self._estados[numero]
        limpo = _limpo(texto)

        if limpo in PARAR:
            self.encerrar(numero)
            return (
                "Tudo bem, paramos por aqui. Quando quiser continuar, mande "
                "*PERFIL*. Qual é a sua dúvida?"
            )
        pular = limpo in PULAR

        proximo = self._aplicar(numero, estado, texto, limpo, pular)
        if proximo is None:
            # Resposta que não serve para a pergunta: se parece dúvida, deixa
            # o pipeline responder; se é curta, pede de novo.
            if len(limpo) > 25 or "?" in texto:
                return None
            return "Não entendi 🙈 " + self._pergunta(estado)
        if proximo == "fim_relacao":
            self.encerrar(numero)
            return "Anotado! 💛 Agora, qual é a sua dúvida?"
        if proximo in ("outro", "sem_relacao"):
            self.encerrar(numero)
            return FIM_OUTRO if proximo == "outro" else FIM_SEM_RELACAO
        if proximo == "fim":
            self.encerrar(numero)
            return self._resumo(estado)
        estado.passo = proximo
        return self._pergunta(estado)

    def _aplicar(self, numero, estado, texto, limpo, pular) -> str | None:
        """Grava a resposta e devolve o próximo passo; None se não entendeu."""
        hoje = self._hoje()
        passo = estado.passo
        m = self.memoria

        if passo == "relacao" and estado.somente_relacao:
            if pular:
                return "fim_relacao"
            relacao = RELACOES.get(limpo)
            if not relacao:
                return None
            m.salvar_relacao(numero, relacao)
            return "fim_relacao"

        if passo == "relacao":
            if pular:
                # Sem saber se é mãe, pai ou responsável legal, não há base
                # para guardar dado do bebê.
                return "sem_relacao"
            relacao = RELACOES.get(limpo)
            if not relacao:
                return None
            m.salvar_relacao(numero, relacao)
            estado.relacao = relacao
            return "outro" if relacao == "outro" else "fase"

        if passo == "fase":
            if pular:
                return "idade_mae"
            fase = FASES.get(limpo)
            if not fase:
                return None
            return "dpp" if fase == "gestante" else "nascimento"

        if passo == "dpp":
            if not pular:
                data = ler_data(texto, hoje, futuro=True)
                if not data or not hoje <= data <= hoje + dt.timedelta(days=300):
                    return None
                m.salvar_gestacao(numero, data)
                estado.anotado.append(f"parto previsto para {data:%d/%m/%Y}")
            return "idade_mae"

        if passo == "nascimento":
            if pular:
                # Sem data de nascimento não há bebê para registrar.
                return "idade_mae"
            data = ler_data(texto, hoje, futuro=False)
            if not data or not hoje - dt.timedelta(days=6 * 366) <= data <= hoje:
                return None
            estado.bebe_id = m.salvar_bebe(numero, data)
            estado.anotado.append(f"nasceu em {data:%d/%m/%Y}")
            return "sexo"

        if passo == "sexo":
            if not pular:
                sexo = SEXOS.get(limpo)
                if not sexo:
                    return None
                m.salvar_sexo(estado.bebe_id, sexo)
                estado.anotado.append("menina" if sexo == "feminino" else "menino")
            return "peso"

        if passo == "peso":
            if not pular:
                gramas = ler_peso_g(texto)
                if gramas is None:
                    return None
                m.registrar_do_bebe(estado.bebe_id, "peso_g", hoje, numero_valor=gramas)
                estado.anotado.append(f"{gramas / 1000:.2f} kg".replace(".", ","))
            return "comprimento"

        if passo == "comprimento":
            if not pular:
                cm = ler_comprimento_cm(texto)
                if cm is None:
                    return None
                m.registrar_do_bebe(estado.bebe_id, "altura_cm", hoje, numero_valor=cm)
                estado.anotado.append(f"{cm:g} cm".replace(".", ","))
            return "aleitamento"

        if passo == "aleitamento":
            if not pular:
                tipo = ALEITAMENTO.get(limpo)
                if not tipo:
                    return None
                m.registrar_do_bebe(estado.bebe_id, "aleitamento", hoje, texto_valor=tipo)
                estado.anotado.append(NOMES_ALEITAMENTO[tipo])
            return "idade_mae"

        if passo == "idade_mae":
            if not pular:
                idade = ler_idade(texto)
                if idade is None:
                    return None
                m.salvar_ano_nascimento_mae(numero, hoje.year - idade)
                quem = "você" if estado.relacao == "mae" else "mãe"
                estado.anotado.append(f"{quem}: {idade} anos")
            return "fim"

        return None

    def _resumo(self, estado: _Estado) -> str:
        anotado = "; ".join(estado.anotado) if estado.anotado else "nada por enquanto"
        return (
            f"Pronto, anotei! 📒 {anotado[0].upper() + anotado[1:]}.\n\n"
            "Vou levar isso em conta nas respostas. Quando o bebê for pesado ou "
            "medido de novo, é só me contar — ou mandar *PERFIL* para "
            "responder tudo de novo. Para ver o que guardei, *MEUS DADOS*.\n\n"
            "Agora, qual é a sua dúvida?"
        )


def idade_em_texto(nascimento: dt.date, hoje: dt.date) -> str:
    meses = (hoje.year - nascimento.year) * 12 + hoje.month - nascimento.month
    if hoje.day < nascimento.day:
        meses -= 1
    if meses < 1:
        dias = (hoje - nascimento).days
        return f"{dias} dia{'s' if dias != 1 else ''}"
    if meses < 24:
        return f"{meses} {'mês' if meses == 1 else 'meses'}"
    return f"{meses // 12} anos e {meses % 12} meses"


_RELACAO = {"mae": "mãe", "pai": "pai", "responsavel_legal": "responsável legal",
            "outro": "outro cuidador(a)"}


def descrever_perfil(perfil: dict | None, hoje: dt.date) -> str:
    """Perfil em linhas curtas: vai para o modelo e para o MEUS DADOS."""
    if not perfil:
        return ""
    linhas = []
    if perfil.get("relacao"):
        linhas.append(f"Quem conversa: {_RELACAO[perfil['relacao']]}")
    if perfil.get("ano_nascimento_mae"):
        linhas.append(f"Idade da mãe: cerca de {hoje.year - perfil['ano_nascimento_mae']} anos")
    if perfil.get("gestante") and perfil.get("data_provavel_parto"):
        linhas.append(f"Gestante, parto previsto para {perfil['data_provavel_parto']:%d/%m/%Y}")
    for bebe in perfil.get("bebes", []):
        partes = [f"nascido em {bebe['nascimento']:%d/%m/%Y} "
                  f"({idade_em_texto(bebe['nascimento'], hoje)})"]
        if bebe.get("sexo"):
            partes.insert(0, "menina" if bebe["sexo"] == "feminino" else "menino")
        if "peso_g" in bebe:
            gramas, em = bebe["peso_g"]
            partes.append(f"último peso {gramas / 1000:.2f} kg em {em:%d/%m/%Y}".replace(".", ",", 1))
        if "altura_cm" in bebe:
            cm, em = bebe["altura_cm"]
            partes.append(f"comprimento {cm:g} cm em {em:%d/%m/%Y}")
        if "aleitamento" in bebe:
            partes.append(f"alimentação: {NOMES_ALEITAMENTO.get(bebe['aleitamento'][0], bebe['aleitamento'][0])}")
        linhas.append("Bebê: " + ", ".join(partes))
        if bebe.get("vacinas"):
            anotadas = "; ".join(f"{r} ({d:%d/%m/%Y})" for r, d in bebe["vacinas"])
            linhas.append(f"Vacinas anotadas (lidas da caderneta, conferidas pela pessoa): {anotadas}")
    return "\n".join(linhas)


# ----------------------------------------------------------------------
# Informação solta no meio da conversa ("ela pesou 5,2 kg ontem")
# ----------------------------------------------------------------------

_IDADE = re.compile(r"\b(?:eu )?tenho (\d{2}) anos\b")
_PESO = re.compile(
    r"\b(?:pesou|pesa|pesando|peso|esta com|ta com)\D{0,12}?(\d+(?:[.,]\d+)?)\s*"
    r"(kg|kilos?|quilos?|g|gramas)\b"
)
_COMPRIMENTO = re.compile(
    r"\b(?:mede|medindo|mediu|comprimento|altura|tamanho)\D{0,15}?(\d+(?:[.,]\d+)?)\s*"
    r"(cm|centimetros?)\b"
)


def ler_informacao_avulsa(texto: str) -> tuple[str, float] | None:
    """Dado de perfil dito de passagem, em frase curta e sem pergunta.

    Devolve (tipo, valor): ("idade_mae", anos), ("peso_g", gramas) ou
    ("altura_cm", cm). Pergunta ("pesa 5 kg, é normal?") não conta: aí a
    pessoa quer resposta, não cadastro.
    """
    if "?" in texto:
        return None
    limpo = normalize(texto)
    if len(limpo) > 80:
        return None
    if achado := _IDADE.search(limpo):
        idade = ler_idade(achado[1])
        return ("idade_mae", idade) if idade else None
    if achado := _PESO.search(limpo):
        gramas = ler_peso_g(f"{achado[1]} {achado[2][0]}")
        return ("peso_g", gramas) if gramas else None
    if achado := _COMPRIMENTO.search(limpo):
        cm = ler_comprimento_cm(achado[1])
        return ("altura_cm", cm) if cm else None
    return None


def descrever_atualizacao(tipo: str, valor: float, hoje: dt.date) -> str:
    if tipo == "idade_mae":
        return f"Sua idade: {int(valor)} anos"
    if tipo == "peso_g":
        return f"Peso do bebê: {valor / 1000:.2f} kg em {hoje:%d/%m/%Y}".replace(".", ",", 1)
    return f"Comprimento do bebê: {valor:g} cm em {hoje:%d/%m/%Y}".replace(".", ",")
