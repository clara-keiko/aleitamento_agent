"""Lembrete de vacina no dia do mesversário.

Fora da janela de 24 h o WhatsApp só deixa mandar modelo aprovado pela Meta,
com texto fixo. Por isso o lembrete não lista vacinas: avisa que há vacina
prevista para a idade e pede para responder VACINAS; a lista sai na resposta,
pelas fontes oficiais (SUS e SBIm), atualizada.

Só vai em mesversário com vacina no calendário do SUS — conferido no
calendário baixado do gov.br, nunca numa lista fixa no código. Sem
calendário, não manda (tenta de novo na próxima rodada).

Só para quem pediu (consentimento "lembretes"); PARAR LEMBRETES cancela.
"""

import calendar
import datetime as dt
import json
import time
from collections.abc import Callable

from app.logging_utils import get_logger, pseudonymize
from app.piloto import BRASILIA

log = get_logger(__name__)

LEMBRETES = "lembretes"
VERSAO_LEMBRETES = "lembretes-v1"
HORARIO = range(9, 19)  # 9h às 18h59 de Brasília
IDADE_MAXIMA_MESES = 72


def mesversario(nascimento: dt.date, hoje: dt.date) -> int | None:
    """Meses completados hoje, se hoje é mesversário. Nasceu dia 31: o
    mesversário de fevereiro é no último dia do mês."""
    meses = (hoje.year - nascimento.year) * 12 + hoje.month - nascimento.month
    if not 1 <= meses <= IDADE_MAXIMA_MESES:
        return None
    ultimo_dia = calendar.monthrange(hoje.year, hoje.month)[1]
    return meses if hoje.day == min(nascimento.day, ultimo_dia) else None


def idade_por_extenso(meses: int) -> str:
    if meses < 12:
        return f"{meses} {'mês' if meses == 1 else 'meses'}"
    anos, resto = divmod(meses, 12)
    texto = f"{anos} {'ano' if anos == 1 else 'anos'}"
    return texto + (f" e {resto} {'mês' if resto == 1 else 'meses'}" if resto else "")


PERGUNTA_CALENDARIO = """\
Você recebe o Calendário Nacional de Vacinação (SUS) copiado do site do \
Ministério da Saúde. Responda se há alguma vacina de rotina prevista para \
crianças exatamente na idade informada. Use só esse texto."""


def ha_vacina_na_idade(engine, meses: int) -> bool | None:
    """True/False pelo calendário oficial; None se não dá para saber agora."""
    calendario = engine.calendario_sus.atual if engine.calendario_sus else None
    if calendario is None:
        return None
    try:
        resposta = engine.client.responses.create(
            model=engine.settings.openai_model,
            input=[
                {"role": "system", "content": PERGUNTA_CALENDARIO},
                {"role": "system", "content": calendario.texto},
                {"role": "user", "content": f"Idade: {idade_por_extenso(meses)}."},
            ],
            text={"format": {
                "type": "json_schema", "name": "ha_vacina", "strict": True,
                "schema": {"type": "object", "additionalProperties": False,
                           "required": ["ha_vacina"],
                           "properties": {"ha_vacina": {"type": "boolean"}}},
            }},
            max_output_tokens=engine.teto_de_saida,
            **({"reasoning": {"effort": "low"}} if engine.raciocina else {}),
        )
        return bool(json.loads(resposta.output_text)["ha_vacina"])
    except Exception as exc:  # noqa: BLE001
        log.error("lembretes: não consegui consultar o calendário: %s", exc)
        return None


class LembretesDeVacina:
    def __init__(
        self, memoria, engine, channel, settings,
        agora: Callable[[], dt.datetime] | None = None,
        consultar: Callable[[int], bool | None] | None = None,
        pausa_s: float = 3.0,
    ):
        self.memoria = memoria
        self.engine = engine
        self.channel = channel
        self.settings = settings
        self._agora = agora or (lambda: dt.datetime.now(BRASILIA))
        self._consultar = consultar or (lambda meses: ha_vacina_na_idade(engine, meses))
        self._pausa_s = pausa_s
        # Mesma idade, mesmo calendário, mesma resposta: uma consulta por dia.
        self._cache: dict[tuple[dt.date, int], bool] = {}

    def _ha_vacina(self, hoje: dt.date, meses: int) -> bool | None:
        chave = (hoje, meses)
        if chave not in self._cache:
            resposta = self._consultar(meses)
            if resposta is None:
                return None
            self._cache[chave] = resposta
        return self._cache[chave]

    def rodar(self) -> dict:
        agora = self._agora().astimezone(BRASILIA)
        contagem = {"enviados": 0, "sem_vacina": 0, "adiados": 0, "falhas": 0}
        if agora.hour not in HORARIO:
            return contagem
        hoje = agora.date()

        for item in self.memoria.lembretes_do_dia(hoje):
            meses = item["idade_meses"]
            ha_vacina = self._ha_vacina(hoje, meses)
            if ha_vacina is None:
                contagem["adiados"] += 1
                continue
            if not ha_vacina:
                self.memoria.registrar_lembrete(item["bebe_id"], meses, "sem_vacina")
                contagem["sem_vacina"] += 1
                continue

            bebe = "sua bebê" if item["sexo"] == "feminino" else "seu bebê"
            enviado = self.channel.send_template(
                item["telefone"], self.settings.template_lembrete_vacina,
                self.settings.idioma_template, [bebe, idade_por_extenso(meses)],
            )
            if enviado:
                self.memoria.registrar_lembrete(item["bebe_id"], meses, "enviado")
                contagem["enviados"] += 1
                log.info("lembrete de vacina enviado a %s (%d meses)",
                         pseudonymize(item["telefone"]), meses)
            else:
                # Sem registro: tenta de novo na próxima rodada do dia.
                contagem["falhas"] += 1
            time.sleep(self._pausa_s)

        if any(contagem.values()):
            log.info("lembretes de vacina: %s", contagem)
        return contagem
