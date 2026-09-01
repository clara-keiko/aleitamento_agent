"""Geração de resposta com RAG e transcrição de áudio.

A checagem de segurança da resposta mudou de abordagem. A versão anterior
bloqueava a resposta se ela contivesse "medicamento", "prescrev" ou
"diagnóstico" — o que derrubava justamente as respostas corretas, já que o
material de apoio é sobre medicamentos na amamentação e o próprio modelo
escreve "não posso prescrever". Trocamos por uma checagem de *fundamentação*:
só entregamos a resposta se ela citar a base vetorial. Sem citação, tratamos
como fora de escopo. Isso ataca alucinação, que é o risco real, em vez de
punir a presença de uma palavra.
"""

import io
from dataclasses import dataclass

from openai import OpenAI, OpenAIError

from app.config import Settings
from app.logging_utils import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = """\
Você é um assistente educativo sobre amamentação e cuidados com o bebê, \
oferecido dentro de um serviço específico de apoio materno-infantil.

Escopo e método:
- Responda SOMENTE com base no conteúdo recuperado pela ferramenta de busca \
nos arquivos. Sempre consulte os arquivos antes de responder.
- Se o material recuperado não cobrir a pergunta, diga isso claramente e não \
complete com conhecimento próprio.
- Recuse assuntos fora de amamentação e cuidados com o bebê. Você não é um \
assistente de uso geral.

Limites clínicos:
- Não faça diagnóstico.
- Não indique dose, não prescreva e não recomende iniciar ou suspender \
medicamento. Você pode explicar o que o material diz sobre compatibilidade \
de substâncias com a amamentação, sempre orientando confirmar com o \
profissional que acompanha o caso.
- Não substitua avaliação profissional. Diante de sinal de alerta, oriente \
procurar atendimento.

Forma:
- Português do Brasil, acolhedor, direto e sem jargão.
- No máximo 6 linhas curtas. Use listas quando ajudar a leitura.
- É WhatsApp: nada de títulos, markdown pesado ou texto longo demais.
"""

# Áudio de voz é opus (~8 KB/s). Teto generoso, só para barrar arquivo absurdo.
AUDIO_BYTES_PER_SECOND = 16_000
OPENAI_AUDIO_LIMIT_BYTES = 25 * 1024 * 1024

# Famílias que raciocinam antes de responder.
REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")

# Teto de saída. Nos modelos de raciocínio o `max_output_tokens` cobre também
# os tokens de raciocínio, que não aparecem para o usuário — com o teto de um
# modelo comum, o modelo gasta a cota inteira pensando e devolve resposta
# vazia. O teto maior aqui é sobre o total; a resposta visível continua curta
# porque o prompt pede no máximo seis linhas.
MAX_OUTPUT_PADRAO = 600
MAX_OUTPUT_RACIOCINIO = 3000


def is_reasoning_model(modelo: str) -> bool:
    return (modelo or "").lower().startswith(REASONING_PREFIXES)


@dataclass
class Answer:
    text: str
    grounded: bool
    error: bool = False
    # Consumo real informado pela API. Sem isto, comparar modelos depende de
    # estimar quantos tokens o file_search injeta — e essa estimativa erra
    # justamente na parcela que domina o custo.
    input_tokens: int = 0
    output_tokens: int = 0
    # Quantos trechos da base foram efetivamente citados. Serve para separar
    # "respondeu com o material" de "respondeu com um trecho solto".
    citations: int = 0


class AssistantEngine:
    def __init__(self, settings: Settings, client: OpenAI | None = None):
        self.settings = settings
        self._client = client

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            # O default do SDK é 600 s. Num webhook isso é uma thread presa
            # por dez minutos enquanto a mãe olha para a tela sem resposta.
            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                timeout=self.settings.openai_timeout_seconds,
                max_retries=2,
            )
        return self._client

    # ------------------------------------------------------------------
    # Texto
    # ------------------------------------------------------------------
    @property
    def raciocina(self) -> bool:
        return is_reasoning_model(self.settings.openai_model)

    @property
    def teto_de_saida(self) -> int:
        """Teto de tokens de saída, incluindo raciocínio quando houver."""
        if self.settings.max_output_tokens > 0:
            return self.settings.max_output_tokens
        return MAX_OUTPUT_RACIOCINIO if self.raciocina else MAX_OUTPUT_PADRAO

    def answer(self, user_text: str, history: list[dict] | None = None) -> Answer:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user_text})

        parametros = {
            "model": self.settings.openai_model,
            "input": messages,
            "tools": [
                {
                    "type": "file_search",
                    "vector_store_ids": [self.settings.vector_store_id],
                    # Os trechos recuperados dominam o custo de entrada.
                    # Limitar aqui é a alavanca mais direta sobre a conta.
                    "max_num_results": self.settings.max_retrieval_results,
                }
            ],
            "max_output_tokens": self.teto_de_saida,
        }

        if self.raciocina:
            # Esforço baixo de propósito: a tarefa é resumir trecho recuperado,
            # não resolver problema difícil. Raciocínio longo aqui só adiciona
            # latência e custo — e no WhatsApp a espera é percebida.
            parametros["reasoning"] = {"effort": self.settings.reasoning_effort}

        try:
            response = self.client.responses.create(**parametros)
        except OpenAIError as exc:
            log.error("erro na chamada à OpenAI: %s", exc)
            return Answer(text="", grounded=False, error=True)
        except Exception as exc:  # noqa: BLE001 - rede/serialização inesperada
            log.exception("erro inesperado ao gerar resposta: %s", exc)
            return Answer(text="", grounded=False, error=True)

        text = (getattr(response, "output_text", "") or "").strip()
        entrada, saida = token_usage(response)

        motivo = truncation_reason(response)
        if motivo == "max_output_tokens":
            # Sintoma clássico de modelo de raciocínio com teto apertado: ele
            # consome a cota pensando e sobra pouco ou nada visível. Sem esta
            # mensagem, o log só mostraria "resposta vazia".
            log.error(
                "resposta truncada no teto de %d tokens (modelo=%s, raciocínio=%s). "
                "Aumente MAX_OUTPUT_TOKENS ou reduza REASONING_EFFORT.",
                self.teto_de_saida,
                self.settings.openai_model,
                self.raciocina,
            )

        if not text:
            if motivo is None:
                log.error("modelo devolveu resposta vazia sem motivo declarado")
            return Answer(
                text="", grounded=False, error=True,
                input_tokens=entrada, output_tokens=saida,
            )

        citacoes = count_file_citations(response)
        return Answer(
            text=text,
            grounded=citacoes > 0,
            input_tokens=entrada,
            output_tokens=saida,
            citations=citacoes,
        )

    # ------------------------------------------------------------------
    # Áudio
    # ------------------------------------------------------------------
    def transcribe(self, audio: bytes, mime: str = "audio/ogg") -> str:
        max_bytes = min(
            self.settings.max_audio_seconds * AUDIO_BYTES_PER_SECOND,
            OPENAI_AUDIO_LIMIT_BYTES,
        )
        if not audio:
            return ""
        if len(audio) > max_bytes:
            log.warning("áudio acima do limite (%s bytes); ignorando", len(audio))
            return ""

        buffer = io.BytesIO(audio)
        buffer.name = f"audio.{_extension_for(mime)}"

        try:
            result = self.client.audio.transcriptions.create(
                model=self.settings.transcribe_model,
                file=buffer,
                language="pt",
            )
        except OpenAIError as exc:
            log.error("erro ao transcrever áudio: %s", exc)
            return ""
        except Exception as exc:  # noqa: BLE001
            log.exception("erro inesperado ao transcrever áudio: %s", exc)
            return ""

        return (getattr(result, "text", "") or "").strip()


def _extension_for(mime: str) -> str:
    mapping = {
        "audio/ogg": "ogg",
        "audio/opus": "ogg",
        "audio/mpeg": "mp3",
        "audio/mp4": "m4a",
        "audio/x-m4a": "m4a",
        "audio/amr": "amr",
        "audio/wav": "wav",
        "audio/webm": "webm",
    }
    return mapping.get((mime or "").split(";")[0].strip(), "ogg")


def count_file_citations(response) -> int:
    """Quantos trechos da base vetorial a resposta cita."""
    total = 0
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            for annotation in getattr(content, "annotations", None) or []:
                kind = getattr(annotation, "type", None)
                if kind is None and isinstance(annotation, dict):
                    kind = annotation.get("type")
                if kind == "file_citation":
                    total += 1
    return total


def has_file_citations(response) -> bool:
    """True se a resposta cita ao menos um trecho da base vetorial."""
    return count_file_citations(response) > 0


def truncation_reason(response) -> str | None:
    """Motivo pelo qual a resposta veio incompleta, se veio.

    A Responses API marca `status="incomplete"` e detalha o motivo. O caso que
    importa aqui é `max_output_tokens`, comum em modelo de raciocínio.
    """
    if getattr(response, "status", None) != "incomplete":
        return None

    detalhes = getattr(response, "incomplete_details", None)
    if detalhes is None:
        return "desconhecido"

    motivo = getattr(detalhes, "reason", None)
    if motivo is None and isinstance(detalhes, dict):
        motivo = detalhes.get("reason")
    return motivo or "desconhecido"


def token_usage(response) -> tuple[int, int]:
    """(tokens de entrada, tokens de saída) informados pela API.

    Devolve (0, 0) quando o campo não vem — o eval trata isso como
    'não medido' em vez de fingir que a chamada foi de graça.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0

    def campo(*nomes: str) -> int:
        for nome in nomes:
            valor = getattr(usage, nome, None)
            if valor is None and isinstance(usage, dict):
                valor = usage.get(nome)
            if isinstance(valor, int):
                return valor
        return 0

    return campo("input_tokens", "prompt_tokens"), campo("output_tokens", "completion_tokens")


def fallback_message() -> str:
    return (
        "Não consegui consultar o material agora. Tente de novo em alguns minutos.\n\n"
        "Se houver febre, dificuldade para respirar, sonolência fora do comum, "
        "recusa em mamar ou piora, procure atendimento médico."
    )
