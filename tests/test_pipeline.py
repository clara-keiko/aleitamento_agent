import pytest

from app.channels.base import IncomingMessage
from app.config import Settings
from app.llm import Answer
from app.pipeline import MessagePipeline


class FakeChannel:
    name = "fake"

    def __init__(self, media: bytes | None = None):
        self.sent: list[tuple[str, str]] = []
        self.media = media

    def verify_signature(self, raw_body, headers, url):
        return True

    def parse_webhook(self, payload):
        return []

    def send_text(self, to, body):
        self.sent.append((to, body))
        return True

    def fetch_media(self, message):
        return self.media

    @property
    def bodies(self) -> str:
        return "\n".join(body for _, body in self.sent)


class FakeEngine:
    def __init__(self, answer: Answer | None = None, transcript: str = ""):
        self.answer_value = answer or Answer(text="Resposta educativa.", grounded=True)
        self.transcript = transcript
        self.calls: list[tuple[str, list]] = []

    def answer(self, text, history=None):
        self.calls.append((text, history or []))
        return self.answer_value

    def transcribe(self, audio, mime="audio/ogg"):
        return self.transcript


def build(answer=None, transcript="", media=None, **overrides):
    kwargs = {
        "openai_api_key": "k",
        "vector_store_id": "vs",
        "whatsapp_token": "t",
        "phone_number_id": "p",
        "verify_token": "v",
        "app_secret": "s",
    }
    kwargs.update(overrides)
    settings = Settings(**kwargs)
    channel = FakeChannel(media=media)
    engine = FakeEngine(answer=answer, transcript=transcript)
    return MessagePipeline(settings, channel, engine), channel, engine


def text_message(body: str, message_id: str = "wamid.1", sender: str = "5511999999999"):
    return IncomingMessage(message_id=message_id, sender=sender, kind="text", text=body)


class TestPrimeiroContato:
    def test_boas_vindas_apenas_uma_vez(self):
        pipeline, channel, _ = build()
        pipeline.handle(text_message("como pegar a pega correta?", "m1"))
        pipeline.handle(text_message("e a posição?", "m2"))

        boas_vindas = [b for _, b in channel.sent if "assistente educativo" in b]
        assert len(boas_vindas) == 1

    def test_boas_vindas_sobrevivem_a_expiracao_da_conversa(self):
        """O histórico expira em 30 min; reapresentar o serviço a cada meia
        hora seria ruído. As duas coisas têm TTL separado."""
        from app.memory import ConversationStore

        pipeline, channel, _ = build()
        # Conversa que expira imediatamente, usuário conhecido permanece.
        pipeline.conversations = ConversationStore(ttl_seconds=0)

        pipeline.handle(text_message("qual a pega correta?", "m1"))
        pipeline.handle(text_message("e a posição?", "m2"))

        boas_vindas = [b for _, b in channel.sent if "assistente educativo" in b]
        assert len(boas_vindas) == 1

    def test_boas_vindas_declaram_automacao_e_emergencia(self):
        pipeline, channel, _ = build()
        pipeline.handle(text_message("oi", "m1"))
        primeira = channel.sent[0][1]
        assert "automatizada" in primeira
        assert "192" in primeira


class TestDeduplicacao:
    def test_mesmo_id_nao_responde_duas_vezes(self):
        """O retry da Meta não pode virar resposta duplicada."""
        pipeline, channel, engine = build()
        msg = text_message("qual a melhor posição para amamentar?", "wamid.repetido")

        pipeline.handle(msg)
        antes = len(channel.sent)
        pipeline.handle(msg)

        assert len(channel.sent) == antes
        assert len(engine.calls) == 1


class TestTriagem:
    def test_emergencia_nao_chama_o_modelo(self):
        pipeline, channel, engine = build()
        pipeline.handle(text_message("meu bebe nao respira"))

        assert engine.calls == []
        assert "192" in channel.bodies

    def test_encaminhamento_nao_chama_o_modelo(self):
        pipeline, channel, engine = build()
        pipeline.handle(text_message("meu bebê está com febre desde ontem"))

        assert engine.calls == []
        assert "sinal de alerta" in channel.bodies

    def test_pergunta_geral_sensivel_recebe_resposta_e_nota(self):
        pipeline, channel, engine = build()
        pipeline.handle(text_message("posso amamentar se eu estiver com febre?"))

        assert len(engine.calls) == 1
        assert "Resposta educativa." in channel.bodies
        assert "informação geral" in channel.bodies


class TestFundamentacao:
    def test_resposta_sem_citacao_vira_fora_de_escopo(self):
        """Sem citação da base tratamos como alucinação, não entregamos."""
        pipeline, channel, _ = build(
            answer=Answer(text="A capital da França é Paris.", grounded=False)
        )
        pipeline.handle(text_message("qual a capital da frança?"))

        assert "Paris" not in channel.bodies
        assert "amamentação" in channel.bodies

    def test_erro_do_modelo_devolve_fallback_seguro(self):
        pipeline, channel, _ = build(answer=Answer(text="", grounded=False, error=True))
        pipeline.handle(text_message("como aumentar a produção de leite?"))

        assert "Tente de novo" in channel.bodies

    def test_termo_medicamento_nao_bloqueia_mais_a_resposta(self):
        """Regressão: o filtro antigo derrubava respostas corretas."""
        texto = "O material indica checar a compatibilidade do medicamento com o profissional."
        pipeline, channel, _ = build(answer=Answer(text=texto, grounded=True))
        pipeline.handle(text_message("posso tomar dipirona amamentando?"))

        assert texto in channel.bodies


class TestHistorico:
    def test_contexto_e_repassado_ao_modelo(self):
        pipeline, _, engine = build()
        pipeline.handle(text_message("como aumentar a produção de leite?", "m1"))
        pipeline.handle(text_message("e quanto tempo demora?", "m2"))

        _, history = engine.calls[1]
        assert any(turn["role"] == "user" for turn in history)
        assert any(turn["role"] == "assistant" for turn in history)

    def test_resposta_nao_entregue_fica_fora_do_historico(self):
        pipeline, _, engine = build(answer=Answer(text="ruído", grounded=False))
        pipeline.handle(text_message("pergunta fora de escopo", "m1"))
        pipeline.handle(text_message("outra pergunta", "m2"))

        _, history = engine.calls[1]
        assert history == []


class TestOptOut:
    def test_apaga_historico_e_confirma(self):
        pipeline, channel, engine = build()
        pipeline.handle(text_message("como aumentar a produção de leite?", "m1"))
        pipeline.handle(text_message("SAIR", "m2"))
        pipeline.handle(text_message("como aumentar a produção de leite?", "m3"))

        assert "apaguei o histórico" in channel.bodies
        _, history = engine.calls[-1]
        assert history == []


class TestAudio:
    def test_audio_e_transcrito_e_respondido(self):
        pipeline, channel, engine = build(transcript="qual a pega correta?", media=b"fake-audio")
        pipeline.handle(
            IncomingMessage(
                message_id="a1", sender="5511999999999", kind="audio", media_id="mid"
            )
        )

        assert engine.calls[0][0] == "qual a pega correta?"
        assert "Resposta educativa." in channel.bodies

    def test_audio_com_emergencia_pula_o_modelo(self):
        pipeline, channel, engine = build(transcript="o bebe nao respira", media=b"fake-audio")
        pipeline.handle(
            IncomingMessage(
                message_id="a2", sender="5511999999999", kind="audio", media_id="mid"
            )
        )

        assert engine.calls == []
        assert "192" in channel.bodies

    def test_audio_desligado_avisa_o_usuario(self):
        pipeline, channel, engine = build(media=b"fake-audio", enable_audio=False)
        pipeline.handle(
            IncomingMessage(
                message_id="a3", sender="5511999999999", kind="audio", media_id="mid"
            )
        )

        assert engine.calls == []
        assert "texto e áudio" in channel.bodies

    def test_tipo_nao_suportado(self):
        pipeline, channel, engine = build()
        pipeline.handle(
            IncomingMessage(message_id="s1", sender="5511999999999", kind="unsupported")
        )

        assert engine.calls == []
        assert "texto e áudio" in channel.bodies


class TestSmallTalk:
    """Regressão: "obrigada" caía na checagem de fundamentação e recebia
    "não encontrei isso no material" — resposta rude para quem agradece."""

    def test_agradecimento_nao_vai_ao_modelo(self):
        pipeline, channel, engine = build()
        pipeline.handle(text_message("obrigada!", "m1"))

        assert engine.calls == []
        assert "Estou por aqui" in channel.bodies

    def test_saudacao_simples(self):
        pipeline, channel, engine = build()
        pipeline.handle(text_message("bom dia", "m1"))
        assert engine.calls == []

    def test_so_emoji(self):
        pipeline, channel, engine = build()
        pipeline.handle(text_message("👍", "m1"))
        assert engine.calls == []

    def test_pergunta_real_nao_e_confundida_com_social(self):
        pipeline, _, engine = build()
        pipeline.handle(text_message("oi, como aumento a produção de leite?", "m1"))
        assert len(engine.calls) == 1

    def test_emergencia_vence_saudacao(self):
        """"oi, meu bebê não respira" é emergência, não cumprimento."""
        pipeline, channel, engine = build()
        pipeline.handle(text_message("oi, meu bebe nao respira", "m1"))

        assert engine.calls == []
        assert "192" in channel.bodies


class TestRateLimit:
    def test_excesso_de_mensagens_e_contido(self):
        pipeline, channel, engine = build(rate_limit_messages=3)
        for i in range(6):
            pipeline.handle(text_message("qual a melhor posição?", f"m{i}"))

        assert len(engine.calls) == 3
        assert "pausa curta" in channel.bodies


@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("", 0),
        ("   ", 0),
    ],
)
def test_texto_vazio_nao_chama_o_modelo(texto, esperado):
    pipeline, _, engine = build()
    pipeline.handle(text_message(texto))
    assert len(engine.calls) == esperado
