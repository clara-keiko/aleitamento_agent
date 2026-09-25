"""Canal Evolution API.

Dois motivos para estes testes serem detalhados apesar de o canal ser o
"plano C": o formato do webhook foi escrito a partir do que se conhece da
Baileys, sem conferir a documentação corrente, e é justamente por isso que
cada forma aceita precisa estar fixada — se a Evolution mudar, o teste que
quebrar aponta a linha a ajustar.

O resto cobre os dois jeitos de o canal virar um problema em produção:
responder ao próprio eco (laço infinito) e responder em grupo (caminho mais
curto para denúncia e banimento).
"""

import base64

import pytest

from app.channels.evolution import EvolutionChannel
from app.config import Settings

CHAVE = "chave-da-evolution"


def canal(**overrides):
    kwargs = {
        "provider": "evolution",
        "evolution_base_url": "https://evo.exemplo.com/",
        "evolution_api_key": CHAVE,
        "evolution_instance": "lactai",
        "openai_api_key": "k",
        "vector_store_id": "vs",
    }
    kwargs.update(overrides)
    return EvolutionChannel(Settings(**kwargs))


def evento(message: dict, *, from_me=False, jid="5521999999999@s.whatsapp.net",
           id_="3EB0ABC"):
    return {
        "event": "messages.upsert",
        "instance": "lactai",
        "data": {
            "key": {"remoteJid": jid, "fromMe": from_me, "id": id_},
            "pushName": "Fulana",
            "message": message,
        },
    }


class FakeResposta:
    def __init__(self, status=200, corpo=None, texto=""):
        self.status_code = status
        self._corpo = corpo if corpo is not None else {}
        self.text = texto

    def json(self):
        return self._corpo

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}")


class TestAutenticidade:
    def test_apikey_correta_passa(self):
        assert canal().verify_signature(b"", {"apikey": CHAVE}, "u") is True

    def test_apikey_errada_e_recusada(self):
        assert canal().verify_signature(b"", {"apikey": "outra"}, "u") is False

    def test_sem_header_e_recusada(self):
        assert canal().verify_signature(b"", {}, "u") is False

    def test_sem_chave_configurada_recusa_em_vez_de_liberar(self):
        """Falha fechando: configuração incompleta não vira porta aberta."""
        c = canal(evolution_api_key="")
        assert c.verify_signature(b"", {"apikey": ""}, "u") is False

    def test_pode_ser_desligada_para_teste_local(self):
        c = canal(require_signature=False)
        assert c.verify_signature(b"", {}, "u") is True


class TestParse:
    def test_texto_simples(self):
        msgs = canal().parse_webhook(evento({"conversation": "qual a pega correta?"}))
        assert len(msgs) == 1
        assert msgs[0].kind == "text"
        assert msgs[0].text == "qual a pega correta?"
        assert msgs[0].sender == "5521999999999"
        assert msgs[0].message_id == "3EB0ABC"

    def test_texto_estendido(self):
        """Resposta a outra mensagem chega em extendedTextMessage."""
        msgs = canal().parse_webhook(
            evento({"extendedTextMessage": {"text": "e a posição?"}})
        )
        assert msgs[0].kind == "text"
        assert msgs[0].text == "e a posição?"

    def test_audio(self):
        msgs = canal().parse_webhook(
            evento({"audioMessage": {"mimetype": "audio/ogg; codecs=opus"}})
        )
        assert msgs[0].kind == "audio"
        assert msgs[0].media_id == "3EB0ABC"
        assert msgs[0].media_mime.startswith("audio/ogg")

    def test_imagem_vira_nao_suportada(self):
        msgs = canal().parse_webhook(evento({"imageMessage": {"mimetype": "image/jpeg"}}))
        assert msgs[0].kind == "unsupported"

    def test_eco_da_propria_resposta_e_ignorado(self):
        """Responder a si mesmo seria laço infinito."""
        assert canal().parse_webhook(evento({"conversation": "oi"}, from_me=True)) == []

    def test_mensagem_de_grupo_e_ignorada(self):
        """Bot em grupo é o caminho mais curto para denúncia."""
        msgs = canal().parse_webhook(
            evento({"conversation": "oi"}, jid="12345-67890@g.us")
        )
        assert msgs == []

    def test_jid_com_sufixo_de_dispositivo(self):
        msgs = canal().parse_webhook(
            evento({"conversation": "oi"}, jid="5521999999999:12@s.whatsapp.net")
        )
        assert msgs[0].sender == "5521999999999"

    def test_evento_que_nao_e_mensagem_e_ignorado(self):
        assert canal().parse_webhook({"event": "connection.update", "data": {}}) == []

    def test_data_em_lista(self):
        """Algumas versões mandam data como lista."""
        payload = evento({"conversation": "primeira"})
        payload["data"] = [payload["data"]]
        assert len(canal().parse_webhook(payload)) == 1

    def test_payload_vazio(self):
        assert canal().parse_webhook({}) == []

    def test_sem_id_ou_remetente_e_descartado(self):
        payload = evento({"conversation": "oi"})
        payload["data"]["key"]["id"] = ""
        assert canal().parse_webhook(payload) == []


class TestEnvio:
    def test_envia_com_apikey_e_numero(self, monkeypatch):
        chamadas = []

        def fake_post(url, **kwargs):
            chamadas.append((url, kwargs))
            return FakeResposta(200)

        monkeypatch.setattr("app.channels.evolution.post_with_retry", fake_post)
        assert canal().send_text("5521999999999", "resposta curta") is True

        url, kwargs = chamadas[0]
        assert url == "https://evo.exemplo.com/message/sendText/lactai"
        assert kwargs["headers"]["apikey"] == CHAVE
        assert kwargs["json"] == {"number": "5521999999999", "text": "resposta curta"}

    def test_resposta_longa_vira_varios_baloes_com_pausa(self, monkeypatch):
        """A pausa evita rajada, que é o padrão que os detectores procuram."""
        pausas = []
        monkeypatch.setattr(
            "app.channels.evolution.post_with_retry",
            lambda url, **kw: FakeResposta(200),
        )
        c = canal(evolution_send_delay_ms=900)
        assert c.send_text("5521999999999", "frase longa. " * 500,
                           sleep=pausas.append) is True

        assert pausas, "mensagem dividida deveria ter pausa entre os balões"
        assert all(p == 0.9 for p in pausas)

    def test_sem_divisao_nao_ha_pausa(self, monkeypatch):
        pausas = []
        monkeypatch.setattr(
            "app.channels.evolution.post_with_retry",
            lambda url, **kw: FakeResposta(200),
        )
        canal().send_text("5521999999999", "curta", sleep=pausas.append)
        assert pausas == []

    @pytest.mark.parametrize("status", [401, 404, 500])
    def test_erro_do_servidor_devolve_falso(self, monkeypatch, status):
        monkeypatch.setattr(
            "app.channels.evolution.post_with_retry",
            lambda url, **kw: FakeResposta(status, texto="erro"),
        )
        assert canal().send_text("5521999999999", "oi") is False

    def test_falha_de_rede_devolve_falso(self, monkeypatch):
        monkeypatch.setattr(
            "app.channels.evolution.post_with_retry", lambda url, **kw: None
        )
        assert canal().send_text("5521999999999", "oi") is False


class TestMidia:
    def test_base64_e_decodificado(self, monkeypatch):
        conteudo = b"audio-falso"
        monkeypatch.setattr(
            "app.channels.evolution.requests.post",
            lambda url, **kw: FakeResposta(
                200, {"base64": base64.b64encode(conteudo).decode()}
            ),
        )
        from app.channels.base import IncomingMessage

        msg = IncomingMessage(
            message_id="3EB0ABC", sender="5521999999999", kind="audio", media_id="3EB0ABC"
        )
        assert canal().fetch_media(msg) == conteudo

    def test_sem_media_id_nao_chama_a_api(self):
        from app.channels.base import IncomingMessage

        msg = IncomingMessage(message_id="1", sender="5521999999999", kind="audio")
        assert canal().fetch_media(msg) is None

    def test_resposta_sem_base64_devolve_none(self, monkeypatch):
        monkeypatch.setattr(
            "app.channels.evolution.requests.post",
            lambda url, **kw: FakeResposta(200, {}),
        )
        from app.channels.base import IncomingMessage

        msg = IncomingMessage(
            message_id="1", sender="5521999999999", kind="audio", media_id="1"
        )
        assert canal().fetch_media(msg) is None


class TestSelecaoDoProvedor:
    def test_variavel_de_ambiente_escolhe_o_canal(self):
        from app.channels import build_channel

        canal_escolhido = build_channel(Settings(
            provider="evolution", evolution_base_url="https://e",
            evolution_api_key="k", evolution_instance="i",
            openai_api_key="k", vector_store_id="vs",
        ))
        assert canal_escolhido.name == "evolution"

    def test_health_cobra_as_variaveis_certas(self):
        faltando = Settings(
            provider="evolution", openai_api_key="k", vector_store_id="vs"
        ).missing_channel()
        assert faltando == [
            "EVOLUTION_API_KEY", "EVOLUTION_BASE_URL", "EVOLUTION_INSTANCE"
        ]
