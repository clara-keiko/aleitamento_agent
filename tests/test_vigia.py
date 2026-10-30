"""Religar o WhatsApp sozinho depois de um reinício da Evolution.

Antes, cada reinício do servidor (deploy, sincronização do Blueprint)
deixava o número mudo até alguém escanear o QR de novo.
"""

import requests

from app.vigia import VigiaDaConexao
from tests.test_channels_evolution import FakeResposta, canal


class CanalFalso:
    def __init__(self, estado="open", resultado="reconectando"):
        self.estado = estado
        self.resultado = resultado
        self.pedidos = 0

    def estado_da_conexao(self):
        return self.estado

    def reconectar(self):
        self.pedidos += 1
        return self.resultado


class Relogio:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class TestVigia:
    def test_conectado_nao_faz_nada(self):
        c = CanalFalso("open")
        assert VigiaDaConexao(c).verificar() == "nada"
        assert c.pedidos == 0

    def test_fechado_pede_reconexao(self):
        c = CanalFalso("close")
        assert VigiaDaConexao(c).verificar() == "reconectando"
        assert c.pedidos == 1

    def test_ja_conectando_ou_evolution_fora_nao_insiste(self):
        for estado in ("connecting", "desconhecido"):
            c = CanalFalso(estado)
            VigiaDaConexao(c).verificar()
            assert c.pedidos == 0

    def test_sem_sessao_espera_antes_de_gerar_outro_qr(self):
        c = CanalFalso("close", resultado="precisa_qr")
        relogio = Relogio()
        vigia = VigiaDaConexao(c, espera_apos_qr_s=1800, agora=relogio)

        assert vigia.verificar() == "precisa_qr"
        relogio.t = 600
        assert vigia.verificar() == "aguardando"
        relogio.t = 1801
        assert vigia.verificar() == "precisa_qr"
        assert c.pedidos == 2

    def test_reconexao_comum_tenta_de_novo_na_proxima_rodada(self):
        c = CanalFalso("close", resultado="falhou")
        vigia = VigiaDaConexao(c)
        vigia.verificar()
        vigia.verificar()
        assert c.pedidos == 2


class TestReconectarNaEvolution:
    def test_sessao_salva_religa_sem_qr(self, monkeypatch):
        chamadas = []

        def get(url, **kw):
            chamadas.append((url, kw["headers"]["apikey"]))
            return FakeResposta(200, {"pairingCode": None, "code": None, "count": 0})

        monkeypatch.setattr("app.channels.evolution.requests.get", get)
        assert canal().reconectar() == "reconectando"
        assert chamadas == [("https://evo.exemplo.com/instance/connect/lactai", "chave-da-evolution")]

    def test_sessao_desfeita_devolve_qr(self, monkeypatch):
        monkeypatch.setattr(
            "app.channels.evolution.requests.get",
            lambda url, **kw: FakeResposta(200, {"base64": "data:image/png;base64,AAA"}),
        )
        assert canal().reconectar() == "precisa_qr"

    def test_falha_de_rede(self, monkeypatch):
        def explode(url, **kw):
            raise requests.ConnectionError("fora do ar")

        monkeypatch.setattr("app.channels.evolution.requests.get", explode)
        assert canal().reconectar() == "falhou"
