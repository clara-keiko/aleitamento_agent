import requests

from app import http


class FakeResponse:
    def __init__(self, status_code: int, headers: dict | None = None):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = ""


def run(responses, attempts=3):
    """Executa post_with_retry com uma sequência de respostas/exceções."""
    chamadas = {"n": 0}
    dormidas: list[float] = []

    def fake_post(url, **kwargs):
        indice = chamadas["n"]
        chamadas["n"] += 1
        resultado = responses[min(indice, len(responses) - 1)]
        if isinstance(resultado, Exception):
            raise resultado
        return resultado

    original = http.requests.post
    http.requests.post = fake_post
    try:
        resposta = http.post_with_retry(
            "https://exemplo.com", attempts=attempts, sleep=dormidas.append
        )
    finally:
        http.requests.post = original

    return resposta, chamadas["n"], dormidas


class TestPostWithRetry:
    def test_sucesso_na_primeira_nao_repete(self):
        resposta, tentativas, dormidas = run([FakeResponse(200)])
        assert resposta.status_code == 200
        assert tentativas == 1
        assert dormidas == []

    def test_erro_de_cliente_nao_e_repetido(self):
        """400 é configuração errada; insistir só atrasa o diagnóstico."""
        resposta, tentativas, _ = run([FakeResponse(400)])
        assert resposta.status_code == 400
        assert tentativas == 1

    def test_401_nao_e_repetido(self):
        _, tentativas, _ = run([FakeResponse(401)])
        assert tentativas == 1

    def test_429_e_repetido(self):
        resposta, tentativas, dormidas = run([FakeResponse(429), FakeResponse(200)])
        assert resposta.status_code == 200
        assert tentativas == 2
        assert len(dormidas) == 1

    def test_500_e_repetido_ate_o_limite(self):
        resposta, tentativas, _ = run([FakeResponse(503)], attempts=3)
        assert resposta.status_code == 503
        assert tentativas == 3

    def test_falha_de_rede_e_repetida(self):
        resposta, tentativas, _ = run(
            [requests.ConnectionError("sem rede"), FakeResponse(200)]
        )
        assert resposta.status_code == 200
        assert tentativas == 2

    def test_falha_de_rede_em_todas_devolve_none(self):
        resposta, tentativas, _ = run([requests.Timeout("estourou")], attempts=3)
        assert resposta is None
        assert tentativas == 3

    def test_respeita_retry_after(self):
        _, _, dormidas = run(
            [FakeResponse(429, {"Retry-After": "2"}), FakeResponse(200)]
        )
        assert dormidas == [2.0]

    def test_retry_after_invalido_cai_no_backoff(self):
        _, _, dormidas = run(
            [FakeResponse(429, {"Retry-After": "daqui a pouco"}), FakeResponse(200)]
        )
        assert len(dormidas) == 1
        assert 0 < dormidas[0] <= http.MAX_DELAY_SECONDS

    def test_backoff_cresce(self):
        _, _, dormidas = run([FakeResponse(500)], attempts=4)
        assert len(dormidas) == 3
        assert dormidas[-1] > dormidas[0]
