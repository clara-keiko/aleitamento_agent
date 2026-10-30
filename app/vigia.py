"""Religa o WhatsApp quando a sessão cai por reinício do servidor.

No canal não oficial, o número ficava mudo depois de cada reinício da
Evolution até alguém rodar o QR de novo — e ninguém percebia antes do alerta.
O vigia confere a conexão de tempos em tempos e, se ela estiver fechada, pede
para religar com a sessão salva.

Ele não resolve sessão desfeita no celular (aparelho desconectado, número
banido): aí a Evolution devolve um QR, que só um humano pode escanear. Nesse
caso o vigia espera mais antes de tentar de novo, para não gerar QR sem
parar, e o /health/ready segue em 503 para o monitor externo avisar.
"""

import threading
import time
from collections.abc import Callable

from app.logging_utils import get_logger

log = get_logger(__name__)


class VigiaDaConexao:
    def __init__(
        self,
        channel,
        intervalo_s: float = 120,
        espera_apos_qr_s: float = 1800,
        agora: Callable[[], float] = time.monotonic,
    ):
        self.channel = channel
        self.intervalo_s = intervalo_s
        self.espera_apos_qr_s = espera_apos_qr_s
        self._agora = agora
        self._proxima_tentativa = 0.0
        self._parar = threading.Event()

    def verificar(self) -> str:
        """Uma rodada. Devolve o que fez, para log e teste."""
        estado = self.channel.estado_da_conexao()
        # "connecting" já está tentando; "desconhecido" é a Evolution fora do
        # ar, e pedir reconexão a ela não adianta.
        if estado != "close":
            return "nada"
        if self._agora() < self._proxima_tentativa:
            return "aguardando"

        resultado = self.channel.reconectar()
        if resultado == "precisa_qr":
            self._proxima_tentativa = self._agora() + self.espera_apos_qr_s
            log.error(
                "WhatsApp desconectado e a sessão não vale mais: é preciso parear "
                "pelo QR (scripts/evolution_setup.py qr)"
            )
        else:
            log.warning("WhatsApp estava desconectado; reconexão pedida (%s)", resultado)
        return resultado

    def iniciar(self) -> None:
        threading.Thread(target=self._rodar, name="vigia-whatsapp", daemon=True).start()

    def parar(self) -> None:
        self._parar.set()

    def _rodar(self) -> None:
        # Espera a primeira volta: logo após o deploy a Evolution pode estar
        # subindo, e é ela mesma quem tenta primeiro.
        while not self._parar.wait(self.intervalo_s):
            try:
                self.verificar()
            except Exception:  # noqa: BLE001 - o vigia não pode morrer
                log.exception("falha no vigia da conexão")
