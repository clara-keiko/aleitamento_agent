"""Cifra dos campos sensíveis antes de irem para o banco.

O banco (e seus backups) guarda só texto cifrado. A chave fica na variável
CHAVE_DADOS do Render, nunca no banco: quem tiver uma cópia do banco sem a
chave não lê conversa, nome nem observação.

Fernet (AES-128-CBC + HMAC-SHA256): cifra autenticada, então um valor
adulterado no banco é recusado em vez de virar lixo na resposta.

⚠️ Perder CHAVE_DADOS torna ilegível tudo o que foi guardado. Trocar exige
recifrar (ainda não implementado).
"""

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken


class ChaveAusente(RuntimeError):
    pass


class Cofre:
    def __init__(self, chave: str):
        if not chave:
            raise ChaveAusente("CHAVE_DADOS vazia: memória persistente desligada")
        # Qualquer texto vira uma chave Fernet válida: o generateValue do
        # Render não sai no formato exato que a Fernet pede.
        derivada = hashlib.sha256(("lactai-dados:" + chave).encode("utf-8")).digest()
        self._fernet = Fernet(base64.urlsafe_b64encode(derivada))

    def cifrar(self, texto: str) -> bytes:
        return self._fernet.encrypt(texto.encode("utf-8"))

    def decifrar(self, dado: bytes) -> str:
        try:
            return self._fernet.decrypt(bytes(dado)).decode("utf-8")
        except InvalidToken as exc:
            raise ValueError("dado cifrado inválido ou chave errada") from exc
