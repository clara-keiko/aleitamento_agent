"""Estado curto de conversa, deduplicação e rate limit.

Tudo em memória de processo, protegido por lock porque o processamento roda
em threadpool. Isso é suficiente para um piloto de uma instância só.

Limitação conhecida: com mais de uma réplica, ou a cada redeploy, o estado
se perde — a deduplicação deixa de valer entre instâncias e o usuário pode
receber a mensagem de boas-vindas de novo. Para produção com mais de uma
instância, trocar por Redis mantendo esta mesma interface.
"""

from collections import deque
from dataclasses import dataclass, field
import threading
import time


@dataclass
class _Conversation:
    turns: deque = field(default_factory=deque)
    last_seen: float = field(default_factory=time.monotonic)
    greeted: bool = False


class ConversationStore:
    def __init__(self, max_turns: int = 6, ttl_seconds: int = 1800):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._data: dict[str, _Conversation] = {}

    def _expired(self, conversation: _Conversation, now: float) -> bool:
        return now - conversation.last_seen > self.ttl_seconds

    def _purge(self, now: float) -> None:
        stale = [key for key, conv in self._data.items() if self._expired(conv, now)]
        for key in stale:
            del self._data[key]

    def history(self, key: str) -> list[dict]:
        """Turnos anteriores no formato de mensagens da API."""
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            conversation = self._data.get(key)
            if not conversation:
                return []
            return list(conversation.turns)

    def append(self, key: str, role: str, content: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            conversation = self._data.setdefault(key, _Conversation())
            conversation.turns.append({"role": role, "content": content})
            while len(conversation.turns) > self.max_turns:
                conversation.turns.popleft()
            conversation.last_seen = now

    def mark_greeted(self, key: str) -> bool:
        """Marca o usuário como já saudado. Retorna True se era o 1º contato."""
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            conversation = self._data.setdefault(key, _Conversation())
            conversation.last_seen = now
            if conversation.greeted:
                return False
            conversation.greeted = True
            return True

    def forget(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)


class MessageDeduplicator:
    """Evita responder duas vezes o mesmo `message_id`.

    A Meta reentrega o webhook quando não recebe 200 rápido. Sem isso, uma
    resposta lenta da OpenAI vira duas ou três mensagens iguais para a mãe.
    """

    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 10_000):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._seen: dict[str, float] = {}

    def check_and_mark(self, message_id: str) -> bool:
        """True se é a primeira vez que vemos esta mensagem."""
        if not message_id:
            return True

        now = time.monotonic()
        with self._lock:
            expired = [key for key, ts in self._seen.items() if now - ts > self.ttl_seconds]
            for key in expired:
                del self._seen[key]

            if message_id in self._seen:
                return False

            if len(self._seen) >= self.max_entries:
                oldest = min(self._seen, key=self._seen.get)
                del self._seen[oldest]

            self._seen[message_id] = now
            return True


class RateLimiter:
    """Janela deslizante por usuário: contém abuso e custo de token."""

    def __init__(self, max_messages: int = 20, window_seconds: int = 600):
        self.max_messages = max_messages
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._hits: dict[str, deque] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] > self.window_seconds:
                hits.popleft()

            if not hits:
                # Evita crescer o dict indefinidamente com usuários inativos.
                self._hits.pop(key, None)
                self._hits[key] = hits

            if len(hits) >= self.max_messages:
                return False

            hits.append(now)
            return True
