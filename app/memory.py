"""Estado curto de conversa, deduplicação e rate limit.

Tudo em memória de processo, protegido por lock porque o processamento roda
em threadpool. Isso é suficiente para um piloto de uma instância só.

Limitação conhecida: com mais de uma réplica, ou a cada redeploy, o estado
se perde — a deduplicação deixa de valer entre instâncias e o usuário pode
receber a mensagem de boas-vindas de novo. Para produção com mais de uma
instância, trocar por Redis mantendo esta mesma interface.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field

# Quanto tempo lembramos que já demos boas-vindas a alguém. Precisa ser MUITO
# maior que o TTL da conversa: o histórico pode expirar em 30 min, mas
# reapresentar o serviço a cada meia hora é ruído.
GREETING_TTL_SECONDS = 30 * 24 * 3600


@dataclass
class _Conversation:
    turns: deque = field(default_factory=deque)
    last_seen: float = field(default_factory=time.monotonic)


class ConversationStore:
    """Histórico curto por usuário, com expiração."""

    def __init__(self, max_turns: int = 6, ttl_seconds: int = 1800):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._data: dict[str, _Conversation] = {}

    def _purge(self, now: float) -> None:
        stale = [
            key
            for key, conv in self._data.items()
            if now - conv.last_seen > self.ttl_seconds
        ]
        for key in stale:
            del self._data[key]

    def history(self, key: str) -> list[dict]:
        """Turnos anteriores no formato de mensagens da API."""
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            conversation = self._data.get(key)
            return list(conversation.turns) if conversation else []

    def append(self, key: str, role: str, content: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            conversation = self._data.setdefault(key, _Conversation())
            conversation.turns.append({"role": role, "content": content})
            while len(conversation.turns) > self.max_turns:
                conversation.turns.popleft()
            conversation.last_seen = now

    def forget(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)


class KnownUsers:
    """Quem já recebeu as boas-vindas.

    Separado do histórico de propósito: a conversa expira em minutos, a
    apresentação do serviço vale por semanas.
    """

    def __init__(self, ttl_seconds: int = GREETING_TTL_SECONDS, max_entries: int = 100_000):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._seen: dict[str, float] = {}

    def is_first_contact(self, key: str) -> bool:
        """True apenas na primeira vez; marca o usuário como conhecido."""
        now = time.monotonic()
        with self._lock:
            _purge_by_age(self._seen, now, self.ttl_seconds)

            if key in self._seen:
                self._seen[key] = now
                return False

            _evict_oldest_if_full(self._seen, self.max_entries)
            self._seen[key] = now
            return True

    def forget(self, key: str) -> None:
        with self._lock:
            self._seen.pop(key, None)


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
            _purge_by_age(self._seen, now, self.ttl_seconds)

            if message_id in self._seen:
                return False

            _evict_oldest_if_full(self._seen, self.max_entries)
            self._seen[message_id] = now
            return True


class RateLimiter:
    """Janela deslizante por usuário: contém abuso e custo de token."""

    def __init__(
        self,
        max_messages: int = 20,
        window_seconds: int = 600,
        max_tracked_users: int = 50_000,
    ):
        self.max_messages = max_messages
        self.window_seconds = window_seconds
        self.max_tracked_users = max_tracked_users
        self._lock = threading.Lock()
        self._hits: dict[str, deque] = {}

    def _drop_inactive(self, now: float) -> None:
        """Sem isso o dict cresce para sempre, um registro por telefone."""
        inactive = [
            key
            for key, hits in self._hits.items()
            if not hits or now - hits[-1] > self.window_seconds
        ]
        for key in inactive:
            del self._hits[key]

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            self._drop_inactive(now)

            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] > self.window_seconds:
                hits.popleft()

            if len(hits) >= self.max_messages:
                return False

            if len(self._hits) > self.max_tracked_users:
                # Rede de segurança: sob carga anômala, prefere esquecer o
                # mais antigo a deixar a memória crescer sem limite.
                oldest = min(self._hits, key=lambda k: self._hits[k][-1] if self._hits[k] else 0)
                if oldest != key:
                    del self._hits[oldest]

            hits.append(now)
            return True

    def forget(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)


def _purge_by_age(store: dict[str, float], now: float, ttl: float) -> None:
    expired = [key for key, ts in store.items() if now - ts > ttl]
    for key in expired:
        del store[key]


def _evict_oldest_if_full(store: dict[str, float], max_entries: int) -> None:
    while len(store) >= max_entries:
        oldest = min(store, key=store.get)
        del store[oldest]
