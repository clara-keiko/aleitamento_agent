from app.memory import ConversationStore, KnownUsers, MessageDeduplicator, RateLimiter


class TestConversationStore:
    def test_guarda_e_devolve_turnos(self):
        store = ConversationStore(max_turns=4)
        store.append("u1", "user", "oi")
        store.append("u1", "assistant", "olá")

        assert store.history("u1") == [
            {"role": "user", "content": "oi"},
            {"role": "assistant", "content": "olá"},
        ]

    def test_respeita_o_limite_de_turnos(self):
        store = ConversationStore(max_turns=2)
        for i in range(6):
            store.append("u1", "user", f"msg{i}")

        historico = store.history("u1")
        assert len(historico) == 2
        assert historico[-1]["content"] == "msg5"

    def test_usuarios_sao_isolados(self):
        store = ConversationStore()
        store.append("u1", "user", "segredo do u1")
        assert store.history("u2") == []

    def test_expira_por_ttl(self):
        store = ConversationStore(ttl_seconds=0)
        store.append("u1", "user", "oi")
        assert store.history("u1") == []

    def test_forget_apaga(self):
        store = ConversationStore()
        store.append("u1", "user", "oi")
        store.forget("u1")
        assert store.history("u1") == []


class TestKnownUsers:
    def test_so_o_primeiro_contato_retorna_true(self):
        known = KnownUsers()
        assert known.is_first_contact("u1") is True
        assert known.is_first_contact("u1") is False
        assert known.is_first_contact("u1") is False

    def test_usuarios_diferentes_sao_independentes(self):
        known = KnownUsers()
        assert known.is_first_contact("u1") is True
        assert known.is_first_contact("u2") is True

    def test_ttl_longo_sobrevive_a_expiracao_da_conversa(self):
        """A conversa expira em 30 min; a apresentação do serviço não."""
        known = KnownUsers(ttl_seconds=30 * 24 * 3600)
        known.is_first_contact("u1")
        assert known.is_first_contact("u1") is False

    def test_forget_permite_nova_saudacao(self):
        known = KnownUsers()
        known.is_first_contact("u1")
        known.forget("u1")
        assert known.is_first_contact("u1") is True

    def test_nao_cresce_alem_do_limite(self):
        known = KnownUsers(max_entries=10)
        for i in range(50):
            known.is_first_contact(f"u{i}")
        assert len(known._seen) <= 10


class TestMessageDeduplicator:
    def test_primeira_vez_passa_e_segunda_nao(self):
        dedup = MessageDeduplicator()
        assert dedup.check_and_mark("wamid.1") is True
        assert dedup.check_and_mark("wamid.1") is False

    def test_ids_diferentes_passam(self):
        dedup = MessageDeduplicator()
        assert dedup.check_and_mark("wamid.1") is True
        assert dedup.check_and_mark("wamid.2") is True

    def test_id_vazio_sempre_passa(self):
        dedup = MessageDeduplicator()
        assert dedup.check_and_mark("") is True
        assert dedup.check_and_mark("") is True

    def test_nao_cresce_alem_do_limite(self):
        dedup = MessageDeduplicator(max_entries=10)
        for i in range(100):
            dedup.check_and_mark(f"wamid.{i}")
        assert len(dedup._seen) <= 10


class TestRateLimiter:
    def test_libera_ate_o_limite(self):
        limiter = RateLimiter(max_messages=3, window_seconds=600)
        assert [limiter.allow("u1") for _ in range(5)] == [True, True, True, False, False]

    def test_usuarios_sao_independentes(self):
        limiter = RateLimiter(max_messages=1)
        assert limiter.allow("u1") is True
        assert limiter.allow("u2") is True
        assert limiter.allow("u1") is False

    def test_janela_expirada_libera_de_novo(self):
        limiter = RateLimiter(max_messages=1, window_seconds=0)
        assert limiter.allow("u1") is True
        assert limiter.allow("u1") is True

    def test_nao_acumula_usuarios_inativos(self):
        """Sem a limpeza, o dict crescia um registro por telefone, para sempre."""
        limiter = RateLimiter(max_messages=5, window_seconds=0)
        for i in range(200):
            limiter.allow(f"u{i}")
        assert len(limiter._hits) <= 2

    def test_forget_zera_o_contador(self):
        limiter = RateLimiter(max_messages=1)
        limiter.allow("u1")
        assert limiter.allow("u1") is False
        limiter.forget("u1")
        assert limiter.allow("u1") is True
