# Agente de Aleitamento

Assistente educativo de **amamentação e cuidados com o bebê** no WhatsApp. Responde
apenas com base em material curado (Ministério da Saúde e literatura de aleitamento),
com triagem clínica antes do modelo e recusa explícita fora do escopo.

> **Não faz diagnóstico, não prescreve e não substitui avaliação profissional.**
> Sinais de emergência interrompem o fluxo e encaminham para o 192.

📄 **[docs/OPERACAO.md](docs/OPERACAO.md)** — como conseguir um número sem chip,
quanto custa operar e o roadmap. Leia antes de colocar no ar.

---

## Como funciona

```
WhatsApp ──► /webhook ──► assinatura ──► 200 imediato
                                             │
                                    (background)
                                             ▼
                      dedup ──► rate limit ──► áudio→texto
                                             │
                                    triagem clínica
                        ┌────────────────────┼────────────────────┐
                    emergência          sinal de alerta      educativo
                     (192)              (procurar UBS)            │
                                                        RAG na base curada
                                                                  │
                                                       cita a base? ──não──► fora de escopo
                                                                  │sim
                                                              resposta
```

A triagem roda **antes** do modelo de propósito: uma emergência não pode depender de a
OpenAI estar no ar.

## Rodando local

```bash
pip install -r requirements-dev.txt
cp .env.example .env      # preencha as chaves
python ingest_openai_kb.py   # indexa docs/ e imprime o VECTOR_STORE_ID
uvicorn main:app --reload
```

Sem `.env` completo o serviço **sobe assim mesmo** e o `/health` diz o que falta:

```bash
curl localhost:8000/health
# {"status":"degraded", ..., "missing_env":["APP_SECRET","OPENAI_API_KEY"]}
```

Para testar o webhook local sem `APP_SECRET`, use `REQUIRE_SIGNATURE=false`
— **apenas em desenvolvimento**.

## Testes e avaliação

```bash
python -m pytest          # 120 testes
ruff check .              # lint
python evals/run_eval.py  # triagem clínica contra o conjunto dourado
```

O eval é a peça que responde *"o agente ainda está seguro?"* e *"trocar de modelo
melhora?"*. Ele roda no CI e **falha o build se um caso de emergência deixar de ser
reconhecido**.

Para comparar modelos de verdade (chama a API, custa alguns centavos):

```bash
python evals/run_eval.py --live --model gpt-4o-mini
python evals/run_eval.py --live --model gpt-5-mini
```

Ele imprime taxa de acerto por categoria, latência p50/p95 e custo estimado por mês.
Ver [docs/OPERACAO.md §6](docs/OPERACAO.md) para a análise da escolha de modelo.

⚠️ O conjunto dourado (`evals/golden_set.yaml`) foi escrito para ser editado por quem
entende de amamentação. **Antes do piloto, uma consultora precisa revisar cada caso** —
hoje o eval mede consistência, não correção clínica.

## Configuração

Todas as variáveis estão documentadas em [`.env.example`](.env.example). As essenciais:

| Variável | Para quê |
|---|---|
| `WHATSAPP_PROVIDER` | `meta` (padrão) ou `twilio` |
| `VERIFY_TOKEN` | Handshake do webhook; você inventa e repete no painel da Meta |
| `WHATSAPP_TOKEN` | Token de acesso — gere um permanente via System User |
| `PHONE_NUMBER_ID` | ID do número no painel (não é o telefone) |
| `APP_SECRET` | Assina o webhook. **Obrigatório em produção** |
| `OPENAI_API_KEY` | Chave da OpenAI |
| `VECTOR_STORE_ID` | Saída do `ingest_openai_kb.py` |
| `PSEUDONYM_KEY` | Pseudonimiza telefones no log (`openssl rand -hex 32`) |

## Deploy

`render.yaml` e `Dockerfile` estão prontos. Depois do deploy, registre o webhook no
painel da Meta apontando para `https://SEU-APP/webhook` com o mesmo `VERIFY_TOKEN` e
assine o campo `messages`.

⚠️ Use o plano **Starter**, não o free: o free hiberna após 15 min e o primeiro webhook
depois da pausa se perde no cold start.

## Estrutura

```
main.py               FastAPI: webhook, handshake, health
app/config.py         variáveis de ambiente (nunca quebra no import)
app/guardrails.py     triagem clínica, small talk e mensagens fixas
app/llm.py            RAG e transcrição; checagem de fundamentação
app/pipeline.py       orquestra a mensagem até a resposta
app/memory.py         histórico, usuários conhecidos, dedup, rate limit
app/http.py           POST com retry e backoff
app/channels/         Meta Cloud API e Twilio atrás da mesma interface
app/logging_utils.py  log sem dado pessoal
evals/                conjunto dourado + runner de avaliação
docs/                 base de conhecimento + OPERACAO.md
tests/                120 testes
```

Dois endpoints de saúde, com papéis diferentes:

- `GET /health` — **liveness**, sempre 200 se o processo está de pé. É o que a
  plataforma monitora. Não devolve 503 por configuração incompleta de propósito:
  se devolvesse, o deploy nunca subiria e ninguém leria o log para descobrir o que falta.
- `GET /health/ready` — **readiness**, 503 enquanto faltar variável.

## Privacidade

Dado de saúde de mãe e bebê é **sensível** pela LGPD. O serviço não grava telefone nem
conteúdo em log (pseudônimo HMAC), o histórico expira em 30 min e o comando `SAIR`
apaga a conversa. As pendências de conformidade estão listadas em
[docs/OPERACAO.md §4.2](docs/OPERACAO.md).
