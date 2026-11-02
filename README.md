# Agente de Aleitamento

Assistente educativo de **amamentação e cuidados com o bebê** no WhatsApp. Responde
apenas com base em material curado (Ministério da Saúde e literatura de aleitamento),
com triagem clínica antes do modelo e recusa explícita fora do escopo.

> **Não faz diagnóstico, não prescreve e não substitui avaliação profissional.**
> Sinais de emergência interrompem o fluxo e encaminham para o 192.

🎬 **[docs/DEMO.md](docs/DEMO.md)** — roteiro de apresentação em 5 minutos, com plano B.
Antes de apresentar, rode `python scripts/preflight.py` (veredito GO / NO-GO).

📋 **[docs/GO_LIVE.md](docs/GO_LIVE.md)** — checklist para colocar no ar, das duas
trilhas paralelas (burocracia da Meta × técnica) até o piloto.

📄 **[docs/OPERACAO.md](docs/OPERACAO.md)** — número sem chip, custos, verificação de
CNPJ, política de IA da Meta e escolha de modelo.

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

## Protótipo no navegador

O agente roda em `/chat` **sem precisar de WhatsApp** — sem app na Meta, sem número,
sem verificação. Mesmo pipeline, mesma triagem, mesma base: o que for validado aqui é
o que a mãe recebe lá.

**Requer Python 3.11 ou maior.** ⚠️ O macOS vem de fábrica com o 3.9, que **não roda
este projeto** — confira com `python3 --version` antes de começar. Se for 3.9 ou 3.10,
instale o 3.11 (`brew install python@3.11` ou o instalador de python.org) e use o
ambiente virtual abaixo.

```bash
python3.11 -m venv .venv     # use 3.11 explicitamente
source .venv/bin/activate    # o prompt passa a mostrar (.venv)

pip install -r requirements-dev.txt
cp .env.example .env         # preencha OPENAI_API_KEY
python ingest_openai_kb.py   # indexa docs/ e imprime o VECTOR_STORE_ID
uvicorn main:app --reload
# http://localhost:8000/chat
```

Dentro do ambiente virtual, `python` e `pip` já apontam para o 3.11 — não precisa
escrever `python3.11` nem `pip3`. A cada novo terminal, rode `source .venv/bin/activate`
de novo.

O botão **inspeção** mostra qual camada respondeu cada mensagem — é o que distingue
"a resposta está errada" de "a triagem interceptou".

⚠️ Antes de publicar o link, defina `WEB_ACCESS_CODE` (`openssl rand -hex 8`). Sem ele,
qualquer pessoa com a URL gasta sua cota da OpenAI.

## Rodando local

Para o WhatsApp, siga [docs/GO_LIVE.md](docs/GO_LIVE.md).

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

Para comparar modelos (chama a API, custa alguns centavos):

```bash
python evals/run_eval.py --comparar gpt-4o-mini,gpt-5-mini --relatorio comparacao.md
```

Imprime tabela lado a lado com acerto por categoria, taxa de fundamentação, latência
p50/p95 e custo — calculado com o **consumo real de tokens**, não estimativa. O
`--relatorio` grava as respostas dos modelos lado a lado para revisão humana.

Quando a taxa de fundamentação satura (≥95%), número para de separar os modelos. Aí
entra a revisão cega — respostas anonimizadas, ordem balanceada, julgadas por quem
entende de amamentação:

```bash
python evals/run_eval.py --comparar gpt-4o-mini,gpt-5-mini --relatorio-cego revisao.md
# a consultora preenche revisao.md
python evals/apurar.py revisao.md
```

Preços em [`evals/precos.yaml`](evals/precos.yaml), editável sem mexer em código.
Análise em [docs/OPERACAO.md §6](docs/OPERACAO.md).

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
app/channels/         Meta Cloud API, Twilio e web atrás da mesma interface
app/static/chat.html  protótipo de chat no navegador
app/logging_utils.py  log sem dado pessoal
evals/                conjunto dourado + runner de avaliação
docs/                 base de conhecimento + OPERACAO.md
tests/                120 testes
```

Endpoints:

| Rota | Para quê |
|---|---|
| `GET /chat` | Protótipo de chat no navegador |
| `POST /api/chat` | API do protótipo (`{session, text}` → `{replies, outcome}`) |
| `GET/POST /webhook` | WhatsApp (handshake e mensagens) |
| `GET /health` | Liveness |
| `GET /health/ready` | Readiness |

Os dois endpoints de saúde têm papéis diferentes:

- `GET /health` — **liveness**, sempre 200 se o processo está de pé. É o que a
  plataforma monitora. Não devolve 503 por configuração incompleta de propósito:
  se devolvesse, o deploy nunca subiria e ninguém leria o log para descobrir o que falta.
- `GET /health/ready` — **readiness**, 503 enquanto faltar variável.

## Privacidade

Dado de saúde de mãe e bebê é **sensível** pela LGPD. O serviço não grava telefone nem
conteúdo em log (pseudônimo HMAC), o histórico expira em 30 min e o comando `SAIR`
apaga a conversa. As pendências de conformidade estão listadas em
[docs/OPERACAO.md §4.2](docs/OPERACAO.md).
