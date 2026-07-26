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

## Testes

```bash
python -m pytest
```

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
main.py               FastAPI: webhook, handshake, /health
app/config.py         variáveis de ambiente (nunca quebra no import)
app/guardrails.py     triagem clínica e mensagens fixas
app/llm.py            RAG e transcrição; checagem de fundamentação
app/pipeline.py       orquestra a mensagem até a resposta
app/memory.py         histórico curto, deduplicação, rate limit
app/channels/         Meta Cloud API e Twilio atrás da mesma interface
app/logging_utils.py  log sem dado pessoal
docs/                 base de conhecimento + OPERACAO.md
tests/                75 testes
```

## Privacidade

Dado de saúde de mãe e bebê é **sensível** pela LGPD. O serviço não grava telefone nem
conteúdo em log (pseudônimo HMAC), o histórico expira em 30 min e o comando `SAIR`
apaga a conversa. As pendências de conformidade estão listadas em
[docs/OPERACAO.md §4.2](docs/OPERACAO.md).
