# Roteiro de entrada no ar

Checklist executável para sair de "tenho CNPJ" e chegar em "o agente responde no
WhatsApp". Pesquisa de julho de 2026; itens marcados com ⚠️ mudam com frequência.

**A ideia central: são duas trilhas paralelas.** A burocracia da Meta leva dias de
espera, mas **não bloqueia nada do lado técnico** — o número de teste permite terminar
e validar a aplicação inteira hoje. Não fique esperando a verificação para começar.

```
Trilha A (burocracia)   ──── 3 a 10 dias úteis ────┐
                                                   ├──► convergência ──► piloto
Trilha B (técnica)      ──── dá para fazer hoje ───┘
```

---

## Trilha A — burocracia (comece hoje, depois é espera)

### A1. Portfólio empresarial

`business.facebook.com` → criar portfólio com os dados do CNPJ.

Use a **razão social** exata do cartão CNPJ. Se o portfólio já existe com um nome
"de uso interno", corrija agora — todo o resto é comparado contra este campo.

### A2. Verificação de negócio

**Configurações do negócio** → **Central de Segurança** → *Verificação da empresa* →
**Iniciar verificação**.

Documentos e causas de recusa: [`OPERACAO.md` §2.7](OPERACAO.md). O resumo: razão
social e endereço têm que bater **caractere por caractere** com o cartão CNPJ, e o
documento precisa ser o PDF original da Receita, não foto.

⚠️ 3 a 10 dias úteis.

### A3. Nome de exibição

Este item pega muita gente de surpresa: **todo número precisa de um nome de exibição
aprovado pela Meta antes de conseguir enviar qualquer mensagem.** Sem aprovação, a API
devolve o erro `#131037`.

Regras que causam recusa:

- Nome genérico ou que não se associa ao negócio verificado
- Nome que **não corresponde** à razão social ou ao site
- Termos restritos ("Oficial", "Global") e símbolos não suportados

Escolha algo claramente ligado ao CNPJ e ao material — por exemplo, o nome do projeto
seguido do da organização. E capriche: ⚠️ há limite de recursos de apelação, e ao
esgotá-lo o nome fica travado por **7 a 60 dias**.

### A4. Comprar o número

Compre **um** número (Twilio, Telnyx ou Salvy) e teste a verificação antes de contratar
plano anual. Comparação em [`OPERACAO.md` §2.2](OPERACAO.md).

Requisito que invalida tudo se ignorado: **o número não pode ter WhatsApp ativo**, nem
no app comum nem no Business. Se já teve, apague a conta antes e aguarde.

---

## Trilha B — técnica (não depende da Trilha A)

### B1. App e número de teste

1. `developers.facebook.com` → **Meus apps** → **Criar app**
2. Caso de uso: **Outro** → tipo **Empresa** (*Business*)
3. Vincule ao portfólio empresarial criado em A1
4. No painel do app → **Adicionar produto** → **WhatsApp** → **Configurar**

A tela *Início rápido da API* entrega tudo de uma vez. Anote:

| Onde aparece | O que é | Vai para |
|---|---|---|
| "Identificação do número de telefone" | número de teste | `PHONE_NUMBER_ID` |
| "Token de acesso temporário" | ⚠️ expira em 24 h | `WHATSAPP_TOKEN` |
| Configurações → Básico → *Chave secreta do app* | assina o webhook | `APP_SECRET` |

Ainda nessa tela, em **Para**, clique em *Gerenciar lista de números* e **cadastre o
seu celular** (até 5). Chega um código por WhatsApp para confirmar.

Os outros dois valores você inventa:

```bash
# Token do handshake do webhook — qualquer string, você repete no painel
openssl rand -hex 16
# Chave que pseudonimiza telefones no log
openssl rand -hex 32
```

### B1.5. Prove que o número funciona *antes* de subir código

Vale muito fazer isto primeiro: isola problema de credencial de problema de
aplicação. Se falhar aqui, não adianta debugar o servidor.

```bash
export TOKEN="EAAG..."        # token temporário
export PHONE_ID="123456789"   # PHONE_NUMBER_ID
export MEU_CEL="5511999999999"  # seu número, com 55, sem + e sem espaços

curl -X POST "https://graph.facebook.com/v25.0/$PHONE_ID/messages" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"messaging_product\":\"whatsapp\",\"to\":\"$MEU_CEL\",
       \"type\":\"template\",
       \"template\":{\"name\":\"hello_world\",\"language\":{\"code\":\"en_US\"}}}"
```

⚠️ **Repare que é `template`, não `text`.** Esta é a pegadinha que mais trava gente no
primeiro dia: fora de uma janela de 24 h aberta pelo usuário, **só template passa**.
Texto livre só funciona *depois* que a pessoa te mandou uma mensagem. Se você tentar
`type: text` de cara, recebe erro e parece que a credencial está errada — não está.

Chegou o "Hello World" no seu celular? Credenciais OK.

### B2. Indexar a base

```bash
cp .env.example .env
# preencha ao menos OPENAI_API_KEY
python ingest_openai_kb.py
```

Saída esperada: 17 arquivos, `Status do lote: completed`, `falharam: 0`, e o
`VECTOR_STORE_ID` no final.

Se algum arquivo falhar, o script **para com erro** e lista quais. Não ignore: base
incompleta vira "não encontrei isso no material" para a mãe, sem ninguém entender
por quê. O script é idempotente — rodar de novo não duplica nem cobra a mais.

### B3. Subir o serviço

Duas opções. **Túnel local** é melhor para o primeiro dia: você vê o log em tempo real
e itera em segundos, sem esperar redeploy.

<details>
<summary><b>Opção 1 — túnel local (iteração rápida)</b></summary>

```bash
pip install -r requirements-dev.txt
uvicorn main:app --reload --port 8000

# noutro terminal
cloudflared tunnel --url http://localhost:8000    # ou: ngrok http 8000
```

Use a URL pública que o túnel imprime no passo B4. Ela muda a cada reinício do túnel —
por isso não serve para produção.
</details>

<details>
<summary><b>Opção 2 — Render (o que vai para produção)</b></summary>

Novo *Web Service* apontando para o repositório. O `render.yaml` já traz build, start
e healthcheck. Plano **Starter**: o free hiberna após 15 min e perde o primeiro
webhook depois da pausa.

Variáveis a preencher no painel:

| Variável | Valor |
|---|---|
| `WHATSAPP_PROVIDER` | `meta` |
| `PHONE_NUMBER_ID` | da tela B1 |
| `WHATSAPP_TOKEN` | da tela B1 |
| `APP_SECRET` | Configurações → Básico |
| `VERIFY_TOKEN` | o `openssl rand -hex 16` |
| `PSEUDONYM_KEY` | gerado automaticamente pelo `render.yaml` |
| `OPENAI_API_KEY` | sua chave |
| `VECTOR_STORE_ID` | saída do B2 |
</details>

Confirme antes de seguir:

```bash
curl https://SUA-URL/health/ready
# {"status":"ok", ..., "missing_env":[]}
```

`503` não é falha de deploy: o campo `missing_env` lista exatamente o que falta.

### B4. Assinar o webhook

Painel do app → **WhatsApp** → **Configuração** → seção *Webhook* → **Editar**:

- **URL de retorno de chamada:** `https://SUA-URL/webhook`
- **Token de verificação:** o mesmo valor de `VERIFY_TOKEN`
- **Verificar e salvar** — fica verde na hora, ou não salva

Se recusar, teste o handshake você mesmo e compare:

```bash
curl "https://SUA-URL/webhook?hub.mode=subscribe\
&hub.verify_token=SEU_VERIFY_TOKEN&hub.challenge=12345"
# tem que responder exatamente: 12345
```

Depois de salvar, em **Campos do webhook** clique em *Gerenciar* e **assine
`messages`**.

⚠️ Este é o erro silencioso mais comum de todos: webhook verificado, tudo verde, e
nenhuma mensagem chega — porque o campo `messages` não foi assinado. Não existe aviso.

### B5. Testar de ponta a ponta

Cadastre o seu próprio celular na lista de até 5 destinatários de teste e mande
mensagens reais. O que precisa acontecer:

- [ ] Primeira mensagem → boas-vindas com escopo, aviso de automação e o 192
- [ ] `"qual a melhor posição para amamentar?"` → resposta vinda do material
- [ ] `"meu bebe nao respira"` → emergência imediata, sem passar pelo modelo
- [ ] `"meu bebê está com febre desde ontem"` → encaminhamento
- [ ] `"posso amamentar com febre?"` → resposta educativa **com** nota de segurança
- [ ] Um **áudio** → transcreve e responde
- [ ] `"obrigada"` → resposta social curta, não "não encontrei no material"
- [ ] `"qual a capital da França?"` → recusa fora de escopo
- [ ] `SAIR` → confirma que apagou o histórico
- [ ] Nos logs: telefone aparece como `u_xxxxxxxx`, nunca em claro

**Importante:** aqui você manda mensagem *primeiro*, o que abre a janela de 24 h. Por
isso o agente responde com texto livre normalmente — não precisa de template.

Cada mensagem gera uma linha de log assim:

```
INFO app.pipeline processado user=u_09dfcccad8c3 outcome=respondido kind=text duracao_ms=2841
```

O campo `outcome` é o seu painel de controle:

| `outcome` | Significa |
|---|---|
| `respondido` | Resposta veio da base, com citação |
| `respondido_com_nota` | Tema sensível: respondeu e anexou o alerta |
| `emergencia` / `encaminhamento` | Triagem clínica agiu, modelo nem foi chamado |
| `social` | Saudação ou agradecimento |
| `fora_de_escopo` | Modelo não citou a base — recusou |
| `erro_modelo` | Timeout ou falha na OpenAI |
| `duplicada` | Reentrega da Meta, corretamente ignorada |

#### Se algo não funcionar

| Sintoma | Causa provável |
|---|---|
| Nada acontece, log vazio | Campo `messages` não assinado (B4) |
| `403` no log do servidor | `APP_SECRET` diferente do painel |
| Recebe, mas não responde | Token expirado (24 h) — gere outro |
| Tudo vira `fora_de_escopo` | `VECTOR_STORE_ID` errado ou base vazia |
| Resposta duplicada | Não deveria ocorrer; abra issue com o log |
| `duracao_ms` > 10000 | Reduza `MAX_RETRIEVAL_RESULTS` |

### B6. Escolher o modelo com dado

```bash
python evals/run_eval.py --live --model gpt-4o-mini
python evals/run_eval.py --live --model gpt-5-mini
```

Compare fundamentação, latência p95 e custo. Análise em
[`OPERACAO.md` §6](OPERACAO.md).

### B7. Revisão clínica — o item que não dá para pular

Sentar com uma **consultora de amamentação** e revisar `evals/golden_set.yaml` caso a
caso, aprovando as respostas que o agente dá hoje.

Enquanto isso não acontecer, o eval mede consistência, não correção. **É o único item
desta lista que nenhuma quantidade de código substitui.**

---

## Convergência — quando A e B terminarem

### C1. Registrar o número real

No painel → **WhatsApp** → **Números de telefone** → **Adicionar número**. Verificação
por SMS ou, em número fixo/digital, por **chamada de voz** ("Me ligue").

### C2. Token permanente

**Configurações do negócio** → **Usuários do sistema** → criar um usuário do sistema →
atribuir o app com permissão `whatsapp_business_messaging` → **Gerar token**.

Guarde na hora: ele só aparece uma vez.

### C3. Trocar para produção

No Render, atualize `PHONE_NUMBER_ID` e `WHATSAPP_TOKEN` para os do número real.
Redeploy, e repita o checklist B5 inteiro — agora valendo.

### C4. Piloto controlado

Comece com 20 ou 30 mães, não 300. Acompanhe nos logs:

| Métrica | O que observar |
|---|---|
| `outcome=fora_de_escopo` | Alto em perguntas que *estão* no material → problema de recuperação, não de modelo |
| `outcome=emergencia` / `encaminhamento` | Confira uma a uma no começo |
| `duracao_ms` | Acima de ~5 s a mãe acha que travou |
| `outcome=erro_modelo` | Timeout ou cota |

---

## Armadilhas que custam dias

| Armadilha | Consequência |
|---|---|
| Número já tem WhatsApp ativo | Registro falha e não há mensagem clara dizendo por quê |
| Nome de exibição não aprovado | Erro `#131037`, nenhuma mensagem sai |
| Esgotar apelações do nome | Nome travado por 7 a 60 dias |
| Usar o token de 24 h em produção | Para de funcionar no dia seguinte |
| Esquecer de assinar o campo `messages` | Webhook verificado, verde, e nada chega |
| Testar no número de produção | Comportamento suspeito pode bloquear o número |
| Render no plano free | Primeiro webhook após 15 min ocioso se perde |
| Nome fantasia no lugar da razão social | Verificação recusada |

---

## Ordem recomendada da primeira semana

| Dia | Trilha A | Trilha B |
|---|---|---|
| 1 | A1, A2 (envia e espera), A4 (compra e testa) | B1, B2, B3, B4 |
| 1–2 | — | B5, B6 |
| 2–5 | A3 (nome de exibição) | B7 (agendar a consultora) |
| 5–10 | aguardar verificação | ajustes do que B5/B6 revelarem |
| Depois | C1 → C4 | |
