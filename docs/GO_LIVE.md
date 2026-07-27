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

`developers.facebook.com` → **Criar app** → tipo **Business** → adicionar o produto
**WhatsApp**.

A Meta entrega na hora um número de teste e o `PHONE_NUMBER_ID`. Anote também o
**App Secret** (Configurações → Básico) — é ele que assina o webhook.

⚠️ O token que aparece nessa tela **expira em 24 h**. Serve para hoje; o permanente
vem no passo C2.

### B2. Indexar a base

```bash
cp .env.example .env      # preencha OPENAI_API_KEY
python ingest_openai_kb.py
```

Guarde o `VECTOR_STORE_ID` impresso no final. O script é idempotente: rodar de novo
não duplica nem cobra a mais.

### B3. Deploy

Render, plano **Starter** (o free hiberna e perde o primeiro webhook). Preencha as
variáveis conforme o `render.yaml`.

Confirme que subiu:

```bash
curl https://SEU-APP.onrender.com/health/ready
# {"status":"ok", ..., "missing_env":[]}
```

Se vier `503`, o `missing_env` diz exatamente o que falta.

### B4. Assinar o webhook

No painel do app → **WhatsApp** → **Configuração**:

- **URL de callback:** `https://SEU-APP.onrender.com/webhook`
- **Token de verificação:** o mesmo valor de `VERIFY_TOKEN`
- Clique em **Verificar e salvar** (deve ficar verde na hora)
- Em *Campos do webhook*, assine **`messages`** — sem isso nada chega

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
