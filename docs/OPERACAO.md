# Operação: número, custo e evolução do agente

Documento de decisão para colocar o agente no ar. Pesquisa feita em **julho de 2026**;
preço e política de plataforma mudam rápido, então confirme os valores marcados
com ⚠️ antes de assinar qualquer coisa. Câmbio usado nas conversões: **US$ 1 ≈ R$ 5,08**.

---

## 1. Resumo da decisão

**Você não precisa de chip.** O WhatsApp Cloud API nunca usou chip — ele registra um
*número*, e esse número pode ser VoIP, fixo ou virtual, desde que receba SMS ou
chamada de voz para o código de verificação.

Caminho recomendado, em duas etapas:

1. **Hoje, R$ 0:** use o **número de teste da Meta**. Ele vem pronto na conta de
   desenvolvedor, não precisa de número seu, e destrava o desenvolvimento imediatamente.
   Limite: 5 destinatários cadastrados. Serve para você e a equipe validarem o agente.
2. **Para o piloto real:** compre um **número virtual** (Twilio, Telnyx ou Salvy) e
   registre-o direto na **Meta Cloud API**, sem BSP intermediário. Custo do número:
   ~US$ 1–6/mês.

O que de fato trava a etapa 2 não é o número — é a **verificação de negócio da Meta**,
que no Brasil pede **CNPJ ativo**, comprovante de endereço e documento do
representante legal. Sem verificação você fica limitado a **250 destinatários únicos
por 24 h**, o que ainda comporta um piloto pequeno.

> Se hoje não há CNPJ disponível, essa é a dependência a resolver primeiro — não o
> número. O passo a passo, os documentos e o que fazer sem CNPJ estão na **§2.7**.

---

## 2. Os caminhos para o número, comparados

| # | Caminho | Custo do número | Prazo | CNPJ? | Serve para quê |
|---|---------|-----------------|-------|-------|----------------|
| A | **Número de teste da Meta** | R$ 0 | minutos | não | Desenvolvimento e demo. Máx. 5 destinatários |
| B | **Número virtual + Cloud API direta** | US$ 1–6/mês | 1–3 dias | sim, para escalar | **Recomendado** para o piloto e produção |
| C | **Twilio como BSP** | US$ 1,15/mês + markup por msg | horas | sim | Setup mais fácil; caro no volume |
| D | **BSP brasileiro** (Zenvia, Take Blip, Gupshup, 360dialog) | mensalidade fixa | 3–10 dias | sim | Quem quer nota fiscal em R$, suporte em PT-BR |
| E | **Coexistência com WhatsApp Business App** | — | — | sim | Só se você recuperar um número que já usa |
| F | **Não oficial** (Evolution API, Baileys, WPPConnect) | ~R$ 0 | horas | não | ❌ Não use aqui — ver §2.6 |

### 2.1 Número de teste da Meta (comece por aqui)

Em `developers.facebook.com` → criar app do tipo Business → adicionar o produto
WhatsApp. A Meta entrega um número de teste e um `PHONE_NUMBER_ID` na hora.

- Você cadastra até **5 números de destino**, cada um confirmado por código.
- O token inicial **expira em 24 h**. Para o serviço não cair todo dia, gere um token
  permanente por **System User** no Business Manager.
- ⚠️ Não dá para usar em piloto com mães reais — o limite de 5 é rígido.

### 2.2 Número virtual registrado direto na Cloud API (recomendado)

Você compra o número em uma operadora virtual e registra **você mesmo** na Meta.
Sem BSP no meio, **não existe markup por mensagem**.

Requisitos do número:

- Recebe **SMS ou chamada de voz** (número fixo/digital verifica por chamada — na tela
  de verificação escolha "Me ligue").
- **Não pode ter WhatsApp ativo**. Se já teve, apague a conta antes e espere.
- Não pode ser um número descartável gratuito — esses são bloqueados pela Meta.

Fornecedores que costumam funcionar:

| Fornecedor | Observação |
|---|---|
| **Twilio** | Número EUA ~US$ 1,15/mês. Alta taxa de aprovação. Número BR exige CNPJ e bundle regulatório |
| **Telnyx / Plivo** | Semelhante, às vezes mais barato |
| **Salvy** | Brasileiro, número móvel virtual, integra com WhatsApp API |
| **Zadarma / DIDWW** | Números de vários países, baratos |

⚠️ **Nem todo VoIP passa.** A Meta restringe faixas VoIP e a lista muda. Antes de
pagar por 12 meses, compre **um** número e teste a verificação.

⚠️ **Sobre número dos EUA:** resolve o lado telecom (sem Anatel, sem CNPJ para comprar
o número), mas uma mãe brasileira recebendo orientação de saúde de um **+1** tende a
desconfiar. Para um serviço materno-infantil, o número brasileiro vale o trabalho extra.

### 2.3 Twilio como BSP

A Twilio vende o número e cuida do registro do sender junto à Meta. É o caminho de
menor atrito técnico — e o código já suporta (`WHATSAPP_PROVIDER=twilio`).

O problema aparece no volume: a Twilio cobra a taxa dela **por mensagem**, inclusive
nas mensagens que a Meta entrega de graça. Ver §3.4.

### 2.4 BSP brasileiro

Zenvia, Take Blip, Gupshup, 360dialog. Vantagem real: **faturamento em reais com NF-e**,
suporte em português e ajuda na verificação de negócio. 360dialog cobra ⚠️ ~€49/mês por
número sem markup; os brasileiros negociam caso a caso.

Faz sentido se a burocracia é o gargalo, ou se a contabilidade precisa de nota fiscal
brasileira. Não faz sentido só pelo aspecto técnico.

### 2.5 Coexistência

Desde 2025 dá para conectar um número que já roda no WhatsApp Business App à Cloud API
mantendo o histórico. Não te ajuda agora, já que o número antigo se perdeu — fica
registrado para o caso de você recuperá-lo.

### 2.6 Por que não usar Evolution API / Baileys aqui

É o caminho barato e tentador: roda em cima do WhatsApp Web, sem taxa, sem verificação.
Três motivos para não usar **neste projeto**:

1. **Viola os Termos de Serviço.** Bans acontecem, e relatos apontam sobrevivência
   típica de semanas antes da detecção. Um serviço de saúde que some do ar sem aviso é
   pior do que não existir.
2. **Quebra sozinho.** É protocolo reversamente engenheirado; a Meta muda e você fica
   fora do ar até a comunidade atualizar a lib.
3. **Não sustenta a responsabilidade.** Orientação materno-infantil precisa de canal
   auditável, com identidade verificada e trilha de conformidade. Um número que pode
   sumir a qualquer momento não atende.

### 2.7 Como fazer a verificação de negócio na Meta

**Antes de tudo: talvez você não precise dela agora.** Uma conta **não verificada**
consegue conversar com até **250 destinatários únicos a cada 24 h**. Isso comporta um
piloto de 100 mães sem nenhuma burocracia. A verificação é o que destrava os degraus
seguintes (1.000 → 10.000 → 100.000 → ilimitado) e o cadastro de mais números.

⚠️ A Meta aperta esse limiar de tempos em tempos; confirme no painel antes de contar
com ele.

#### Onde fica

**Meta Business Suite** → **Configurações do negócio** → **Central de Segurança** →
seção *Verificação da empresa* → **Iniciar verificação**.

#### O que preencher — tem que bater *exatamente*

Este é o ponto onde a maioria das solicitações morre. Os dados digitados precisam ser
idênticos aos do documento:

| Campo | Regra |
|---|---|
| Razão social | A **razão social** do cartão CNPJ, não o nome fantasia nem a marca |
| CNPJ | Ativo na Receita Federal |
| Endereço | Endereço **registrado**, com a mesma formatação do documento. Nada de caixa postal |
| Telefone | Precisa receber ligação ou SMS para o código |
| Site | Domínio próprio, com e-mail no mesmo domínio, ajuda bastante |

#### Documentos aceitos no Brasil

- **Cartão CNPJ** (comprovante de inscrição da Receita Federal) — o principal
- **Contrato Social** — para sociedades
- **CCMEI** — o equivalente para MEI
- **Comprovante de endereço** no nome da empresa (conta de luz, água, telefone)
- Documento do **representante legal**

Prazo: ⚠️ tipicamente **3 a 10 dias úteis**, mais em época de fila.

#### Por que costuma ser recusado

Por ordem de frequência:

1. **Nome fantasia no lugar da razão social.** Se o cartão diz "MARIA SILVA
   DESENVOLVIMENTO LTDA" e você digitou "AppAM", é recusa automática.
2. **Endereço diferente.** Abreviação, complemento, CEP formatado de outro jeito.
3. **Documento ilegível.** Foto cortada, tremida ou de baixa resolução. Use o PDF
   original baixado do site da Receita.
4. **Documento vencido.** Comprovante de endereço antigo.
5. **Documento que mostra só o nome, sem o endereço.** Precisa dos dois.

#### E se não houver CNPJ nenhum?

Correção ao que escrevi antes: eu disse que "MEI tem CNPJ e costuma passar". É mais
matizado. O MEI **tem** CNPJ e pode tentar, usando o CCMEI no lugar do Contrato Social
— mas a taxa de recusa é maior, porque MEI costuma estar registrado em **endereço
residencial** e o comprovante de endereço raramente sai no nome da empresa. Dá para
passar; não conte com isso como certo.

Sem nenhum CNPJ, as saídas reais são:

- **Institucional.** Uma ONG, maternidade, banco de leite, UBS ou universidade parceira
  abre a WABA no CNPJ dela e o projeto opera sob esse guarda-chuva. Para um serviço
  materno-infantil isso normalmente é fácil de conseguir — e ainda melhora a confiança
  da mãe, porque o nome exibido no WhatsApp passa a ser o da instituição.
- **Abrir MEI.** Custa cerca de R$ 80/ano de DAS e sai no mesmo dia.
- **Ficar não verificada** e rodar o piloto dentro do teto de 250 destinatários/24 h.

Para o seu caso, a via institucional é a que eu perseguiria primeiro: resolve o CNPJ,
resolve a credibilidade e ajuda na revisão clínica do conteúdo ao mesmo tempo.

---

## 3. Precificação

### 3.1 A notícia boa: o canal é grátis para este caso de uso

A Meta mudou o modelo em **1º de julho de 2025**: a cobrança passou de conversa de 24 h
para **por mensagem de template**. Combinado com a mudança de novembro de 2024, o
resultado prático é:

| Tipo de mensagem | Custo |
|---|---|
| Usuário te manda mensagem e você responde em até 24 h (*service*) | **grátis** |
| Mensagem livre (texto, áudio, imagem) dentro da janela de 24 h | **grátis** |
| Template *utility* dentro da janela aberta | **grátis** |
| Template *utility* fora da janela (Brasil) | ⚠️ ~US$ 0,0080 |
| Template *authentication* (Brasil) | ⚠️ ~US$ 0,0315 |
| Template *marketing* (Brasil) | ⚠️ ~US$ 0,0625 |

**Seu agente é 100% reativo**: a mãe pergunta, ele responde dentro da janela.
Isso significa **custo zero de WhatsApp**. O que você paga é OpenAI e hospedagem.

Se um dia você adicionar mensagens proativas (ex.: "como está a amamentação na 2ª
semana?"), aí sim entra template *utility* — e é a partir daí que o custo do canal aparece.

⚠️ Um detalhe a confirmar no painel: algumas fontes de BSP citam "1.000 conversas de
serviço grátis por mês" como se fosse um teto. As fontes mais recentes indicam serviço
gratuito **sem** teto. Confirme no seu Business Manager antes de projetar volume alto.

### 3.2 Custo por interação (OpenAI)

Estimativa por pergunta respondida:

| Item | Tokens | Observação |
|---|---|---|
| Prompt de sistema | ~400 | fixo |
| Histórico curto | ~600 | 6 turnos |
| **Trechos recuperados pelo `file_search`** | **~5.000** | domina o custo |
| Pergunta | ~60 | |
| **Entrada total** | **~6.000** | |
| Saída | ~250 | resposta curta de WhatsApp |

| Modelo | Entrada / Saída (por 1M) | Custo/interação |
|---|---|---|
| `gpt-4o-mini` (atual) | US$ 0,15 / 0,60 | **~US$ 0,0011** |
| `gpt-4.1-mini` | ⚠️ US$ 0,40 / 1,60 | ~US$ 0,0028 |

Transcrição de áudio (`gpt-4o-mini-transcribe`): ⚠️ ~US$ 0,003/min → nota de voz de
30 s ≈ **US$ 0,0015**.

Base vetorial: US$ 0,10/GB/dia, **primeiro GB grátis**. Seus 44 MB de PDF/DOCX viram
bem menos de 1 GB de texto indexado → **R$ 0**.

### 3.3 Três cenários

Assumindo 40% das mensagens em áudio (realista para esse público) e `gpt-4o-mini`:

| | **Piloto** | **Operação** | **Escala** |
|---|---|---|---|
| Mães ativas | 100 | 1.000 | 10.000 |
| Interações/mês | 1.000 | 12.000 | 150.000 |
| WhatsApp (Meta) | US$ 0 | US$ 0 | US$ 0 |
| OpenAI — texto | US$ 1,05 | US$ 12,60 | US$ 157,50 |
| OpenAI — transcrição | US$ 0,60 | US$ 7,20 | US$ 90,00 |
| Base vetorial | US$ 0 | US$ 0 | US$ 0 |
| Número virtual | US$ 2 | US$ 2 | US$ 2 |
| Hospedagem | US$ 7 (Render Starter) | US$ 7 | US$ 25 (Pro) |
| **Total/mês** | **~US$ 11** | **~US$ 29** | **~US$ 275** |
| **Em reais** | **~R$ 55** | **~R$ 147** | **~R$ 1.400** |
| **Por mãe/mês** | R$ 0,55 | R$ 0,15 | R$ 0,14 |

**R$ 0,14 por mãe por mês em escala.** O custo marginal é irrelevante perto do valor
clínico. O gasto real do projeto é o trabalho humano: curadoria do conteúdo, validação
por consultora de amamentação e supervisão.

### 3.4 O que muda com um BSP

No cenário de escala (150.000 interações/mês), a Twilio cobrando ⚠️ ~US$ 0,005 por
mensagem enviada adicionaria **~US$ 750/mês** — quase **3× todo o resto somado**.

Conclusão: use BSP para destravar a burocracia se precisar, mas planeje migrar para
Cloud API direta antes de escalar. A abstração de canal já implementada existe
justamente para essa migração não custar reescrita.

### 3.5 Custo escondido: o plano free do Render

O plano gratuito hiberna após 15 min sem tráfego e leva ~1 min para acordar. Nesse
intervalo a Meta reentrega o webhook várias vezes e desiste. Para um serviço de saúde,
**US$ 7/mês do plano Starter não é opcional** — é o que separa "funciona" de
"funciona às vezes". O `render.yaml` já está com `plan: starter`.

---

## 4. Bloqueio novo: a política de IA da Meta (janeiro de 2026)

Esse é o achado mais importante da pesquisa, e ele **muda como o produto deve se
apresentar**.

Em **15 de janeiro de 2026** a Meta atualizou os Termos da Solução WhatsApp Business
para **proibir assistentes de IA de propósito geral** na plataforma. Foi assim que
ChatGPT, Copilot e Perplexity saíram do WhatsApp. Contas registradas a partir de
**15 de outubro de 2025** já estavam sujeitas à regra.

**O que continua permitido:** empresas usarem IA para atender seus próprios usuários
num escopo definido — FAQ, suporte, agendamento, informação sobre o próprio serviço.
A distinção é *IA como produto* (proibido) versus *IA como ferramenta de atendimento*
(permitido).

**Onde seu agente cai:** do lado permitido, desde que se apresente como o canal
educativo de um serviço específico de apoio materno-infantil — e **não** como
"assistente de IA que responde suas dúvidas". Na prática:

- ✅ Escopo travado em amamentação e cuidados com o bebê, com recusa explícita fora dele
- ✅ Respostas ancoradas numa base de conteúdo curada, não em conhecimento aberto
- ✅ Identificação clara como serviço automatizado de uma organização
- ❌ Nunca deixar conversar sobre qualquer assunto
- ❌ Nunca se posicionar como assistente genérico

Isso está implementado: a mensagem de boas-vindas declara o escopo e a natureza
automatizada, e a checagem de fundamentação recusa qualquer resposta que não venha da
base curada (§5.2).

### 4.1 Saúde no WhatsApp: o que a política diz

A Política de Mensagens do WhatsApp restringe **telemedicina** e o envio ou solicitação
de informação de saúde *quando a regulação local proíbe esse tráfego em sistemas sem
requisitos reforçados*. É uma cláusula **condicional**, não uma proibição geral — e no
Brasil não há norma que barre conteúdo educativo de saúde no WhatsApp. Chatbots de
saúde pública brasileiros operam no canal.

O que mantém você do lado seguro:

- Conteúdo **educativo**, nunca diagnóstico, dose ou prescrição
- **Não solicitar** dado clínico identificável (não peça nome do bebê, laudo, foto de lesão)
- Encaminhar sinal de alerta para atendimento presencial
- Disclaimer visível no primeiro contato

### 4.2 LGPD

Mensagem sobre saúde de mãe e bebê é **dado pessoal sensível** (art. 5º, II). Isso
implica:

| Obrigação | Situação |
|---|---|
| Base legal (consentimento ou tutela da saúde, art. 11) | ⬜ definir e registrar |
| Aviso de privacidade no primeiro contato | ✅ na mensagem de boas-vindas |
| Não gravar conteúdo/telefone em log claro | ✅ pseudonimização por HMAC |
| Direito de eliminação | ✅ comando `SAIR` apaga o histórico |
| Prazo de retenção definido | ✅ TTL de 30 min na memória de conversa |
| Transferência internacional (OpenAI nos EUA) | ⬜ declarar no aviso de privacidade |
| Contrato com operador (OpenAI/Meta) | ⬜ revisar DPA |
| Relatório de impacto (RIPD) | ⬜ recomendado antes do piloto público |
| Encarregado (DPO) indicado | ⬜ pode ser a responsável pelo projeto |

⚠️ Vale marcar no painel da OpenAI a opção de **não usar os dados para treinamento**
(padrão na API, mas confirme) e considerar *zero data retention* se o volume justificar.

---

## 5. Estado da arte e o que já mudou no código

### 5.1 O que estava quebrado

Auditando o `main.py` original, seis problemas impediriam o agente de funcionar em
produção — dois deles fariam a mãe receber resposta errada ou nenhuma resposta:

| # | Problema | Consequência real |
|---|---|---|
| 1 | Filtro que bloqueava respostas contendo "medicamento", "prescrev", "diagnóstico" | **Derrubava as respostas certas.** A base é sobre medicamentos na amamentação e o próprio modelo escreve "não posso prescrever". A mãe recebia sempre a mensagem genérica |
| 2 | Webhook processava de forma síncrona | A Meta reentrega quando o 200 demora; `file_search` leva segundos → **resposta duplicada ou triplicada** |
| 3 | Sem verificação de assinatura | Endpoint público: qualquer um que descobrisse a URL injetava mensagens e gastava sua cota |
| 4 | `raise ValueError` no import | Faltando uma variável, o processo **morria em loop** e o log do erro nunca era lido |
| 5 | Triagem por `substring` sem normalizar acento | `"não respira"` não casava com **"nao respira"** — justo o caso mais grave |
| 6 | `print` do telefone e do texto | Dado sensível em log de terceiro, contra a LGPD |

Além disso: `"febre"` disparava encaminhamento médico mesmo em pergunta conceitual, de
modo que *"posso amamentar se eu estiver com febre?"* nunca chegava à base.

### 5.2 O que foi implementado

**Fundamentação em vez de censura de palavra.** A checagem antiga procurava palavras
proibidas na resposta. A nova verifica se a resposta **cita a base vetorial**
(`file_citation`). Sem citação, o agente responde "não encontrei isso no material".
Isso ataca alucinação — o risco de verdade — em vez de punir vocabulário. É também o que
sustenta a conformidade com a política de escopo da Meta.

**Triagem clínica em duas camadas.** Emergência (não respira, convulsão, cianose) para
tudo e devolve o 192, sem depender de a OpenAI estar no ar. Sinal de alerta separa
*relato* ("meu bebê está com febre desde ontem" → encaminha) de *dúvida conceitual*
("posso amamentar com febre?" → responde **e** anexa nota de segurança). Tudo com
limite de palavra e sem acento.

**Webhook confiável.** Assinatura `X-Hub-Signature-256` verificada, resposta 200
imediata, processamento em background e deduplicação por `message_id`.

**Áudio.** Nota de voz é transcrita e segue o mesmo fluxo — inclusive a triagem de
emergência. Para esse público é a diferença entre usar e não usar o serviço.

**Canal plugável.** `WHATSAPP_PROVIDER=meta|twilio` troca o provedor sem tocar em
guardrail nem em RAG. É o seguro contra o gargalo do número.

**Privacidade.** Telefone vira pseudônimo HMAC no log, conteúdo nunca é gravado,
histórico expira em 30 min e `SAIR` apaga tudo.

**75 testes** cobrindo triagem, deduplicação, assinatura, fundamentação, áudio,
opt-out e rate limit.

### 5.3 Onde o projeto está frente ao estado da arte

| Dimensão | Hoje | Estado da arte |
|---|---|---|
| Recuperação | `file_search` da OpenAI | Busca híbrida (vetor + BM25) com reranker; controle de chunking |
| Verificação | Citação presente/ausente | *LLM-as-judge* de fundamentação por afirmação |
| Avaliação | Testes de guardrail | Conjunto dourado de perguntas validado por consultora IBCLC, rodando a cada mudança |
| Escalonamento | Encaminha para UBS/192 | Transferência para consultora humana na fila |
| Interface | Texto e áudio livres | Menu estruturado + texto (a Meta prefere bot estruturado) |
| Memória | 30 min em processo | Redis, com perfil da mãe (semanas pós-parto) para personalizar |
| Observabilidade | Log estruturado | Painel de taxa de fora-de-escopo, latência, disparo de guardrail |

O ponto mais fraco é o **`file_search` como caixa-preta**: você não controla chunking
nem reranking, e a qualidade da recuperação é o que determina a qualidade da resposta.
Vale medir antes de trocar — se a recuperação já está boa para o seu material, trocar por
pgvector é complexidade sem retorno.

---

## 6. A IA está boa? Devemos trocar de modelo?

Resposta curta: **o modelo não é o seu gargalo — mas o `gpt-4o-mini` está velho e
vale subir de versão dentro da OpenAI.** Trocar de *fornecedor* seria caro pelo motivo
errado.

### 6.1 Primeiro, uma boa notícia: vocês já estão na API certa

A **Assistants API é desligada em 26 de agosto de 2026** — daqui a um mês. Muito
projeto de RAG vai quebrar nessa data.

**O seu não.** O código usa a **Responses API** com `file_search` e `vector_store_ids`,
que é exatamente o *destino* da migração, não a origem. Nada a fazer.

⚠️ Um detalhe achado ao pinar as dependências: o SDK `openai` **1.x não tem**
`client.responses` nem `client.vector_stores`. Um pin em 1.x instala, importa e passa
nos testes com mock — e quebra na primeira mensagem real. O `requirements.txt` está em
`openai==2.48.0`, e há um teste (`tests/test_sdk_contract.py`) que falha no CI se
alguém baixar a versão.

### 6.2 Custo, no seu workload real (6.000 tokens de entrada, 250 de saída)

| Modelo | Entrada /1M | Saída /1M | Por pergunta | Por mês (12 mil) | Mantém o `file_search`? |
|---|---|---|---|---|---|
| **`gpt-4o-mini`** (atual) | US$ 0,15 | US$ 0,60 | US$ 0,0011 | **US$ 12,60** | ✅ |
| **`gpt-5-mini`** | ⚠️ US$ 0,25 | ⚠️ US$ 2,00 | US$ 0,0020 | **US$ 24,00** | ✅ |
| `gpt-5.4-mini` | ⚠️ US$ 0,75 | ⚠️ US$ 4,50 | US$ 0,0056 | US$ 67,50 | ✅ |
| Gemini 3 Flash | ⚠️ US$ 0,50 | ⚠️ US$ 3,00 | US$ 0,0038 | US$ 45,00 | ❌ |
| Claude Haiku 4.5 | ⚠️ US$ 1,00 | ⚠️ US$ 5,00 | US$ 0,0073 | US$ 87,00 | ❌ |

Repare na escala: mesmo a opção **mais cara** custa US$ 87/mês para mil mães — R$ 0,44
por mãe. **Custo não decide nada aqui.** Qualquer um desses cabe no orçamento.

### 6.3 O que realmente pesa: trocar de fornecedor custa o RAG inteiro

O `file_search` da OpenAI faz, gerenciado, uma pilha que ninguém vê: extração de texto
de PDF e DOCX, chunking, embeddings, busca híbrida (vetorial + palavra-chave),
reranking e rastreio de citação — que é justamente o que sustenta nossa checagem de
fundamentação.

Sair da OpenAI significa **reconstruir tudo isso**: pgvector, pipeline de embeddings,
reranker, e a lógica de citação. São semanas de trabalho e uma nova superfície de bugs.

E o ponto decisivo: **num RAG, a qualidade da resposta é determinada muito mais pela
recuperação do que pelo QI do modelo.** Se o trecho certo do material chega ao contexto,
qualquer modelo moderno escreve uma resposta boa. Se não chega, nenhum modelo salva.
Trocar de fornecedor troca a parte que menos importa e reconstrói a que mais importa.

### 6.4 A recomendação

1. **Fique na OpenAI por enquanto.** A integração com `file_search` vale mais que a
   diferença entre modelos pequenos de 2026.
2. **Saia do `gpt-4o-mini`.** É um modelo de 2024. Teste o `gpt-5-mini`: o custo dobra
   para US$ 24/mês a mil mães, o que é irrelevante, e a diferença em seguir instrução
   ("não prescreva", "não invente") tende a ser real.
3. **Decida com o eval, não com opinião.** É para isso que ele existe:

   ```bash
   python evals/run_eval.py --live --model gpt-4o-mini
   python evals/run_eval.py --live --model gpt-5-mini
   ```

   Compare taxa de fundamentação, recusa fora de escopo, vazamento de dose em `med-03`,
   latência p95 e custo. Aí a escolha vira dado.
4. **Meça a recuperação antes de culpar o modelo.** Se a taxa de "fora de escopo" estiver
   alta em perguntas que *estão* no material, o problema é a recuperação — e trocar de
   modelo não resolve.

### 6.5 Quando trocar de fornecedor faria sentido

- **Latência.** O WhatsApp é conversa; acima de ~5 s a mãe acha que travou. Claude
  Haiku 4.5 tem fama de primeiro token muito rápido (⚠️ ~600 ms em teste de terceiros).
  Se o eval mostrar p95 ruim e isso não melhorar reduzindo `MAX_RETRIEVAL_RESULTS`, aí
  sim vale medir outro fornecedor.
- **Juiz de segurança independente (P2).** Uma segunda IA, **de outro fornecedor**,
  checando se a resposta está fundamentada e livre de prescrição. O argumento aqui é
  bom: modelos da mesma família tendem a compartilhar pontos cegos. Isso usa o segundo
  fornecedor como *auditor*, não como motor — e não exige refazer o RAG.
- **Residência de dados no Brasil.** Se a LGPD virar exigência de não sair do país, o
  problema deixa de ser escolha de modelo e vira outra arquitetura inteira.

### 6.6 O que melhoraria a resposta mais do que qualquer troca de modelo

Em ordem de impacto:

1. **Validação clínica do conjunto dourado** por uma consultora de amamentação. Hoje o
   eval mede consistência, não correção. É a lacuna mais séria do projeto.
2. **Medir e ajustar a recuperação** — quantos trechos, de que tamanho, do documento certo.
3. **Curadoria da base.** Os PDFs grandes (cadernetas, 13 MB cada) têm muita coisa que
   não é amamentação e competem com o material específico na recuperação.
4. Só então: trocar de modelo.

---

## 7. Roadmap sugerido

### P0 — antes de qualquer mãe real usar

- [ ] Resolver o CNPJ e concluir a verificação de negócio da Meta (§2.7) — ou decidir
      rodar o piloto não verificado, dentro do teto de 250 destinatários/24 h
- [ ] Comprar **um** número virtual e testar a verificação antes de contratar plano anual
- [ ] Rodar `ingest_openai_kb.py` e guardar o `VECTOR_STORE_ID`
- [ ] Gerar token permanente via System User (o de teste expira em 24 h)
- [ ] Revisar `evals/golden_set.yaml` com uma **consultora de amamentação** e aprovar,
      uma a uma, as respostas que o agente dá hoje (`--live`)
- [ ] Definir a base legal LGPD e publicar o aviso de privacidade
- [ ] Render no plano Starter (não usar o free)

### P1 — primeiras semanas de piloto

- [ ] Escalonamento para humano quando o agente recusar duas vezes seguidas
- [ ] Painel de acompanhamento: fora-de-escopo, disparo de guardrail, latência, custo
- [ ] Menu estruturado de temas na abertura (alinha com a preferência da Meta e reduz
      pergunta fora de escopo)
- [ ] Redis no lugar da memória em processo, se subir mais de uma instância

### P2 — evolução

- [ ] Avaliação automatizada de fundamentação a cada deploy
- [ ] Personalização por semanas pós-parto
- [ ] Mensagens proativas por template *utility* (aqui entra custo de canal — §3.1)
- [ ] Medir a recuperação antes de considerar trocar o `file_search`

---

## 8. Fontes

Pesquisa de julho de 2026. Preço e política mudam; reconfirme o que estiver marcado com ⚠️.

- [Pricing on the WhatsApp Business Platform — Meta for Developers](https://developers.facebook.com/documentation/business-messaging/whatsapp/pricing)
- [Meta is Updating WhatsApp Pricing on July 1, 2025 — Twilio](https://www.twilio.com/en-us/changelog/meta-is-updating-whatsapp-pricing-on-july-1--2025)
- [WhatsApp Business API Pricing Brazil 2026 — Message Central](https://www.messagecentral.com/blog/whatsapp-business-api-pricing-brazil)
- [Messaging Limits — Meta for Developers](https://developers.facebook.com/documentation/business-messaging/whatsapp/messaging-limits)
- [Sobre a verificação da empresa no Meta Business Suite](https://pt-br.facebook.com/business/help/1095661473946872)
- [Why was my business verification submission rejected? — Meta](https://en-gb.facebook.com/business/help/2342133782492969)
- [Meta recusou sua verificação de empresa? 7 correções](https://anylinga.com/blog/pt/meta-business-verification-rejected-7-fixes.html)
- [Meta Verification Documents: What Gets Accepted (2026)](https://singhamandeep.com/meta-business-verification-documents-required/)
- [Not All Chatbots Are Banned: WhatsApp's 2026 AI Policy Explained — respond.io](https://respond.io/blog/whatsapp-general-purpose-chatbots-ban)
- [Meta bans general-purpose AI chatbots on WhatsApp Business — Dataslayer](https://www.dataslayer.ai/blog/meta-bans-general-purpose-ai-chatbots-on-whatsapp-business)
- [WhatsApp Business Messaging Policy](https://business.whatsapp.com/policy)
- [Is Telemedicine Allowed on the WhatsApp Business API? — tyntec](https://www.tyntec.com/helpcenter/docs/faqs/whatsapp-business/whatsapp-commerce-policy/is-telemedicine-allowed-on-the-whatsapp-business-api/)
- [Which Twilio Phone Numbers are Compatible with WhatsApp?](https://help.twilio.com/articles/360026678054)
- [Brazil: Regulatory Guidelines — Twilio](https://www.twilio.com/en-us/guidelines/br/regulatory)
- [Virtual Number for WhatsApp Business — 2026 Buyer's Guide](https://www.go4whatsup.com/guides/virtual-number-for-whatsapp/)
- [Número Virtual para WhatsApp via API — Salvy](https://salvy.com.br/numerovirtualmovel)
- [Telefone Digital no WhatsApp API — Huggy](https://blog.huggy.io/como-usar-numero-digital-no-whatsapp/)
- [Why Cheap WhatsApp Bots Get Your Number Banned — SporeSec](https://sporesec.com/en/blog/whatsapp-unofficial-api-ban-risk)
- [API Oficial do WhatsApp vs Evolution API e Baileys — Tipefy](https://blog.tipefy.com/api-oficial-do-whatsapp-vs-evolution-api-e-baileys-o-que-muda-na-pratica-para-sua-empresa)
- [OpenAI API Pricing (July 2026) — BenchLM](https://benchlm.ai/openai/api-pricing)
- [OpenAI Whisper API Pricing 2026](https://diyai.io/ai-tools/speech-to-text/openai-whisper-api-pricing-2026/)
- [Assistants API (v2) FAQ — OpenAI Help Center](https://help.openai.com/en/articles/8550641-assistants-api-v2-faq)
- [Render Pricing 2026](https://costbench.com/software/developer-tools/render/)
- [Best WhatsApp Business API Providers in Brazil 2026 — Message Central](https://www.messagecentral.com/blog/best-whatsapp-business-api-providers-brazil)
