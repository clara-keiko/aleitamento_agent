# Memória, perfil e lembretes — desenho

Proposta para o LactAI lembrar de cada mãe e de cada bebê, e avisar de vacinas,
consultas e do que importa em cada fase. **Ainda não implementado.** As
decisões marcadas com ⚖️ são da Watanabe AI Tech, não técnicas.

> Este documento não é parecer jurídico. Por tratar dado de saúde de criança,
> vale uma revisão de advogado(a) de proteção de dados antes de abrir para o
> público.

---

## 1. O que a mãe ganha

| Recurso | Exemplo |
|---|---|
| **Memória da conversa** | Na terça, o LactAI ainda sabe que na segunda ela falou de rachadura. |
| **Perfil da mãe e do bebê** | "A Helena tem 2 meses e mama no peito exclusivo": a resposta já parte disso. |
| **Lembrete de vacina** | "Amanhã a Helena completa 4 meses: é a 2ª dose da pentavalente, VIP, pneumo e rotavírus." |
| **Lembrete de consulta** | "Está chegando a consulta de 6 meses com o pediatra." |
| **O que esperar da fase** | Aos 3–4 meses: "é comum o bebê se distrair no peito; não é falta de leite." |

---

## 2. LGPD: a base de tudo

### 2.1 Por que é exigente

- **Dado de saúde é dado sensível** (art. 5º, II). Fora do atendimento por
  profissional de saúde, a base legal que sobra é o **consentimento
  específico e destacado** (art. 11, I).
- **Dado de criança** exige consentimento **específico e em destaque de pelo
  menos um dos pais** e tratamento no **melhor interesse da criança**
  (art. 14). A mãe consente pelo bebê.
- Quem guarda tem de **provar** o consentimento (art. 8º, §2º).
- **Memória infinita não existe na LGPD:** o dado só pode ficar enquanto
  servir à finalidade (art. 15 e 16). A tradução prática é "enquanto ela usar
  o serviço", com um prazo de inatividade que apaga tudo. ⚖️

### 2.2 Consentimentos separados

Cada finalidade tem o seu "sim", e cada uma pode ser retirada sozinha:

| Finalidade | O que libera | Sem ela |
|---|---|---|
| `atendimento` | Responder às perguntas (já existe) | Só a resposta de emergência (192) |
| `memoria` | Guardar conversa e perfil da mãe e do bebê | Funciona como hoje: memória de 30 min |
| `lembretes` | Mandar mensagem sem ela ter escrito antes | Nenhuma mensagem ativa |
| `melhoria` | Equipe ler conversas, sem identificação, para revisar respostas | Conversas nunca lidas por pessoas |

Primeira mensagem de quem entra (um balão, perguntas curtas, resposta por
número):

> Para eu lembrar de você e do seu bebê entre uma conversa e outra, preciso
> guardar o que você me contar sobre a saúde de vocês dois. Isso fica
> protegido, só serve para te atender e você apaga quando quiser (é só
> mandar *APAGAR*).
>
> *1* – Pode guardar
> *2* – Não guarde, só responda minhas dúvidas

Depois do "1", em outra mensagem: lembretes (sim/não). A pergunta sobre
`melhoria` vem só depois de alguns dias de uso, separada, para não virar
"aceito tudo".

### 2.3 Direitos da titular (art. 18), por palavra no próprio WhatsApp

| Mensagem | O que acontece |
|---|---|
| `MEUS DADOS` | Ela recebe o que está guardado: perfil, bebês, lembretes ativos e consentimentos |
| `CORRIGIR` | Ela corrige um dado do perfil (por exemplo, a data de nascimento) |
| `PARAR LEMBRETES` | Revoga `lembretes`; o resto continua |
| `APAGAR` / `SAIR` | Apaga **tudo** na hora: conversa, perfil, bebês, lembretes. Fica só a prova do consentimento e da revogação, sem conteúdo |
| e-mail ao encarregado | Cópia completa (portabilidade) e demais pedidos |

### 2.4 Segurança (art. 46)

- **Criptografia dos campos sensíveis no próprio aplicativo** (texto das
  mensagens, resumo, observações, nomes), com uma chave `CHAVE_DADOS` que fica
  só nas variáveis do Render, nunca no banco. Um vazamento do banco ou de um
  backup não expõe conteúdo.
- **Telefone:** HMAC para achar a conta (como hoje) **e** telefone cifrado,
  porque lembrete precisa de destino. Sem `lembretes`, o telefone cifrado nem
  é gravado.
- **Painel:** nunca mostra conteúdo nem telefone. A leitura de conversas
  (para revisão clínica) só vale para quem consentiu com `melhoria`, mostra o
  texto sem nome nem telefone, e **cada acesso fica registrado** (quem, quando,
  por quê).
- **Logs:** continuam sem conteúdo.

### 2.5 Retenção ⚖️

| Dado | Prazo proposto |
|---|---|
| Conversa, resumo, perfil, bebês | Enquanto houver uso. **Apagado após 12 meses sem nenhuma mensagem** |
| Lembretes já enviados | 90 dias (prova de envio), depois só a contagem |
| Prova de consentimento/revogação | 5 anos após o fim (defesa em processo), sem conteúdo |
| Registro de acesso da equipe | 2 anos |

### 2.6 O que precisa existir fora do código ⚖️

- **Encarregado(a)** de dados nomeado(a), com contato na política
  (pode ser alguém da própria empresa).
- **Política de privacidade** reescrita: finalidades, prazos, operadores.
  O código só muda junto com ela.
- **Transferência internacional** declarada: OpenAI (EUA), Meta (EUA) e
  Render (servidor em Oregon, EUA).
- **RIPD** (relatório de impacto, art. 38). Para dado sensível de criança é o
  documento que a ANPD pede primeiro. Posso redigir o rascunho.
- **Plano de incidente**: quem avisa a ANPD e as titulares, e em quanto tempo
  (art. 48).

---

## 3. Banco de dados

**Postgres no Render** (plano basic-256mb, ~US$ 7/mês) em vez do SQLite atual:
backup automático diário, recuperação, e não depende de um disco preso a um
serviço. ⚖️ custo. O registro do piloto migra para ele.

Campos com sufixo `_cifrado` guardam texto cifrado (`bytea`). Datas em UTC;
"dia" e "hora de enviar" calculados no fuso de Brasília.

```sql
-- Quem conversa. Uma conta por número de WhatsApp.
CREATE TABLE contas (
  id                 uuid PRIMARY KEY,
  telefone_hash      text UNIQUE NOT NULL,        -- HMAC; para achar a conta
  telefone_cifrado   bytea,                       -- só com consentimento de lembretes
  nome_preferido_cifrado bytea,                   -- como ela quer ser chamada (opcional)
  gestante           boolean NOT NULL DEFAULT false,
  data_provavel_parto date,                       -- se gestante
  criada_em          timestamptz NOT NULL DEFAULT now(),
  ultima_mensagem_em timestamptz NOT NULL DEFAULT now()   -- base da retenção
);

-- Prova de consentimento. Nunca apagada junto com a conta: é o que mostra
-- que pedimos, o que ela respondeu e quando retirou.
CREATE TABLE consentimentos (
  id             bigserial PRIMARY KEY,
  telefone_hash  text NOT NULL,                   -- sobrevive à exclusão da conta
  finalidade     text NOT NULL CHECK (finalidade IN
                   ('atendimento','memoria','lembretes','melhoria')),
  versao_texto   text NOT NULL,                   -- qual texto ela leu (ex.: 'memoria-v1')
  concedido_em   timestamptz NOT NULL,
  revogado_em    timestamptz,
  evidencia      text NOT NULL                    -- id da mensagem do WhatsApp com o "sim"
);
CREATE INDEX ON consentimentos (telefone_hash, finalidade) WHERE revogado_em IS NULL;

-- Bebês da conta (gêmeos, segundo filho).
CREATE TABLE bebes (
  id                   uuid PRIMARY KEY,
  conta_id             uuid NOT NULL REFERENCES contas ON DELETE CASCADE,
  apelido_cifrado      bytea,                     -- opcional; "a Helena"
  data_nascimento      date NOT NULL,             -- base de vacinas, consultas e fases
  semanas_gestacao     smallint,                  -- prematuro muda o calendário
  alimentacao          text CHECK (alimentacao IN
                         ('exclusivo','misto','formula','complementar')),
  observacoes_cifradas bytea,                     -- o que ela contou e importa
  criado_em            timestamptz NOT NULL DEFAULT now(),
  atualizado_em        timestamptz NOT NULL DEFAULT now()
);

-- Memória completa. Só com consentimento de memória.
CREATE TABLE mensagens (
  id               bigserial PRIMARY KEY,
  conta_id         uuid NOT NULL REFERENCES contas ON DELETE CASCADE,
  autor            text NOT NULL CHECK (autor IN ('mae','lactai')),
  conteudo_cifrado bytea NOT NULL,
  tipo             text NOT NULL DEFAULT 'texto',  -- texto | audio (transcrito)
  desfecho         text,                           -- respondido, emergencia... (do pipeline)
  criada_em        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON mensagens (conta_id, criada_em DESC);

-- Resumo corrido da história, atualizado a cada ~10 mensagens. É ele, e não
-- a conversa inteira, que vai para o modelo: "memória infinita" sem custo
-- infinito.
CREATE TABLE resumos (
  conta_id        uuid PRIMARY KEY REFERENCES contas ON DELETE CASCADE,
  resumo_cifrado  bytea NOT NULL,
  ate_mensagem_id bigint NOT NULL,
  atualizado_em   timestamptz NOT NULL DEFAULT now()
);

-- Conteúdo de referência: não é dado pessoal. Toda linha passa por revisão
-- de profissional de saúde antes de valer.
CREATE TABLE calendario (
  codigo         text PRIMARY KEY,                -- 'vacina_penta_1', 'consulta_6m', 'fase_3m'
  tipo           text NOT NULL CHECK (tipo IN ('vacina','consulta','fase')),
  idade_dias     integer NOT NULL,                -- a partir do nascimento
  titulo         text NOT NULL,                   -- "Pentavalente – 1ª dose"
  texto          text NOT NULL,                   -- o que vai na mensagem
  rede           text,                            -- 'sus', 'privada', 'ambas'
  fonte          text NOT NULL,                   -- "PNI 2026", "Caderneta da Criança 7ª ed."
  revisado_por   text,                            -- nome e registro do profissional
  revisado_em    date,
  ativo          boolean NOT NULL DEFAULT false   -- só vale depois de revisado
);

-- Vacinas e consultas de cada bebê: as previstas pelo calendário e as que a
-- mãe informou que já fez.
CREATE TABLE marcos_do_bebe (
  id             bigserial PRIMARY KEY,
  bebe_id        uuid NOT NULL REFERENCES bebes ON DELETE CASCADE,
  marco_codigo   text NOT NULL REFERENCES calendario (codigo),
  data_prevista  date NOT NULL,
  realizado_em   date,                            -- informado pela mãe
  UNIQUE (bebe_id, marco_codigo)
);

-- Fila de envio. Gerada a partir de marcos_do_bebe e das mensagens por fase.
CREATE TABLE lembretes (
  id             bigserial PRIMARY KEY,
  conta_id       uuid NOT NULL REFERENCES contas ON DELETE CASCADE,
  bebe_id        uuid REFERENCES bebes ON DELETE CASCADE,
  marco_codigo   text REFERENCES calendario (codigo),
  enviar_em      timestamptz NOT NULL,            -- sempre entre 9h e 19h de Brasília
  template       text NOT NULL,                   -- modelo aprovado na Meta
  status         text NOT NULL DEFAULT 'pendente'
                   CHECK (status IN ('pendente','enviado','falhou','cancelado')),
  enviado_em     timestamptz,
  erro           text
);
CREATE INDEX ON lembretes (enviar_em) WHERE status = 'pendente';

-- Toda vez que alguém da equipe lê uma conversa.
CREATE TABLE acessos_da_equipe (
  id        bigserial PRIMARY KEY,
  quem      text NOT NULL,
  conta_id  uuid,                                 -- pode ficar órfão se a conta for apagada
  acao      text NOT NULL,                        -- 'leu_conversa', 'exportou'
  motivo    text NOT NULL,                        -- 'revisão clínica semanal'
  quando    timestamptz NOT NULL DEFAULT now()
);
```

Tabelas do piloto que já existem (`convites`, `testadores`, `uso_diario`,
`gasto_mensal`, `recusas`) passam para o mesmo banco sem mudança.

### 3.1 Apagar de verdade

`APAGAR` faz `DELETE FROM contas WHERE id = …`. Bebês, mensagens, resumo,
marcos e lembretes vão juntos (`ON DELETE CASCADE`). Ficam só
`consentimentos` (sem conteúdo) e o contador de uso do piloto (sem conteúdo).
Backups: o Render guarda os diários por 7 dias; depois disso, o dado some
também deles. A política informa esse prazo.

---

## 4. Como a memória entra na resposta

A cada pergunta, o modelo recebe, nesta ordem:

1. **Perfil** (curto, estruturado): "Mãe: Ana. Bebê: Helena, 2 meses e 10 dias,
   nascida com 39 semanas, peito exclusivo."
2. **Resumo da história** (até ~300 palavras).
3. **Últimas 6 trocas**, literais.
4. A pergunta.

O perfil é atualizado por **pergunta direta**, não por adivinhação: a data
de nascimento é perguntada uma vez, logo depois do consentimento ("Quando o
seu bebê nasceu? Pode ser só dia e mês"). Coisas ditas no meio da conversa
("ele nasceu prematuro, de 35 semanas") são capturadas por uma extração
automática e **confirmadas com ela** antes de gravar.

---

## 5. Lembretes no WhatsApp oficial

- Fora da janela de 24 h desde a última mensagem dela, o WhatsApp **só deixa
  enviar modelos (templates) aprovados pela Meta**. Cada tipo de lembrete vira
  um modelo, com campos variáveis (nome do bebê, vacina, data).
- **Custo por mensagem** (cobrado pela Meta no cartão da conta). Lembrete de
  vacina e de consulta tende a ser categoria *utilidade*; dica por fase tende
  a ser *marketing*, mais cara. ⚖️ Conferir a tabela atual da Meta para o
  Brasil antes de ligar.
- A Meta também exige opt-in para mensagem ativa: é o consentimento
  `lembretes`.
- Envio: 9h–19h de Brasília, no máximo 1 lembrete por dia por mãe (agrupa
  vacinas do mesmo dia), parando sozinho em `PARAR LEMBRETES`.
- Agendador: uma rodada por hora dentro do próprio agente (como o vigia da
  conexão), ou um Cron Job do Render.

---

## 6. Ordem de entrega

| Fase | Entrega | Depende de |
|---|---|---|
| **1. Fundação LGPD** | Postgres, contas, consentimentos, criptografia, `MEUS DADOS` / `APAGAR`, política nova | Decisões ⚖️ 2.5, 2.6 e banco |
| **2. Memória** | Conversa guardada, resumo, perfil no prompt | Fase 1 |
| **3. Perfil do bebê** | Pergunta da data de nascimento, gestante/DPP, prematuridade | Fase 2 |
| **4. Calendário** | Vacinas (PNI) e consultas de puericultura revisadas | **Revisão de profissional de saúde** |
| **5. Lembretes** | Modelos aprovados na Meta, agendador, `PARAR LEMBRETES` | Fases 3–4 e aprovação dos modelos |
| **6. Fases** | Mensagens do que esperar em cada idade | Revisão clínica + custo de marketing |

As testadoras atuais recebem a pergunta de consentimento na próxima mensagem
que mandarem; até responderem, o atendimento segue como hoje.
