# Memória, perfil, lembretes de vacina e e-Lactancia — desenho

Proposta para o LactAI ser a **memória do cuidador(a)**: lembrar de cada
conversa, guardar a história do bebê (peso, altura, alimentação) e avisar das
vacinas no mesversário. **Ainda não implementado.** Decisões marcadas com ⚖️
são da Watanabe AI Tech, não técnicas.

> Não é parecer jurídico. Por tratar dado de saúde de criança, vale revisão de
> advogado(a) de proteção de dados antes de abrir ao público.

Decisões já tomadas (26/11/2026):

- O conteúdo vem de material de autoria clínica; não há etapa de revisão
  clínica separada.
- Quem conversa é o **cuidador(a)**: mãe, pai, avó, outra pessoa.
- Guardar do bebê: nascimento, sexo, peso, altura, tipo de aleitamento; da
  mãe: idade. **Tudo acumulativo**: cada informação nova se soma à história,
  nada é sobrescrito.
- Lembretes **só de vacina, no mesversário**, e só se a pessoa quiser.
- Incluir a consulta de **medicamento na amamentação** com o e-Lactancia.

---

## 1. O que o cuidador(a) ganha

| Recurso | Exemplo |
|---|---|
| **Memória da conversa** | Na terça, o LactAI ainda sabe que na segunda falamos de rachadura. |
| **Perfil que entra na resposta** | "A Helena tem 2 meses e mama no peito exclusivo": a resposta já parte disso. |
| **Diário do bebê** | "Qual era o peso dela no mês passado?" → "4,2 kg em 03/10; hoje 5,1 kg (+900 g)." |
| **Lembrete de vacina** | No mesversário de 4 meses: "Hoje a Helena faz 4 meses! É a vez da 2ª dose da pentavalente, VIP, pneumo 10 e rotavírus." |
| **Medicamento e amamentação** | "Posso tomar ibuprofeno?" → resposta da base + nível de risco no e-Lactancia + link. |

---

## 2. LGPD: a base de tudo

### 2.1 Por que é exigente

- **Dado de saúde é sensível** (art. 5º, II). Fora do atendimento por
  profissional de saúde, a base legal é o **consentimento específico e
  destacado** (art. 11, I).
- **Dado de criança** (art. 14): consentimento específico e em destaque de
  **pelo menos um dos pais ou do responsável legal**, sempre no melhor
  interesse da criança. Como quem conversa pode ser avó ou outra pessoa, o
  LactAI pergunta a relação com o bebê, e **só guarda dado do bebê se a pessoa
  declarar ser mãe, pai ou responsável legal**. Os outros cuidadores são
  atendidos normalmente, mas sem diário do bebê.
- Quem guarda tem de **provar** o consentimento (art. 8º, §2º).
- **"Memória infinita" vira "enquanto usar"** (art. 15 e 16): guarda enquanto
  houver uso e apaga tudo após um período sem nenhuma mensagem. ⚖️ prazo
  proposto: **24 meses**, porque o calendário de vacinas vai até os 4–5 anos e
  há longos intervalos sem vacina.

### 2.2 Consentimentos separados

| Finalidade | O que libera | Sem ela |
|---|---|---|
| `atendimento` | Responder às perguntas (já existe) | Só a orientação de emergência (192) |
| `memoria` | Guardar conversa e diário do bebê | Memória de 30 min, como hoje |
| `lembretes` | Mensagem de vacina no mesversário | Nenhuma mensagem ativa |
| `melhoria` | Equipe ler conversas sem identificação, para melhorar respostas | Conversa nunca lida por pessoas |

Perguntas, cada uma num momento:

1. **Logo na entrada** (`memoria`):
   > Posso guardar o que você me contar sobre você e o bebê, para lembrar nas
   > próximas conversas e montar o histórico de peso, altura e mamadas? Fica
   > protegido, serve só para te atender e você apaga quando quiser
   > (mande *APAGAR*).
   > *1* – Pode guardar  *2* – Não guarde
2. **Relação com o bebê** (mãe, pai, responsável legal, outro): decide se o
   diário do bebê pode existir.
3. **Depois da data de nascimento** (`lembretes`):
   > Quer que eu te avise das vacinas da Helena no dia do mesversário?
   > *1* – Quero  *2* – Não precisa
4. **Depois de alguns dias de uso, em separado** (`melhoria`).

### 2.3 Direitos da titular (art. 18), por palavra no WhatsApp

| Mensagem | O que acontece |
|---|---|
| `MEUS DADOS` | Recebe tudo o que está guardado: perfil, diário do bebê, lembretes, consentimentos |
| `CORRIGIR` | Corrige um dado: grava a correção **como novo registro**, e o errado fica marcado como substituído |
| `PARAR LEMBRETES` | Revoga só `lembretes` |
| `APAGAR` / `SAIR` | Apaga **tudo** na hora. Fica só a prova do consentimento e da revogação, sem conteúdo |
| e-mail ao encarregado(a) | Cópia completa (portabilidade) e demais pedidos |

### 2.4 Segurança (art. 46)

- **Campos sensíveis cifrados no próprio aplicativo** (mensagens, resumo,
  observações, nomes) com a chave `CHAVE_DADOS`, que fica só nas variáveis do
  Render, nunca no banco nem no backup.
- **Telefone:** HMAC para achar a conta (como hoje) e, só com `lembretes`,
  cifrado para poder enviar.
- **Medidas e datas ficam legíveis** (peso, altura, nascimento): sem nome e
  sem telefone, não identificam ninguém, e precisam de conta e gráfico.
- **Painel:** nunca mostra conteúdo nem telefone. Leitura de conversa só para
  quem consentiu `melhoria`, sem identificação, e **cada acesso registrado**.
- **Logs:** continuam sem conteúdo.

### 2.5 Retenção (decidido em 26/11/2026)

| Situação | O que acontece |
|---|---|
| Em uso | Tudo guardado |
| **12 meses sem nenhuma mensagem** | Conta fica **inativa**: nada é apagado, e voltar a escrever reativa tudo |
| **Assinatura vencida** (não pagou) | Perde a assinatura; a conta fica inativa, sem apagar |
| **Inativa há 24 meses** | Apagada (conversa, perfil, diário do bebê) |
| **Pedido de exclusão** (`APAGAR`, ou `SAIR` confirmado) | Apagada **na hora** — é direito da titular (art. 18, VI) |
| Prova de consentimento/revogação | 5 anos após o fim, sem conteúdo |
| Registro de acesso da equipe | 2 anos |

O prazo vai escrito na própria pergunta de consentimento e na política de
privacidade, porque guardar depois que a pessoa para de usar só se sustenta
com transparência. Implementado em `MemoriaPersistente.aplicar_retencao`,
que roda a cada 6 horas (`RETENCAO_MESES_SEM_USO`, `RETENCAO_MESES_INATIVA`).

### 2.6 Fora do código ⚖️

- **Encarregado(a)** de dados nomeado(a), com contato na política.
- **Política de privacidade** nova (finalidades, prazos, operadores). O código
  só vai ao ar junto com ela.
- **Transferência internacional** declarada: OpenAI, Meta, Render (EUA).
- **RIPD** (art. 38): rascunho fica comigo.
- **Plano de incidente** (art. 48).

---

## 3. Banco de dados

**Postgres no Render** (~US$ 7/mês ⚖️), com backup diário. O registro do
piloto (hoje em SQLite) migra para ele.

### 3.1 Princípio: história acumulativa

Nada do que o cuidador(a) informa é sobrescrito. Cada informação é **um
registro com data**: "pesou 4,2 kg em 03/10" e "pesou 5,1 kg em 03/11" são
duas linhas. O valor "atual" é o mais recente. Correção é uma linha nova que
aponta para a que substitui (`substitui_id`) — a errada continua lá, marcada,
o que deixa claro o que foi dito e quando.

Exceção: o que **não muda** com o tempo (data de nascimento, sexo) fica na
linha do bebê; se for corrigido, a correção vai para o histórico de
correções.

### 3.2 Tabelas

Campos `_cifrado` são texto cifrado (`bytea`). Datas em UTC; "dia" e "hora de
enviar" no fuso de Brasília.

```sql
-- Quem conversa. Uma conta por número de WhatsApp.
CREATE TABLE cuidadores (
  id                     uuid PRIMARY KEY,
  telefone_hash          text UNIQUE NOT NULL,   -- HMAC, para achar a conta
  telefone_cifrado       bytea,                  -- só com consentimento de lembretes
  nome_preferido_cifrado bytea,                  -- como quer ser chamado(a); opcional
  relacao                text CHECK (relacao IN
                           ('mae','pai','responsavel_legal','outro')),
  ano_nascimento_mae     smallint,               -- idade da mãe, mesmo quando quem fala é outra pessoa
  gestante               boolean NOT NULL DEFAULT false,
  data_provavel_parto    date,
  criado_em              timestamptz NOT NULL DEFAULT now(),
  ultima_mensagem_em     timestamptz NOT NULL DEFAULT now()  -- base da retenção
);

-- Prova de consentimento. Sobrevive à exclusão da conta, sem conteúdo.
CREATE TABLE consentimentos (
  id            bigserial PRIMARY KEY,
  telefone_hash text NOT NULL,
  finalidade    text NOT NULL CHECK (finalidade IN
                  ('atendimento','memoria','lembretes','melhoria')),
  versao_texto  text NOT NULL,                   -- 'memoria-v1': qual texto foi lido
  concedido_em  timestamptz NOT NULL,
  revogado_em   timestamptz,
  evidencia     text NOT NULL                    -- id da mensagem com o "sim"
);
CREATE INDEX ON consentimentos (telefone_hash, finalidade) WHERE revogado_em IS NULL;

-- O que não muda: nascimento e sexo.
CREATE TABLE bebes (
  id                uuid PRIMARY KEY,
  cuidador_id       uuid NOT NULL REFERENCES cuidadores ON DELETE CASCADE,
  apelido_cifrado   bytea,                       -- "a Helena"; opcional
  data_nascimento   date NOT NULL,
  sexo              text CHECK (sexo IN ('feminino','masculino')),
  semanas_gestacao  smallint,                    -- prematuridade
  criado_em         timestamptz NOT NULL DEFAULT now()
);

-- Diário do bebê: uma linha por informação, nunca sobrescrita.
CREATE TABLE registros_do_bebe (
  id            bigserial PRIMARY KEY,
  bebe_id       uuid NOT NULL REFERENCES bebes ON DELETE CASCADE,
  tipo          text NOT NULL CHECK (tipo IN
                  ('peso_g','altura_cm','perimetro_cefalico_cm',
                   'aleitamento','vacina_tomada','observacao')),
  valor_numero  numeric,                         -- 5100 (g), 58.5 (cm)
  valor_texto   text,                            -- 'exclusivo','misto','formula','complementar'; código da vacina
  valor_cifrado bytea,                           -- observação livre
  referente_a   date NOT NULL,                   -- quando foi medido/aconteceu
  informado_em  timestamptz NOT NULL DEFAULT now(),
  origem        text NOT NULL,                   -- 'conversa','pergunta_do_lactai','correcao'
  substitui_id  bigint REFERENCES registros_do_bebe,  -- correção de um registro anterior
  CHECK (valor_numero IS NOT NULL OR valor_texto IS NOT NULL OR valor_cifrado IS NOT NULL)
);
CREATE INDEX ON registros_do_bebe (bebe_id, tipo, referente_a DESC);

-- Valor atual de cada tipo, ignorando os substituídos.
CREATE VIEW situacao_atual_do_bebe AS
SELECT DISTINCT ON (r.bebe_id, r.tipo) r.*
FROM registros_do_bebe r
WHERE NOT EXISTS (SELECT 1 FROM registros_do_bebe c WHERE c.substitui_id = r.id)
ORDER BY r.bebe_id, r.tipo, r.referente_a DESC, r.informado_em DESC;

-- Memória completa da conversa.
CREATE TABLE mensagens (
  id               bigserial PRIMARY KEY,
  cuidador_id      uuid NOT NULL REFERENCES cuidadores ON DELETE CASCADE,
  autor            text NOT NULL CHECK (autor IN ('cuidador','lactai')),
  conteudo_cifrado bytea NOT NULL,
  tipo             text NOT NULL DEFAULT 'texto',  -- texto | audio (transcrito)
  desfecho         text,
  criada_em        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON mensagens (cuidador_id, criada_em DESC);

-- Resumo corrido: é ele, não a conversa inteira, que vai para o modelo.
CREATE TABLE resumos (
  cuidador_id     uuid PRIMARY KEY REFERENCES cuidadores ON DELETE CASCADE,
  resumo_cifrado  bytea NOT NULL,
  ate_mensagem_id bigint NOT NULL,
  atualizado_em   timestamptz NOT NULL DEFAULT now()
);

-- Calendário de vacinas (referência, não é dado pessoal). Vem do material
-- da base (SBIm/SBP, PNI). Versão guardada para saber qual valia quando.
CREATE TABLE calendario_vacinas (
  codigo       text PRIMARY KEY,                 -- 'penta_2'
  nome         text NOT NULL,                    -- 'Pentavalente – 2ª dose'
  idade_meses  smallint NOT NULL,                -- mesversário em que é aplicada
  rede         text NOT NULL CHECK (rede IN ('sus','privada','ambas')),
  versao       text NOT NULL,                    -- 'PNI-2026'
  ativo        boolean NOT NULL DEFAULT true
);

-- Fila de lembretes: um por bebê por mesversário que tem vacina.
CREATE TABLE lembretes (
  id          bigserial PRIMARY KEY,
  bebe_id     uuid NOT NULL REFERENCES bebes ON DELETE CASCADE,
  idade_meses smallint NOT NULL,
  enviar_em   timestamptz NOT NULL,              -- mesversário, 9h de Brasília
  template    text NOT NULL,
  status      text NOT NULL DEFAULT 'pendente'
                CHECK (status IN ('pendente','enviado','falhou','cancelado')),
  enviado_em  timestamptz,
  erro        text,
  UNIQUE (bebe_id, idade_meses)                  -- nunca dois avisos do mesmo mês
);
CREATE INDEX ON lembretes (enviar_em) WHERE status = 'pendente';

-- Toda leitura de conversa pela equipe.
CREATE TABLE acessos_da_equipe (
  id          bigserial PRIMARY KEY,
  quem        text NOT NULL,
  cuidador_id uuid,
  acao        text NOT NULL,
  motivo      text NOT NULL,
  quando      timestamptz NOT NULL DEFAULT now()
);

-- Consultas ao e-Lactancia: cache do nível de risco por medicamento.
-- Não é dado pessoal (não guarda quem perguntou).
-- (Só se a consulta automática for autorizada; ver seção 7.)
CREATE TABLE elactancia_cache (
  termo        text PRIMARY KEY,                 -- 'ibuprofeno', normalizado
  produto      text,                             -- nome no e-Lactancia
  risco        text,                             -- 'muito_baixo','baixo','alto','muito_alto'
  url          text,
  consultado_em timestamptz NOT NULL DEFAULT now()
);
```

Tabelas do piloto (`convites`, `testadores`, `uso_diario`, `gasto_mensal`,
`recusas`) passam para o mesmo banco sem mudança.

**Mesversário em dia que não existe** (nasceu em 31/01): o lembrete vai no
último dia do mês (28/02).

### 3.3 Apagar de verdade

`APAGAR` faz `DELETE FROM cuidadores WHERE id = …`. Bebês, diário, mensagens,
resumo e lembretes vão juntos (`ON DELETE CASCADE`). Ficam só
`consentimentos` (sem conteúdo) e os contadores do piloto (sem conteúdo).
Backups do Render somem em 7 dias; a política informa esse prazo.

---

## 4. Estrutura evoluindo sem quebrar (CI/CD)

Os dados vão crescer e a estrutura vai mudar (um campo novo, um tipo de
registro novo). As regras para isso nunca apagar nem corromper o que já
está guardado:

1. **Migrações versionadas no repositório:** `migrations/0001_inicial.sql`,
   `0002_…`. Cada uma roda uma vez só; o banco guarda quais já rodaram
   (`schema_migrations`). Migração aplicada nunca é editada: corrige-se com
   outra.
2. **Aplicadas no deploy, antes do código novo subir:**
   `preDeployCommand: python -m app.migrar` no `render.yaml`. Se a migração
   falhar, o deploy para e o código antigo continua no ar.
3. **Só mudanças aditivas no mesmo deploy** (tabela ou coluna nova, com valor
   padrão). Remover ou renomear é feito em duas etapas, em deploys
   separados: primeiro o código para de usar, depois a coluna sai.
4. **Testadas no CI:** o GitHub Actions sobe um Postgres, aplica todas as
   migrações do zero, roda os testes e confere que o banco migrado bate com
   o esperado.
5. **Tipos de registro novos não pedem migração:** `registros_do_bebe` é
   genérica (tipo + valor + data). Começar a guardar, por exemplo,
   "perímetro cefálico" é só aceitar um tipo a mais.

---

## 5. Como a memória entra na resposta

A cada pergunta, o modelo recebe, nesta ordem:

1. **Perfil** (gerado da `situacao_atual_do_bebe`): "Cuidadora: mãe, 31 anos.
   Bebê: Helena, feminino, 2 meses e 10 dias, nasceu com 39 semanas. Último
   peso 5,1 kg (03/11). Aleitamento exclusivo desde o nascimento."
2. **Resumo da história** (até ~300 palavras).
3. **Últimas 6 trocas**, literais.
4. A pergunta.

**Capturar dados:** quando a conversa traz uma informação ("ela pesou 5,1
na consulta de ontem"), uma extração automática propõe o registro e o LactAI
**confirma antes de gravar**: "Anotei: Helena, 5,1 kg em 25/11. Certo?". A
data de nascimento é perguntada diretamente, uma vez.

**Responder pelo diário:** perguntas como "quanto ela pesava no mês
passado?" ou "quais vacinas ela já tomou?" são respondidas a partir do banco,
não do modelo.

---

## 6. Lembrete de vacina no mesversário

- Só para quem disse **sim** à pergunta de lembretes.
- Só nos mesversários em que há vacina no calendário (ex.: 2, 3, 4, 5, 6, 9,
  12, 15 meses…). Nos outros meses, nenhuma mensagem.
- Às 9h de Brasília do dia do mesversário; um aviso por bebê por mês.
- Fora da janela de 24 h, o WhatsApp só permite **modelo aprovado pela Meta**
  (categoria *utilidade*), cobrado por mensagem. ⚖️ conferir o preço atual da
  Meta para o Brasil.

  Modelo proposto (`lembrete_vacina_mesversario`):
  > Hoje {{1}} completa {{2}} meses! 🎉 Pelo calendário de vacinação, é a vez
  > de: {{3}}. Leve a caderneta da criança à unidade de saúde. Para não
  > receber mais estes avisos, responda PARAR LEMBRETES.
- Agendador: uma rodada por hora dentro do agente (como o vigia da conexão).
- A pessoa pode responder "já tomou": vira `vacina_tomada` no diário.

---

## 7. Medicamento e amamentação (e-Lactancia)

**Decidido em 26/11/2026: só indicar o site, sem consulta automática.**
Quando a pergunta é sobre remédio ou outra substância na amamentação
(`guardrails.is_medication_question`), a resposta da base sai como sempre e
ganha no fim a indicação do e-Lactancia (www.e-lactancia.org) e de confirmar
com o médico. Vale também quando a base não tem a resposta. **Já no ar.**

Consulta automática do nível de risco fica para depois, e só com autorização
da APILAM (apilam.es@gmail.com): o site não tem API pública nem termos que
permitam uso automatizado num produto pago.

---

## 8. Ordem de entrega

| Fase | Entrega |
|---|---|
| **1. Fundação** | Postgres, migrações no deploy e no CI, cuidadores, consentimentos, criptografia, `MEUS DADOS` / `APAGAR` — **feito, desligado até a política nova** (`MEMORIA_ATIVA`) |
| **2. Memória** | Conversa guardada e usada como histórico — **feito**; resumo e perfil no prompt — a fazer |
| **3. Diário do bebê** | Nascimento, sexo, relação, idade da mãe, peso/altura/aleitamento acumulativos, confirmação antes de gravar, perguntas respondidas pelo diário |
| **4. Vacinas** | Calendário, pergunta de lembretes, modelo na Meta, agendador, `PARAR LEMBRETES` |
| **5. e-Lactancia** | Indicação do site nas perguntas sobre remédio — **feito** |

As testadoras atuais recebem a pergunta de consentimento na próxima mensagem;
até responderem, o atendimento segue como hoje.
