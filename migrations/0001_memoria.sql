-- Memória do cuidador(a) e diário do bebê. Ver docs/MEMORIA_E_LEMBRETES.md.
--
-- Regras de evolução (valem para toda migração daqui em diante):
--   * migração aplicada nunca é editada; corrige-se com outra;
--   * só mudança aditiva no mesmo deploy (tabela/coluna nova com padrão);
--   * remover ou renomear: primeiro o código para de usar, depois a coluna sai.
--
-- Campos *_cifrado guardam texto cifrado pelo aplicativo (app/cripto.py).

CREATE TABLE cuidadores (
  id                     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  telefone_hash          text UNIQUE NOT NULL,
  telefone_cifrado       bytea,
  nome_preferido_cifrado bytea,
  relacao                text CHECK (relacao IN ('mae','pai','responsavel_legal','outro')),
  ano_nascimento_mae     smallint CHECK (ano_nascimento_mae BETWEEN 1940 AND 2020),
  gestante               boolean NOT NULL DEFAULT false,
  data_provavel_parto    date,
  -- Retenção: 12 meses sem mensagem → inativa; inativa há 24 meses → apagada.
  -- Assinatura vencida também inativa, sem apagar na hora.
  status                 text NOT NULL DEFAULT 'ativa' CHECK (status IN ('ativa','inativa')),
  inativa_desde          timestamptz,
  motivo_inativa         text CHECK (motivo_inativa IN ('sem_uso','assinatura_vencida')),
  criado_em              timestamptz NOT NULL DEFAULT now(),
  ultima_mensagem_em     timestamptz NOT NULL DEFAULT now()
);

-- Prova do que foi perguntado e respondido. Sobrevive à exclusão da conta e
-- não guarda conteúdo. Uma linha por decisão: a vigente é a mais recente.
CREATE TABLE consentimentos (
  id            bigserial PRIMARY KEY,
  telefone_hash text NOT NULL,
  finalidade    text NOT NULL CHECK (finalidade IN ('atendimento','memoria','lembretes','melhoria')),
  aceito        boolean NOT NULL,
  versao_texto  text NOT NULL,
  decidido_em   timestamptz NOT NULL DEFAULT now(),
  revogado_em   timestamptz,
  evidencia     text NOT NULL
);
CREATE INDEX consentimentos_vigentes ON consentimentos (telefone_hash, finalidade, decidido_em DESC);

CREATE TABLE bebes (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  cuidador_id      uuid NOT NULL REFERENCES cuidadores ON DELETE CASCADE,
  apelido_cifrado  bytea,
  data_nascimento  date NOT NULL,
  sexo             text CHECK (sexo IN ('feminino','masculino')),
  semanas_gestacao smallint CHECK (semanas_gestacao BETWEEN 20 AND 45),
  criado_em        timestamptz NOT NULL DEFAULT now()
);

-- Diário acumulativo: uma linha por informação, nunca sobrescrita.
CREATE TABLE registros_do_bebe (
  id            bigserial PRIMARY KEY,
  bebe_id       uuid NOT NULL REFERENCES bebes ON DELETE CASCADE,
  tipo          text NOT NULL,
  valor_numero  numeric,
  valor_texto   text,
  valor_cifrado bytea,
  referente_a   date NOT NULL,
  informado_em  timestamptz NOT NULL DEFAULT now(),
  origem        text NOT NULL CHECK (origem IN ('conversa','pergunta_do_lactai','correcao')),
  substitui_id  bigint REFERENCES registros_do_bebe ON DELETE CASCADE,
  CHECK (valor_numero IS NOT NULL OR valor_texto IS NOT NULL OR valor_cifrado IS NOT NULL)
);
CREATE INDEX registros_por_tipo ON registros_do_bebe (bebe_id, tipo, referente_a DESC);

CREATE VIEW situacao_atual_do_bebe AS
SELECT DISTINCT ON (r.bebe_id, r.tipo) r.*
FROM registros_do_bebe r
WHERE NOT EXISTS (SELECT 1 FROM registros_do_bebe c WHERE c.substitui_id = r.id)
ORDER BY r.bebe_id, r.tipo, r.referente_a DESC, r.informado_em DESC;

CREATE TABLE mensagens (
  id               bigserial PRIMARY KEY,
  cuidador_id      uuid NOT NULL REFERENCES cuidadores ON DELETE CASCADE,
  autor            text NOT NULL CHECK (autor IN ('cuidador','lactai')),
  conteudo_cifrado bytea NOT NULL,
  tipo             text NOT NULL DEFAULT 'texto' CHECK (tipo IN ('texto','audio')),
  desfecho         text,
  criada_em        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX mensagens_recentes ON mensagens (cuidador_id, id DESC);

CREATE TABLE resumos (
  cuidador_id     uuid PRIMARY KEY REFERENCES cuidadores ON DELETE CASCADE,
  resumo_cifrado  bytea NOT NULL,
  ate_mensagem_id bigint NOT NULL,
  atualizado_em   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE acessos_da_equipe (
  id          bigserial PRIMARY KEY,
  quem        text NOT NULL,
  cuidador_id uuid,
  acao        text NOT NULL,
  motivo      text NOT NULL,
  quando      timestamptz NOT NULL DEFAULT now()
);
