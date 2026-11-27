-- Modo família: até 3 pessoas (titular + 2) compartilham o bebê (perfil e
-- diário), o plano LactAI Plus e os lembretes. Conversas continuam
-- individuais. Só o titular convida e remove.
ALTER TABLE consentimentos DROP CONSTRAINT consentimentos_finalidade_check;
ALTER TABLE consentimentos ADD CONSTRAINT consentimentos_finalidade_check
  CHECK (finalidade IN ('atendimento','memoria','lembretes','melhoria','familia'));

CREATE TABLE familias (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  titular_hash text UNIQUE NOT NULL,
  criada_em    timestamptz NOT NULL DEFAULT now()
);

-- Uma família por pessoa. O telefone cifrado serve para o titular ver o
-- final do número e remover alguém.
CREATE TABLE membros_da_familia (
  telefone_hash    text PRIMARY KEY,
  familia_id       uuid NOT NULL REFERENCES familias ON DELETE CASCADE,
  telefone_cifrado bytea NOT NULL,
  entrou_em        timestamptz NOT NULL DEFAULT now()
);

-- Convite de uso único, com validade.
CREATE TABLE convites_de_familia (
  codigo     text PRIMARY KEY,
  familia_id uuid NOT NULL REFERENCES familias ON DELETE CASCADE,
  criado_em  timestamptz NOT NULL DEFAULT now(),
  expira_em  timestamptz NOT NULL,
  usado_em   timestamptz
);
