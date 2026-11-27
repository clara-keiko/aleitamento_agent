-- Lembretes de vacina no mesversário. Uma linha por bebê por idade: impede
-- avisar duas vezes o mesmo mês e registra os meses sem vacina, para não
-- reconsultar o calendário.
CREATE TABLE lembretes (
  id          bigserial PRIMARY KEY,
  bebe_id     uuid NOT NULL REFERENCES bebes ON DELETE CASCADE,
  idade_meses smallint NOT NULL,
  status      text NOT NULL CHECK (status IN ('enviado','sem_vacina')),
  registrado_em timestamptz NOT NULL DEFAULT now(),
  UNIQUE (bebe_id, idade_meses)
);
