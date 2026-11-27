-- Custo real de cada chamada ao modelo, para calibrar os limites por plano.
-- Sem conteúdo: só quem (HMAC do telefone), o quê, quanto e quando.
-- Telefone vazio = protótipo web.
CREATE TABLE uso_do_modelo (
  id             bigserial PRIMARY KEY,
  telefone_hash  text,
  tipo           text NOT NULL CHECK (tipo IN ('pergunta','vacina','foto_caderneta')),
  custo_usd      numeric(12,6) NOT NULL,
  tokens_entrada integer NOT NULL DEFAULT 0,
  tokens_saida   integer NOT NULL DEFAULT 0,
  buscas_web     integer NOT NULL DEFAULT 0,
  criado_em      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX uso_por_data ON uso_do_modelo (criado_em);
CREATE INDEX uso_por_pessoa ON uso_do_modelo (telefone_hash, criado_em);
