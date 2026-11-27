-- Cobrança pelo Asaas. Cada pessoa ganha um link de pagamento recorrente
-- próprio; o aviso de pagamento traz o id do link, que diz de quem é.
-- O telefone cifrado serve só para confirmar o pagamento no WhatsApp.
CREATE TABLE links_de_pagamento (
  link_id          text PRIMARY KEY,
  telefone_hash    text NOT NULL,
  telefone_cifrado bytea NOT NULL,
  url              text NOT NULL,
  assinatura_asaas text,
  criado_em        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX links_por_pessoa ON links_de_pagamento (telefone_hash);

-- O Asaas entrega cada aviso pelo menos uma vez: um pagamento só renova uma vez.
CREATE TABLE pagamentos_processados (
  pagamento_id  text PRIMARY KEY,
  evento        text NOT NULL,
  processado_em timestamptz NOT NULL DEFAULT now()
);
