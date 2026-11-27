-- Assinatura do plano pago. Contrato, não dado de saúde: fica separado da
-- conta de memória e não some com o APAGAR (a cobrança precisa dele).
-- Sem linha, ou com pago_ate no passado: plano gratuito.
CREATE TABLE assinaturas (
  telefone_hash text PRIMARY KEY,
  plano         text NOT NULL CHECK (plano IN ('acompanhada')),
  inicio        date NOT NULL,          -- âncora do ciclo mensal
  pago_ate      date NOT NULL,          -- vale até este dia, inclusive
  origem        text NOT NULL DEFAULT 'manual',
  atualizado_em timestamptz NOT NULL DEFAULT now()
);
