-- O plano pago passa a se chamar LactAI Plus (nome interno "plus").
ALTER TABLE assinaturas DROP CONSTRAINT assinaturas_plano_check;
UPDATE assinaturas SET plano = 'plus' WHERE plano = 'acompanhada';
ALTER TABLE assinaturas ADD CONSTRAINT assinaturas_plano_check CHECK (plano IN ('plus'));
