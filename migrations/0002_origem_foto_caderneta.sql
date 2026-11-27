-- Vacinas lidas da foto da caderneta (e confirmadas pela pessoa) ganham
-- uma origem própria no diário, para sempre se saber de onde vieram.
ALTER TABLE registros_do_bebe DROP CONSTRAINT registros_do_bebe_origem_check;
ALTER TABLE registros_do_bebe ADD CONSTRAINT registros_do_bebe_origem_check
  CHECK (origem IN ('conversa','pergunta_do_lactai','correcao','foto_caderneta'));

-- Evita a mesma vacina anotada duas vezes (mesma foto mandada de novo).
CREATE UNIQUE INDEX vacina_unica ON registros_do_bebe (bebe_id, valor_texto, referente_a)
  WHERE tipo = 'vacina_tomada';
