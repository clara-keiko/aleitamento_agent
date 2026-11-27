# Lembretes de vacina no mesversário

No dia do mesversário em que há vacina prevista no **calendário do SUS**
(conferido no calendário baixado do gov.br, ver `app/calendario_sus.py`), o
LactAI manda um aviso a quem pediu. O aviso não lista vacinas: pede para
responder **VACINAS**, e a lista sai na resposta (SUS e rede particular, das
fontes oficiais, atualizada).

- Quem recebe: quem autorizou guardar, informou a data de nascimento no
  `PERFIL`, é mãe, pai ou responsável legal e respondeu **1** à pergunta de
  lembretes (que aparece no fim do questionário, ou com `LEMBRETES`).
- Quando: rodada de hora em hora, envio entre 9h e 19h de Brasília, no máximo
  um aviso por bebê por idade. Nasceu dia 31: em meses mais curtos, o aviso
  vai no último dia do mês.
- Parar: `PARAR LEMBRETES` revoga e apaga o telefone guardado para isso.
  `APAGAR` também.

## Modelo de mensagem para aprovar na Meta

Fora da janela de 24 h, o WhatsApp só aceita modelo aprovado.

WhatsApp Manager → **Modelos de mensagem** → **Criar modelo**:

| Campo | Valor |
|---|---|
| Categoria | **Utilidade** |
| Nome | `lembrete_vacina_mesversario` |
| Idioma | **Português (BR)** |

**Corpo:**

```
Oi! Hoje {{1}} completa {{2}}. 🎉

Pelo calendário de vacinação do SUS, há vacinas previstas para essa idade.

Responda *VACINAS* para ver quais são, no SUS e na rede particular.

Para não receber mais estes avisos, responda *PARAR LEMBRETES*.
```

**Exemplos das variáveis** (a Meta pede): `{{1}}` = `sua bebê`,
`{{2}}` = `2 meses`.

Depois de aprovado: `LEMBRETES_ATIVOS` = `true` no `render.yaml`.

Custo: cada aviso é uma conversa de utilidade cobrada pela Meta no cartão da
conta. Confira o preço atual para o Brasil na tabela da Meta.
