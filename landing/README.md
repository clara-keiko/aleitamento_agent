# Landing page

Modelo de página de vendas. Arquivo único, sem dependência externa — abre com
dois cliques e sobe em qualquer hospedagem estática.

```bash
python -m http.server 8080 --directory landing
# http://localhost:8080
```

## Antes de publicar

1. **Apague o aviso de rascunho.** É a linha marcada com
   `<!-- APAGUE ESTA LINHA AO PUBLICAR -->`, logo depois de `<body>`.
2. **Valide os preços.** Os valores são uma proposta, não uma decisão — veja
   abaixo de onde vieram.
3. **Troque os dados do rodapé**: nome do serviço, CNPJ, e-mail e os links de
   privacidade e termos.
4. **Ligue os botões.** Os `href="#"` dos planos precisam apontar para o
   WhatsApp (`https://wa.me/55DDDNUMERO`) e para o checkout.

## De onde vem a estrutura de preços

A escolha não é arbitrária: sai da economia real medida em
[`docs/OPERACAO.md`](../docs/OPERACAO.md).

| | Custo real | O que cobramos |
|---|---|---|
| Responder uma dúvida | ~R$ 0,14 por mãe/mês | R$ 0 |
| Hora de consultora | caro, e não escala | o plano pago |

Por isso o plano gratuito é ilimitado de verdade e o pago vende **tempo de
gente**. É honesto com o custo e coerente com o produto: o agente já foi
construído para encaminhar ao humano quando a dúvida sai do que dá para
resolver por mensagem.

⚠️ **R$ 29,90 é um chute plausível, não um preço validado.** Antes de publicar,
teste disposição a pagar com mães reais e verifique se uma consulta mensal com
consultora cabe nessa margem.

## Decisões de design

- **Fundo ameixa-noite e acento âmbar** vêm do momento real de uso: a mamada
  das 3h da manhã, com o abajur aceso e o celular numa mão só.
- **O herói é a conversa**, não uma foto de banco de imagens. O produto é a
  prova.
- **"O que este serviço não faz" fica na terceira dobra**, não no rodapé. Num
  serviço de saúde, dizer o limite cedo constrói mais confiança do que
  escondê-lo.
- **Dois temas**, claro e escuro, com a hierarquia preservada: os blocos
  escuros do herói e dos planos usam um token próprio, para continuarem se
  destacando quando a página inteira escurece.
