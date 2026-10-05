# Site do LactAI

Página do produto, no ar em **https://aleitamento.com.br/lactai/**. Arquivo único,
sem dependência externa — sobe em qualquer hospedagem estática.

```bash
python3 -m http.server 8080 --directory landing
# http://localhost:8080
```

## Por que ela precisa existir

Não é só marketing. **É o que destrava o nome de exibição no WhatsApp.**

A Meta aceita um nome de exibição diferente da razão social, desde que a ligação
entre a marca e a empresa verificada esteja visível publicamente. Foi exatamente
isso que faltou quando "LactAI" foi recusado da primeira vez: o nome não existia em
lugar nenhum que o revisor pudesse conferir.

⚠️ **O bloco de identificação do rodapé é a única prova dessa ligação.** A página mora
num domínio de terceiro (`aleitamento.com.br`), então a URL não ajuda o revisor: nada
nela contém "LactAI" nem "Watanabe". O rodapé não é decoração — é o item que ele
procura, e por isso diz três coisas em ordem:

1. quem **opera** o LactAI, com razão social e CNPJ — o mesmo CNPJ verificado no
   portfólio empresarial;
2. que a conta de WhatsApp que atende como LactAI é dessa empresa — que é exatamente
   a afirmação sob análise no nome de exibição;
3. por que a página está num domínio de outra empresa.

O item 3 existe para fechar a lacuna que o revisor enxergaria sozinho. Sem ele a
página parece de terceiro; com ele, a hospedagem vira informação declarada.

## Antes de publicar

1. **Troque o e-mail de contato** se `contato@watanabeaitech.com.br` não estiver ativo
   — o revisor pode escrever
2. **Ligue os botões** — os `href="#"` precisam apontar para
   `https://wa.me/55DDDNUMERO` e para o checkout
3. **Publique os links** de Privacidade e Termos, ou remova-os do rodapé enquanto
   não existirem

## Hospedagem

Site estático de um arquivo — hoje servido dentro de `aleitamento.com.br`. Se um dia
migrar para domínio próprio, as três opções gratuitas equivalentes:

| Serviço | Como |
|---|---|
| **Cloudflare Pages** | Conecta o repositório, diretório `landing/`, sem build |
| **Netlify** | Idem, ou arraste a pasta na interface |
| **Vercel** | Idem |

Em qualquer caso, confirme que o HTTPS está ativo — a Meta rejeita site sem
certificado.

## Depois de publicar

1. **Portfólio empresarial** → Configurações do negócio → *Informações da empresa* →
   campo **Site** → `https://aleitamento.com.br/lactai/`
2. **Publique a outra ponta**: uma menção ao LactAI em `watanabeaitech.com.br`, com
   link para esta página. A Meta pede que a relação esteja clara **nos sites das duas
   partes** quando o nome de exibição difere da razão social. Esta é a metade que
   ainda falta — e é a mais fácil das duas, porque o domínio é seu.
3. **Reenvie o nome de exibição** no WhatsApp Manager

Não dá para verificar `aleitamento.com.br` no seu portfólio (*Segurança da marca →
Domínios*) se ele já estiver verificado em outro — um domínio pertence a um portfólio
só. `watanabeaitech.com.br`, esse sim, vale verificar: é o domínio que sustenta a
afirmação do rodapé.

## Sobre os preços

A estrutura sai da economia real medida em [`../docs/OPERACAO.md`](../docs/OPERACAO.md):
responder custa cerca de R$ 0,14 por mãe/mês, enquanto hora de consultora é cara e
não escala. Daí o plano gratuito ser ilimitado e o pago vender tempo de gente — o que
também é coerente com o agente, que encaminha ao humano quando a dúvida sai do que se
resolve por mensagem.

⚠️ **R$ 29,90 é proposta plausível, não preço validado.** A página está pública para
destravar a análise da Meta; antes de direcionar clientes para o checkout, teste
disposição a pagar e confirme se uma consulta mensal com consultora cabe nessa margem.

## Decisões de design

- **Fundo ameixa-noite e acento âmbar** vêm do momento real de uso: a mamada das 3h,
  com o abajur aceso e o celular numa mão só.
- **O herói é a conversa**, não uma foto de banco de imagens. O produto é a prova.
- **"O que este serviço não faz" fica na terceira dobra**, não no rodapé. Num serviço
  de saúde, declarar o limite cedo constrói mais confiança do que escondê-lo — e é o
  mesmo princípio dos guardrails do agente.
- **Dois temas** com hierarquia preservada: os blocos escuros do herói e dos planos
  usam um token próprio, para continuarem se destacando quando a página inteira
  escurece.
