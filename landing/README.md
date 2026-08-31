# Landing do LactAI

Página do produto, destinada a `watanabeaitech.com.br/lactai`. Arquivo único, sem
dependência externa — sobe em qualquer hospedagem estática.

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

O rodapé já traz a linha no formato que a Meta procura:

> **LactAI** é um produto da **Watanabe AI Tech** · CNPJ 00.000.000/0001-00

Publicar em `watanabeaitech.com.br/lactai` reforça isso pela própria URL — a relação
fica evidente antes mesmo de o revisor rolar até o rodapé.

## Antes de publicar

1. **Preencha o CNPJ** no rodapé (hoje está `00.000.000/0001-00`)
2. **Troque o e-mail de contato** (`contato@exemplo.com.br`)
3. **Ligue os botões** — os `href="#"` precisam apontar para
   `https://wa.me/55DDDNUMERO` e para o checkout
4. **Publique os links** de Privacidade e Termos, ou remova-os do rodapé enquanto
   não existirem

## Depois de publicar

Aponte o site no portfólio empresarial: **Configurações do negócio → Informações da
empresa → Site**. É assim que o revisor da Meta sabe onde procurar o nome.

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
