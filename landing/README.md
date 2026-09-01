# Site do LactAI

Página do produto, destinada a **lactai.co**. Arquivo único, sem dependência externa
— sobe em qualquer hospedagem estática.

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

⚠️ **Em domínio próprio, o bloco de identificação do rodapé é a única prova dessa
ligação.** Num caminho como `watanabeaitech.com.br/lactai`, a própria URL faria esse
trabalho; em `lactai.co`, não há nada além do rodapé. Ele não é decoração — é o item
que o revisor procura.

## Antes de publicar

1. **Preencha o bloco de identificação** no rodapé, com dados reais:
   razão social completa, CNPJ, endereço da empresa
2. **Troque o e-mail de contato** (`contato@lactai.co`)
3. **Ligue os botões** — os `href="#"` precisam apontar para
   `https://wa.me/55DDDNUMERO` e para o checkout
4. **Publique os links** de Privacidade e Termos, ou remova-os do rodapé enquanto
   não existirem

## Hospedagem

Site estático de um arquivo. As três opções gratuitas equivalentes:

| Serviço | Como |
|---|---|
| **Cloudflare Pages** | Conecta o repositório, diretório `landing/`, sem build |
| **Netlify** | Idem, ou arraste a pasta na interface |
| **Vercel** | Idem |

Depois aponte o DNS de `lactai.co` para o serviço escolhido (eles dão os registros) e
confirme que o HTTPS ficou ativo — a Meta rejeita site sem certificado.

## Depois de publicar

1. **Portfólio empresarial** → Configurações do negócio → *Informações da empresa* →
   campo **Site** → `https://lactai.co`
2. **Reenvie o nome de exibição** no WhatsApp Manager

Reforço que vale o esforço: cite o LactAI também em `watanabeaitech.com.br`, com link
para `lactai.co`. A Meta pede que a relação esteja clara **nos sites das duas partes**
quando o nome de exibição difere da razão social — com as duas pontas ligadas, a
análise deixa de depender de interpretação.

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
