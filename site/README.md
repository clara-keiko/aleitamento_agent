# Site da Watanabe AI Tech

Site de vendas, destinado a **watanabeaitech.com**. Arquivo único, sem dependência
externa além das fontes do Google.

```bash
python3 -m http.server 8080 --directory site
# http://localhost:8080
```

## Antes de publicar

1. **Ligue o botão do WhatsApp** — `href="https://wa.me/55DDDNUMERO"` precisa do
   número real
2. **Confirme que `contato@watanabeaitech.com.br` recebe** — é o único canal de
   entrada além do WhatsApp
3. **Peça autorização ao cliente** antes de manter a seção do LactAI. Citar cliente
   pelo nome sem combinar é o tipo de coisa que custa a relação — e a seção sai com
   uma tecla de delete se ele preferir não aparecer
4. **Revise os preços.** Ver abaixo.

## O modelo de negócio, e de onde ele saiu

A estrutura tem **dois eixos independentes**: implantação uma vez (define o tipo de
trabalho) e mensalidade por faixa de volume (define o tamanho do cliente). Ela vem de
três constatações medidas no projeto do LactAI, não de comparação com concorrente:

**A burocracia é a barreira, e é ela que se vende.** Colocar um agente no ar exigiu
verificação de CNPJ na Meta, aprovação de nome de exibição, portfólio empresarial,
KYC de BSP. Nenhuma pequena empresa passa por isso sozinha. É o serviço — "chatbot"
qualquer um entrega.

**Infraestrutura não é custo relevante.** [`../scripts/custos.py`](../scripts/custos.py)
mede: R$ 245/mês para mil usuários ativos, e a IA consome 0,7% de uma mensalidade de
R$ 29,90. Quem precifica por token está olhando a linha errada da planilha. A margem
está no tempo de gente.

**O ativo tem que ser do cliente.** Se a conta do WhatsApp fica no seu CNPJ, você
herda a suspensão, a reclamação de spam e o KYC de cada cliente, para sempre. Com a
conta no CNPJ dele e você como administradora, o risco fica onde deve — e isso virou
promessa comercial na seção "O que não fazemos".

### Por que atendimento, e não mensagem

Cobrar por mensagem parece justo e tem um defeito estrutural: **um agente bom
resolve em três mensagens o que um ruim resolve em dez**. Cobrando por mensagem,
a receita sobe quando a qualidade cai — o oposto do que a página inteira vende.

Atendimento (uma conversa, um cliente resolvido) é a unidade de valor e não muda
com a eficiência do agente. Faixa fechada em vez de medição por unidade porque dono
de negócio pequeno tolera preço alto muito melhor do que tolera fatura variável.

A infraestrutura, medida com `custos.py`, é quase plana — o volume sobe 10× e o
custo 2,7×, porque hospedagem domina:

| Atendimentos/mês | Infra | Preço | Margem |
|---|---|---|---|
| 100 | R$ 40 | R$ 129 | R$ 89 |
| 500 | R$ 46 | R$ 190 | R$ 144 |
| 2.000 | R$ 72 | R$ 390 | R$ 318 |
| 5.000 | R$ 124 | R$ 690 | R$ 566 |

Ou seja: as faixas **não** são preço por custo. São preço por valor, com o custo
apenas definindo o piso.

⚠️ **O plano Balcão é o mais arriscado dos quatro.** R$ 89 de margem é uma ligação
de meia hora. Ele só se sustenta com suporte assíncrono — por isso a tabela promete
"resposta em 1 dia útil" nessa faixa e não menciona telefone. Se na prática o
cliente de R$ 129 ligar toda semana, o plano está errado e é para subir o preço,
não para engolir.

### Sobre os números

O preço não é limitado pelo custo de operar. Para uma empresa de bairro com ~300
atendimentos/mês, a infraestrutura inteira sai por **R$ 43/mês**
([`custos.py --maes 100 --interacoes-por-mae 3`](../scripts/custos.py)). O que
limita é a **sua hora de implantação**:

| Implantação | Horas que ela banca a R$ 100/h |
|---|---|
| R$ 3.900 | ~39 h |
| R$ 1.200 | ~12 h |

Daí os dois níveis. O **Essencial** só fecha a conta se a implantação couber em
~12 h, o que exige **base pronta por setor** — clínica, salão, oficina, comércio.
Sem esse trabalho feito antes, vender a R$ 1.200 é fazer 39 horas por 12.

**A consequência prática:** antes de anunciar o Essencial, monte duas ou três
verticais. Enquanto elas não existirem, venda só o sob medida e use o Essencial
como sinal de para onde o negócio vai.

⚠️ **Nenhum destes preços foi validado.** Antes de usá-los em proposta:

- Cronometre quantas horas a implantação do LactAI consumiu. É o único dado real
  de esforço que você tem, e é ele que diz se R$ 3.900 é lucro ou prejuízo
- Decida o seu valor-hora antes de decidir o preço, não depois
- R$ 290/mês tem ~R$ 250 de margem sobre infraestrutura. Ela existe para cobrir
  suporte: duas ligações longas no mês já consomem a margem do cliente inteiro.
  Por isso o escopo de suporte está escrito na página — "horário comercial",
  não "ilimitado"
- Teste disposição a pagar. Preço de tabela publicado é compromisso

## Decisões de design

- **Ficha técnica, não folder.** A página abre com números medidos em vez de
  adjetivos. É coerente com o que se está vendendo: alguém que mede antes de
  entregar.
- **Numeração só onde há sequência.** As quatro semanas são numeradas porque a ordem
  carrega informação; os três serviços não, porque não há primeiro nem último.
- **"O que não fazemos" antes do contato.** Mesma decisão do site do LactAI. Declarar
  limite cedo constrói mais confiança do que escondê-lo — e a seção do QR code
  explica, sem atacar ninguém, por que você custa mais que o concorrente barato.
- **Verde-garrafa em vez do roxo de IA.** O acento aparece em pouca coisa: números,
  marcadores de lista, estados de foco. O laranja é reservado às barras da seção de
  limites — cor de aviso usada só onde há aviso.
- **Serifa no corpo, grotesca nos títulos.** Inversão do padrão de site de tecnologia.
  Quem lê é dono de padaria, clínica, escritório — não desenvolvedor.

## Hospedagem

Site estático de um arquivo. Cloudflare Pages, Netlify ou Vercel servem igualmente:
conecta o repositório, aponta o diretório `site/`, sem build. Depois aponte o DNS de
`watanabeaitech.com` e confirme o HTTPS.

## Uma pendência que vale dinheiro

O site do LactAI afirma que a Watanabe opera o serviço. Este site não menciona o
LactAI em lugar nenhum — e a Meta pede que a relação esteja clara **nos sites das
duas partes** quando o nome de exibição difere da razão social.

A seção "Em operação" resolve isso de graça, desde que o cliente autorize. Se ele não
autorizar, vale uma linha discreta no rodapé mencionando o LactAI com link — é a
metade que falta daquela verificação.
