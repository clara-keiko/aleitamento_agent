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

A estrutura é **implantação uma vez + mensalidade de operação**. Ela vem de três
constatações medidas no projeto do LactAI, não de comparação com concorrente:

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

### Sobre os números

⚠️ **R$ 3.900 / R$ 590 / R$ 2.500 são proposta, não preço validado.** A base de
cálculo é a infraestrutura medida mais o seu tempo, com a âncora do custo de um
atendente (R$ 2.000–3.000/mês mais encargos). Antes de mandar proposta de verdade:

- Cronometre quanto custou a implantação do LactAI em horas suas. É o único dado
  real de esforço que você tem, e ele define se R$ 3.900 é lucro ou prejuízo
- Confirme o limite de 3.000 atendimentos/mês na mensalidade — pela calculadora
  cabe com folga, mas confira com o volume que o cliente projeta
- Teste disposição a pagar antes de fixar. Preço de tabela na página é compromisso

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
