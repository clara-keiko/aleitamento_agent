# Fase de testes (piloto fechado)

Só conversa com o LactAI quem entrou por um link de convite. Cada pessoa tem
um teto de perguntas por dia, e o piloto inteiro tem um teto de gasto no mês.

## Como funciona o convite

1. Você envia o link de convite (copie no painel, abaixo).
2. A pessoa toca no link: o WhatsApp abre com a frase
   *"Quero participar da fase de testes do LactAI PILOTO30"* já escrita.
3. Ela envia. Pronto: o número está autorizado, e ela recebe as boas-vindas.

Nada manual da sua parte. O código tem um número de vagas (30): quando
acabam, o link para de aceitar gente nova sozinho, mesmo que tenha sido
repassado.

| Situação | O que o agente faz |
|---|---|
| Convidada pergunta | Responde normalmente |
| Convidada toca no link de novo | "Você já faz parte" |
| Vagas acabaram | Avisa que o convite está cheio |
| Número sem convite | Avisa que a fase é fechada, **no máximo uma vez a cada 24 h**; depois fica em silêncio |
| Emergência, de qualquer número | Sempre orienta ligar 192 |
| Passou do limite do dia | Avisa que a contagem volta amanhã (meia-noite de Brasília) |
| Gasto do mês atingiu o teto | Avisa que o limite da fase foi atingido; volta no mês seguinte |

Só contam no limite as perguntas que chamam o modelo. Cumprimento ("oi",
"obrigada"), convite, emergência e encaminhamento médico não contam.

## Painel

`https://aleitamento-agent-cmb6.onrender.com/admin/piloto`

O navegador pede usuário e senha: **usuário qualquer**, senha = valor de
`ADMIN_TOKEN` (Render → aleitamento-agent → Environment). Mostra:

- o link de convite pronto para copiar;
- vagas usadas;
- quantas testadoras, quantas perguntaram hoje, quantas perguntas hoje;
- gasto do mês em dólares.

O painel nunca mostra telefone: o banco guarda só o HMAC do número.

### Liberar números à mão

Quando alguém manda a mensagem do convite e fica sem resposta (o caso típico:
o WhatsApp estava desconectado), cole o número no campo **Liberar números à
mão**, no fim do painel. A pessoa é liberada, ocupa uma vaga e recebe as
boas-vindas com um pedido de desculpas pela demora.

- Funciona só com o WhatsApp conectado; se não estiver, o painel avisa e não
  altera nada.
- Só cole números que pediram para entrar. Mensagem para quem não pediu é o
  que leva o WhatsApp a bloquear um número não oficial.
- Para achar quem tentou: no celular do LactAI, as conversas com a frase do
  convite sem resposta embaixo.

## Ajustes (Render → aleitamento-agent → Environment)

| Variável | Valor atual | Para que serve |
|---|---|---|
| `MODO_ACESSO` | `convite` | `aberto` libera para todo mundo |
| `CONVITES` | `PILOTO30:35` | Código e vagas. Mais vagas: `PILOTO30:50`. Mais de um: `PILOTO30:30,EQUIPE:5` |
| `LIMITE_DIARIO_POR_PESSOA` | `20` | `0` = sem limite |
| `ORCAMENTO_MENSAL_USD` | `15` | Teto do mês com a OpenAI. `0` = sem teto |
| `ALLOWED_SENDERS` | (vazio) | Números sempre liberados, sem convite (ex.: o seu) |

Esses valores vêm do `render.yaml`. Mudança feita só no painel do Render pode
voltar ao valor do arquivo na próxima sincronização do Blueprint. Para mudar
de vez, mude no `render.yaml`.

Aumentar vagas não tira ninguém. Trocar o código fecha o link antigo para
gente nova, mas quem já entrou continua.

## Quanto custa

O gasto é calculado com os tokens reais de cada resposta (preço do
gpt-5-mini: US$ 0,25 entrada / US$ 2,00 saída por milhão) mais a taxa da
busca na base (US$ 0,0025 por pergunta). A conta da OpenAI é a referência
final: o painel é uma estimativa próxima, não a fatura. A transcrição de
áudio não entra na conta (custa frações de centavo por minuto).

Ordem de grandeza (estimativa, ainda não medida): uma pergunta deve sair
entre US$ 0,005 e US$ 0,01. Trinta pessoas fazendo 3 perguntas por dia
num mês dão cerca de 2.700 perguntas, entre US$ 15 e US$ 25. Nos primeiros
dias, compare o painel com a conta da OpenAI e ajuste o teto.

## Cuidados

- **Os dados moram no disco `/var/data`** do serviço no Render. Sem esse
  disco, todo deploy tiraria todo mundo do piloto.
- **Não troque `PSEUDONYM_KEY`.** Os números são guardados como HMAC dessa
  chave; trocar a chave tira o acesso de todas as testadoras.
- Se o disco falhar, o agente segue com convites e limites em memória (o log
  avisa), que zeram a cada reinício. Os limites continuam valendo.
