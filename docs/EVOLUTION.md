# Canal Evolution API

Conexão com o WhatsApp **sem Meta e sem BSP**, usando o protocolo do WhatsApp
Web multi-dispositivo. O número entra como aparelho vinculado.

> ⚠️ **Não é conexão oficial.** Viola os termos do WhatsApp; o número pode ser
> banido sem aviso e sem recurso. Serve para protótipo, demonstração e uso
> interno. Para serviço em que alguém depende da resposta, use `meta` ou
> `twilio` — o mesmo agente atende pelos três.

> ⚠️ **Não use o número que já está na WABA.** Um número é *ou* conta comum *ou*
> Cloud API, nunca os dois. Parear por QR exige removê-lo da Cloud API, o que
> desfaz registro e nome de exibição. Use um segundo número.

---

## Teste 1 — sem Evolution e sem número

Prova o canal inteiro (parse, triagem, RAG, dedup) antes de instalar qualquer
coisa. Basta o app já publicado.

No painel do Render, defina e salve:

```
WHATSAPP_PROVIDER = evolution
EVOLUTION_BASE_URL = https://ainda-nao-existe.invalid
EVOLUTION_API_KEY  = uma-chave-qualquer-de-teste
EVOLUTION_INSTANCE = teste
```

Confirme que subiu: `curl https://SEU-APP.onrender.com/health` deve trazer
`"provider":"evolution"` e `"whatsapp_ready":true`.

Agora simule uma mensagem chegando:

```bash
curl -i https://SEU-APP.onrender.com/webhook \
  -H 'Content-Type: application/json' \
  -H 'apikey: uma-chave-qualquer-de-teste' \
  -d '{
    "event": "messages.upsert",
    "instance": "teste",
    "data": {
      "key": {"remoteJid": "5521999999999@s.whatsapp.net",
              "fromMe": false, "id": "TESTE-1"},
      "message": {"conversation": "como sei se a pega está correta?"}
    }
  }'
```

**Esperado:** HTTP 200 na hora. Nos logs, `processado user=… outcome=respondido`
seguido de uma falha de envio — correta, porque `EVOLUTION_BASE_URL` aponta para
lugar nenhum. O que se está testando é a entrada.

Vale repetir trocando o texto por `meu bebe nao respira`: o desfecho tem que ser
`emergencia`, sem passar pelo modelo.

Sem o header `apikey`, a resposta tem que ser **403**.

## Teste 2 — com Evolution de verdade

### Subir a Evolution

Ela roda em Docker. No Render: **New → Web Service → Deploy an existing image**.

```
Imagem:  docker.io/evoapicloud/evolution-api:v2.3.7
Porta:   8080
```

Duas escolhas por trás disso. O repositório é o **`evoapicloud`**, não o
`atendai` que aparece em tutoriais antigos — aquele parou em fevereiro de 2025.
E a versão é **fixa**: o `latest` do `evoapicloud` aponta para uma candidata a
versão, que não é onde se pareia um número em uso.

Se o Render disser *"No public image found"*, é quase sempre o nome sem o
prefixo `docker.io/`.

⚠️ Variáveis e rotas abaixo vêm do que se conhece da v2 e **não foram conferidas
contra a documentação corrente** — confira em `doc.evolution-api.com`. Se algo
divergir, o erro aparece na resposta da própria Evolution.

Variáveis:

```
AUTHENTICATION_API_KEY = <gere uma chave forte; é a senha do serviço>
```

**Acrescente um Disk** (Render → Disks), montado onde a Evolution guarda a
sessão. Sem isso, cada deploy perde o pareamento e pede QR de novo — o sistema
de arquivos do Render é efêmero. Alternativa: apontar a Evolution para Postgres
ou Redis, que ela suporta.

### Rodar tudo na sua máquina

Mais rápido e mais barato que o Render, e é o caminho recomendado para
protótipo. `evolution/docker-compose.yml` sobe **Evolution, Postgres, Redis e o
agente** na mesma rede:

```bash
cd evolution
cp .env.exemplo .env          # preencha as chaves
docker compose up -d
docker compose logs -f agente
```

Como estão na mesma rede do Docker, os dois se acham pelo nome do serviço: o
agente chama `http://evolution:8080` e a Evolution entrega o webhook em
`http://agente:10000/webhook`. **Sem túnel, sem IP, sem expor nada na
internet.**

O agente sobe pelo Dockerfile do projeto, que traz Python 3.11 — a versão do
Python da sua máquina não entra nessa história.

No host: agente em `http://localhost:8000` (o protótipo web fica em `/chat`) e
Evolution em `http://localhost:8080`.

Para parear, use `AGENT_WEBHOOK_URL='http://agente:10000/webhook'` — é o
endereço que a Evolution enxerga, não o seu navegador.

### Criar a instância e parear

`scripts/evolution_setup.py` faz os passos na ordem certa — instância, webhook,
QR — e mostra o estado entre eles:

```bash
export EVOLUTION_BASE_URL='https://sua-evolution.onrender.com'
export EVOLUTION_API_KEY='a-chave-que-voce-gerou'
export EVOLUTION_INSTANCE='lactai'
export AGENT_WEBHOOK_URL='https://SEU-APP.onrender.com/webhook'

python3 scripts/evolution_setup.py criar
python3 scripts/evolution_setup.py qr       # salva e abre qr.png
```

No celular do **segundo número**: WhatsApp → Aparelhos conectados → Conectar um
aparelho → aponte para o QR. Ele expira em cerca de um minuto; rode `qr` de novo
se demorar.

```bash
python3 scripts/evolution_setup.py estado   # tem que dizer "open"
python3 scripts/evolution_setup.py testar 5521SEUCELULAR
```

O `testar` manda uma mensagem sem envolver o agente. Se ela chegar, o envio
está resolvido e o que resta é só apontar o agente para a instância — o que
separa "a Evolution funciona" de "o meu código funciona", em vez de depurar os
dois de uma vez.

### Apontar o agente

No Render, no serviço do agente:

```
EVOLUTION_BASE_URL = https://sua-evolution.onrender.com
EVOLUTION_API_KEY  = a-chave-que-voce-gerou
EVOLUTION_INSTANCE = lactai
```

A `EVOLUTION_API_KEY` tem papel duplo: o agente a envia ao chamar a Evolution, e
a confere nos webhooks que ela manda. Por isso precisa ser a mesma dos dois lados.

A Evolution (v2.3.7, ao menos) **não repete a chave em header** nas chamadas que
faz. Por isso os scripts registram o webhook com a chave na própria URL
(`.../webhook?apikey=...`). O agente também aceita a chave em header `apikey` ou
no campo `apikey` do corpo, para quem configurar de outro jeito. O log de falha
mascara a chave da URL.

### Conversar

De outro celular, mande mensagem para o número pareado. Deve chegar a
apresentação do serviço e depois a resposta.

## Produção no Render

O `render.yaml` sobe três peças pelo mesmo Blueprint do agente: o serviço
`evolution-api` (com disco para a sessão), o banco `evolution-db` e o agente já
apontado para a Evolution. O endereço e a chave passam de um serviço para o
outro sozinhos (`fromService`), sem valor copiado à mão.

Custo aproximado: US$ 7 do agente + US$ 7 da Evolution + ~US$ 6 do banco + o
disco, ~US$ 20/mês.

1. **Aplicar o Blueprint.** Render → Blueprints → o do projeto → sincronizar.
   Ele lista o `evolution-api` e o `evolution-db` como novos; confirme.
2. **Pegar a chave e o endereço.** Em `evolution-api` → Environment, copie o
   `AUTHENTICATION_API_KEY` gerado. O endereço público está no topo do serviço.
3. **Desligar a Evolution local antes de parear a nova**, ou o número fica em
   duas sessões e cada mensagem recebe resposta duplicada:
   ```bash
   cd evolution
   CHAVE=$(grep AUTHENTICATION_API_KEY .env | cut -d= -f2)
   curl -s -X DELETE -H "apikey: $CHAVE" http://localhost:8080/instance/logout/lactai
   docker compose down
   ```
   No celular, em Aparelhos conectados, remova o aparelho antigo se ele ainda
   aparecer.
4. **Parear no servidor:**
   ```bash
   export EVOLUTION_BASE_URL='https://evolution-api-XXXX.onrender.com'
   export EVOLUTION_API_KEY='a-chave-copiada'
   export EVOLUTION_INSTANCE='lactai'
   export AGENT_WEBHOOK_URL='https://SEU-APP.onrender.com/webhook'
   python3 scripts/evolution_setup.py criar
   python3 scripts/evolution_setup.py qr
   python3 scripts/evolution_setup.py estado
   ```
5. **Conferir:** `https://SEU-APP.onrender.com/health/ready` deve responder 200
   com `"whatsapp_connection": "open"`.

### Alerta de queda

Com a Evolution, o `/health/ready` responde **503 quando o WhatsApp
desconecta**, mesmo com toda a configuração certa. O `/health`, que o Render
usa como healthcheck, continua 200 — senão o Render reiniciaria o agente por um
problema que é do WhatsApp.

Para ser avisada: crie um monitor HTTP gratuito (UptimeRobot, Better Stack)
apontando para `/health/ready`, a cada 5 minutos, com alerta por e-mail. Quando
chegar o alerta, `python3 scripts/evolution_setup.py estado` diz se é sessão
caída (refazer o `qr`) ou algo maior.

## Quando não funcionar

| Sintoma | Causa provável |
|---|---|
| 403 com `assinaturas=nenhuma` | webhook registrado sem a chave na URL — rode `./subir.sh` de novo, que reaponta |
| 403 com `assinaturas=evolution(url)` | `EVOLUTION_API_KEY` diferente da chave da Evolution |
| Nada no log do agente | webhook não registrado na instância — refaça o `instance/create`, ou use `POST /webhook/set/{instancia}` |
| `outcome=respondido` e nada chega | `EVOLUTION_BASE_URL` errada, ou instância desconectada |
| HTTP 404 no envio | nome da instância não confere |
| Pede QR a cada deploy | falta o Disk; a sessão não está persistindo |
| Parou de responder do nada | sessão caiu — veja `connectionState`; se for `close`, reparear. Se repetir, pode ser banimento |

## O que continua igual nos três canais

Guardrails, triagem de emergência, checagem de fundamentação, deduplicação,
limite de frequência e transcrição de áudio. `tests/test_webhook_evolution.py`
fixa isso: uma mensagem dizendo "meu bebê não respira" recebe a orientação de
emergência por este canal também.

Duas proteções são específicas daqui, e ambas têm teste:

- **Eco da própria resposta é ignorado.** Sem isso, cada resposta do agente
  chegaria de volta como mensagem nova e dispararia a seguinte, para sempre.
- **Mensagem de grupo é ignorada.** Bot respondendo em grupo é o caminho mais
  curto para denúncia — e denúncia é o que leva a banimento.
