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

⚠️ Nome de imagem, variáveis e rotas abaixo vêm do que se conhece da v2 e **não
foram conferidos contra a documentação corrente** — confira em
`doc.evolution-api.com` antes. Se algo divergir, o erro aparece na resposta da
própria Evolution.

```
Imagem:  atendai/evolution-api:latest
Porta:   8080
```

Variáveis:

```
AUTHENTICATION_API_KEY = <gere uma chave forte; é a senha do serviço>
```

**Acrescente um Disk** (Render → Disks), montado onde a Evolution guarda a
sessão. Sem isso, cada deploy perde o pareamento e pede QR de novo — o sistema
de arquivos do Render é efêmero. Alternativa: apontar a Evolution para Postgres
ou Redis, que ela suporta.

### Criar a instância e parear

```bash
EVO='https://sua-evolution.onrender.com'
CHAVE='a-chave-que-voce-gerou'

# cria a instância já apontando o webhook para o agente
curl -s -X POST "$EVO/instance/create" \
  -H "apikey: $CHAVE" -H 'Content-Type: application/json' \
  -d '{
    "instanceName": "lactai",
    "qrcode": true,
    "integration": "WHATSAPP-BAILEYS",
    "webhook": {
      "url": "https://SEU-APP.onrender.com/webhook",
      "byEvents": false,
      "events": ["MESSAGES_UPSERT"]
    }
  }' | python3 -m json.tool
```

A resposta traz o QR em base64. Para abrir:

```bash
curl -s "$EVO/instance/connect/lactai" -H "apikey: $CHAVE" \
  | python3 -c "import sys,json,base64,pathlib; d=json.load(sys.stdin); \
b=d.get('base64','').split(',')[-1]; pathlib.Path('qr.png').write_bytes(base64.b64decode(b)); print('qr.png')"
open qr.png
```

No celular do **segundo número**: WhatsApp → Aparelhos conectados → Conectar um
aparelho → aponte para o QR.

Confirme: `curl -s "$EVO/instance/connectionState/lactai" -H "apikey: $CHAVE"`
deve dizer `open`.

### Apontar o agente

No Render, no serviço do agente:

```
EVOLUTION_BASE_URL = https://sua-evolution.onrender.com
EVOLUTION_API_KEY  = a-chave-que-voce-gerou
EVOLUTION_INSTANCE = lactai
```

A `EVOLUTION_API_KEY` tem papel duplo: o agente a envia ao chamar a Evolution, e
a confere no header dos webhooks que ela manda. Por isso precisa ser a mesma dos
dois lados.

### Conversar

De outro celular, mande mensagem para o número pareado. Deve chegar a
apresentação do serviço e depois a resposta.

## Quando não funcionar

| Sintoma | Causa provável |
|---|---|
| 403 no log do agente | `EVOLUTION_API_KEY` diferente da chave da Evolution |
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
