#!/usr/bin/env bash
# Sobe o agente e a Evolution, cria a instância, e mostra o QR de pareamento.
#
#   cd evolution && ./subir.sh
#
# Pede só o que não dá para adivinhar: a chave da OpenAI, na primeira vez.
# Rodar de novo é seguro — reaproveita o que já existe e volta a mostrar o QR
# se o número ainda não estiver pareado.

set -euo pipefail
cd "$(dirname "$0")"

INSTANCIA="${EVOLUTION_INSTANCE:-lactai}"
EVO_LOCAL="http://localhost:8080"
AGENTE_LOCAL="http://localhost:8000"
# Endereço que a Evolution enxerga de dentro da rede do Docker — não é o
# mesmo que você abre no navegador.
WEBHOOK_INTERNO="http://agente:10000/webhook"

passo() { printf "\n\033[1m%s\033[0m\n" "$*"; }
erro()  { printf "\n\033[31m%s\033[0m\n" "$*" >&2; }

# ── 0. pré-requisitos ────────────────────────────────────────────────────
if ! docker info >/dev/null 2>&1; then
  erro "O Docker não está rodando."
  echo "Abra o Docker Desktop, espere a baleia ficar verde e rode de novo."
  exit 1
fi

# ── 1. .env ──────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
  passo "Primeira vez: criando .env"
  cp .env.exemplo .env

  chave=$(python3 -c "import secrets; print(secrets.token_hex(24))")
  # A mesma chave serve para enviar e para conferir o webhook que chega.
  python3 - "$chave" <<'PY'
import pathlib, sys
p = pathlib.Path(".env")
p.write_text(p.read_text().replace("AUTHENTICATION_API_KEY=", f"AUTHENTICATION_API_KEY={sys.argv[1]}"))
PY
  echo "  chave da Evolution gerada"

  printf "  cole a sua OPENAI_API_KEY: "
  read -r openai
  python3 - "$openai" <<'PY'
import pathlib, sys
p = pathlib.Path(".env")
p.write_text(p.read_text().replace("OPENAI_API_KEY=", f"OPENAI_API_KEY={sys.argv[1]}"))
PY
  echo "  .env pronto"
fi

set -a; source .env; set +a
CHAVE="$AUTHENTICATION_API_KEY"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  erro "OPENAI_API_KEY está vazia no .env. Preencha e rode de novo."
  exit 1
fi

# ── 2. subir ─────────────────────────────────────────────────────────────
passo "Subindo os contêineres (a primeira vez baixa as imagens, demora)"
docker compose up -d --build

passo "Esperando o agente responder"
for i in $(seq 1 60); do
  if curl -sf "$AGENTE_LOCAL/health" >/dev/null 2>&1; then
    echo "  agente no ar"
    break
  fi
  [[ $i -eq 60 ]] && { erro "O agente não subiu. Veja: docker compose logs agente"; exit 1; }
  sleep 2
done

passo "Esperando a Evolution responder"
for i in $(seq 1 60); do
  if curl -sf -H "apikey: $CHAVE" "$EVO_LOCAL/instance/fetchInstances" >/dev/null 2>&1; then
    echo "  evolution no ar"
    break
  fi
  [[ $i -eq 60 ]] && { erro "A Evolution não subiu. Veja: docker compose logs evolution"; exit 1; }
  sleep 2
done

curl -s "$AGENTE_LOCAL/health" | python3 -m json.tool

# ── 3. instância ─────────────────────────────────────────────────────────
passo "Criando a instância '$INSTANCIA'"
resposta=$(curl -s -X POST "$EVO_LOCAL/instance/create" \
  -H "apikey: $CHAVE" -H 'Content-Type: application/json' \
  -d "{\"instanceName\":\"$INSTANCIA\",\"qrcode\":true,
       \"integration\":\"WHATSAPP-BAILEYS\",
       \"webhook\":{\"url\":\"$WEBHOOK_INTERNO\",\"byEvents\":false,
                    \"events\":[\"MESSAGES_UPSERT\"]}}" || true)

if echo "$resposta" | grep -qi "already in use\|already exists"; then
  echo "  já existia; reapontando o webhook"
  curl -s -X POST "$EVO_LOCAL/webhook/set/$INSTANCIA" \
    -H "apikey: $CHAVE" -H 'Content-Type: application/json' \
    -d "{\"webhook\":{\"enabled\":true,\"url\":\"$WEBHOOK_INTERNO\",
         \"byEvents\":false,\"events\":[\"MESSAGES_UPSERT\"]}}" >/dev/null || true
else
  echo "$resposta" | python3 -m json.tool 2>/dev/null || echo "$resposta"
fi

# ── 4. pareamento ────────────────────────────────────────────────────────
estado=$(curl -s -H "apikey: $CHAVE" "$EVO_LOCAL/instance/connectionState/$INSTANCIA" || true)
if echo "$estado" | grep -q '"open"'; then
  passo "Já está pareada."
else
  passo "Buscando o QR"
  curl -s -H "apikey: $CHAVE" "$EVO_LOCAL/instance/connect/$INSTANCIA" \
    | python3 -c "
import sys, json, base64, pathlib
try:
    d = json.load(sys.stdin)
except ValueError:
    sys.exit('resposta não era JSON')
b = d.get('base64') or d.get('qrcode', {}).get('base64') or ''
if not b:
    print(json.dumps(d, indent=2, ensure_ascii=False))
    sys.exit('sem QR na resposta')
pathlib.Path('qr.png').write_bytes(base64.b64decode(b.split(',')[-1]))
print('  qr.png salvo')
"
  [[ "$(uname)" == "Darwin" ]] && open qr.png || true

  cat <<'TXT'

  No celular do número que vai atender:
    WhatsApp → Aparelhos conectados → Conectar um aparelho

  O QR expira em cerca de 1 minuto. Se perder, rode este script de novo.
TXT

  printf "\n  esperando o pareamento"
  for i in $(seq 1 90); do
    estado=$(curl -s -H "apikey: $CHAVE" "$EVO_LOCAL/instance/connectionState/$INSTANCIA" || true)
    if echo "$estado" | grep -q '"open"'; then
      printf "\n  pareado ✅\n"
      break
    fi
    printf "."
    sleep 2
  done
fi

# ── 5. pronto ────────────────────────────────────────────────────────────
cat <<TXT

──────────────────────────────────────────────────────────
  Agora mande uma mensagem de WhatsApp para o número pareado,
  de outro celular. Por exemplo:

    como sei se a pega está correta?

  Para ver o que acontece:
    docker compose logs -f agente

  Protótipo web (mesmo agente, sem WhatsApp):
    $AGENTE_LOCAL/chat

  Para desligar tudo:
    docker compose down
──────────────────────────────────────────────────────────
TXT
