from typing import Annotated
from fastapi import FastAPI, Query, Request
from fastapi.responses import PlainTextResponse

app = FastAPI()

VERIFY_TOKEN = "puericultura_token"

@app.get("/webhook", response_class=PlainTextResponse)
def verify_webhook(
    hub_mode: Annotated[str | None, Query(alias="hub.mode")] = None,
    hub_verify_token: Annotated[str | None, Query(alias="hub.verify_token")] = None,
    hub_challenge: Annotated[str | None, Query(alias="hub.challenge")] = None,
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN and hub_challenge:
        return hub_challenge
    return PlainTextResponse("Forbidden", status_code=403)

@app.post("/webhook")
async def receive_webhook(request: Request):
    data = await request.json()
    print("EVENTO RECEBIDO:")
    print(data)
    return {"status": "ok"}
