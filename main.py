from fastapi import FastAPI, Request

app = FastAPI()

VERIFY_TOKEN = "aleitamento_token"

@app.get("/webhook")
def verify(hub_mode: str = None, hub_verify_token: str = None, hub_challenge: str = None):
    if hub_verify_token == VERIFY_TOKEN:
        return int(hub_challenge)
    return "error"

@app.post("/webhook")
async def receive_message(request: Request):
    data = await request.json()
    print(data)
    return {"status": "ok"}
