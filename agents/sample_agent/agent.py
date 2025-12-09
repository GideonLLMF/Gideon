from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI(title="Sample Agent")
class HandleReq(BaseModel):
    text: str
    session_id: str | None = None

    
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/handle")
def handle(req: HandleReq):
    return {"agent": "sample_agent", "reply": f"Echo: {req.text}"}

if __name__ == "__main__":
    uvicorn.run("agent:app", port=8102, reload=True)