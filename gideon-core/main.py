from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from registry import AgentRegistry
from planner import Planner
import os
import uuid


app = FastAPI(title="Gideon Core - MVP")
registry = AgentRegistry()
Planner = Planner(registry)

class Query(BaseModel):
    text: str
    session_id: str | None = None

@app.post("/register")
def register_agent(manifest: dict):
    try:
        name = manifest.get('name')
        if not name:
            raise HTTPException(status_code=400,detail = 'missing name')
        registry.register(manifest)
        return {"status": "ok","registered": name}
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))

@app.post("/agnets")
def list_agents():
    return register.list_all()

@app.post("/ask")
def ask(q:Query):
    session_id = q.session_id or str(uuid.uuid4())
    result = planner.handle_text(q.text,session_id = session_id)
    return result

@app.get("/health")
def health():
    return {"status": "ok","agents": len(registry.list_all())}

