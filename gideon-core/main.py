from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from registry import Agent
from planner import Planner
import os
import uuid
from fastapi import FastAPI
from registry import init_db, register_agent, list_agents, Session,engine


app = FastAPI(title="Gideon Core - MVP")
init_db()
registry = Agent()
Planner = Planner(registry)

class Query(BaseModel):
    text: str
    session_id: str | None = None

@app.post("/register")
def register_endpoint(agent_data: dict):
    # Validate required fields
    required = ["name", "endpoint", "capabilities"]
    for field in required:
        if field not in agent_data:
            raise HTTPException(status_code=400, detail=f"{field} is required")

    agent = Agent(**agent_data)
    with Session(engine) as session:
        session.add(agent)
        session.commit()
        session.refresh(agent)

    return {"status": "ok", "registered": agent.name}

@app.post("/ask")
def ask(q:Query):
    session_id = q.session_id or str(uuid.uuid4())
    result = Planner.handle_text(q.text,session_id = session_id)
    return result

@app.get("/health")
def health():
    return {"status": "ok","agents": len(registry.list_all())}

@app.get("/agents")
def list_agents_endpoint():
    return list_agents()
