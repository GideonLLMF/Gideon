from typing import Dict
from sqlmodel import SQLModel, Field, Session, create_engine, select
from sqlalchemy import JSON,Column

class Agent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    endpoint: str
    capabilities: list[str] = Field(sa_column=Column(JSON))

DATABASE_URL = "sqlite:///gideon.db"
engine = create_engine(DATABASE_URL, echo=True)

def init_db():
    SQLModel.metadata.create_all(engine)

def register_agent(agent_data: dict):
    from sqlmodel import Session
    agent = Agent(**agent_data)
    with Session(engine) as session:
        session.add(agent)
        session.commit()
        session.refresh(agent)
    return agent

def list_agents():
    with Session(engine) as session:
        agents = session.exec(select(Agent)).all()
    return agents