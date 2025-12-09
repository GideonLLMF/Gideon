from typing import Dict

class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str,dict] = {}

    def register(self,manifest:dict):
        name = manifest.get("name")
        if not name:
            raise ValueError("manifest must include a name")
        self._agents[name] = manifest

    def get(self,name:str) -> dict | None:
        return self._agents.get(name)
    
    def list_all(self):
        return list(self._agents.values())
    
    def pick_for_intent(self,intent:str) -> dict | None:
        for m in self._agents.values():
            caps = [c.lower() for c in m.get("capabilities",[])]
            if intent.lower() in caps:
                return m
        return next(iter(self._agents.values()),None)