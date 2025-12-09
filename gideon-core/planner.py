import requests 
from models.llm import LLM

class Planner:
    def __init__(self,registry):
        self.registry = registry
        self.llm = LLM()

    def handle_text(self,text:str,session_id:str):
        lowered = text.lower()
        if any(k in lowered for k in ["trade","buy","sell","position","simulate"]):
            agent = self.registry.pick_for_intent("trading")
            if not agent:
                return {"error": "no trading agent registered"}
            endpoint = agent["endpoint"].rstrip("/")
            if "simulate" in lowered or "backtest" in lowered:
                url = endpoint + "/simulate"
                payload = {"text": text,"session_id": session_id}
                try:
                    resp = requests.post(url,json= payload,timeout=20).json()
                    return {"from_agent": agent["name","result":resp]}
                except Exception as e:
                    return {"error":f"agent call failed{e}"}
            else:
                prompt = self._build_prompt(text, session_id)
                ans = self.llm.generate(prompt)
                return {"from_llm": True, "reply": ans}
        else:
            prompt = self._build_prompt(text, session_id)
            ans = self.llm.generate(prompt)
            return {"from_llm": True, "reply": ans}
    
    def _build_prompt(self,text,session_id):
        system = "You are Gideon, a local assistant. Be helpful, concise, and safe."
        prompt = f"{system}\n\nSession: {session_id}\nUser: {text}\n\nReply:"
        return prompt