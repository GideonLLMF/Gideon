import os

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

class LLM:
    def __init__(self):
        self.key = OPENAI_KEY
        if self.key:
            try:
                import openai
                self.client = openai
                self.client.api_key = self.key
            except Exception:
                self.client = None
        else:
            self.client = None

    def generate(self, prompt: str, max_tokens: int = 256)-> str:
        if self.client:
            try:
                resp = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"system","content":"You are Gideon."},
                {"role":"user","content":prompt}],
                max_tokens=max_tokens,
                temperature=0.2,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                return f"LLM call failed: {e}"
        return f"(mock reply) I understood: {prompt[:200]}"