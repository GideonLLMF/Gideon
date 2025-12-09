import os
import requests
import time

def register_with_gideon(gideon_url: str, manifest: dict, retries: int = 3,delay: float = 1.0):
    url = gideon_url.rstrip("/") + "/register"
    for i in range(retries):
        try:
            resp = requests.post(url, json=manifest, timeout=5)
            resp.raise_for_status()
            print("registered with gideon:", resp.json())
            return True
        except Exception as e:
            print(f"register attempt failed: {e}")
            time.sleep(delay)
    return False

if __name__ == "__main__":
    gideon = os.getenv("GIDEON_URL", "http://localhost:8000")
    import json
    sample_manifest = {
        "name": "sample_agent",
        "version": "0.0.1",
        "capabilities": ["echo"],
        "requires_permissions": [],
        "endpoint": "http://localhost:8102",
        "healthcheck": "/health"
    }
    register_with_gideon(gideon, sample_manifest)