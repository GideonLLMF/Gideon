# Security notes (starter)
This file summarizes starter security practices you should adopt before enabling 
dangerous capabilities (like trading execution).- **Secrets**: do not store API keys in source control. Use Vault or environment 
variables and encrypted secrets for CI.- **Agent permissions**: each agent must declare `requires_permissions`. Gideon 
must enforce RBAC and require human confirmation for privileged actions.- **Sandboxing**: run untrusted agents in containers with limited capabilities 
and network access.- **Auditing**: log all actions and keep append-only audit logs for at least 90 
days.- **Trading**: `execute` endpoints must require multi-factor confirmation and be 
disabled by default.