# 3090 Server Setup (run these ON the server)

Goal: get Ollama serving `aya-expanse-8b` on the GPU. Do this once; afterwards
you only pay for server time while you're actively testing.

## 1. Install Ollama (one time)

**Linux server:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
Ollama auto-detects the NVIDIA GPU. Verify the 3090 is seen:
```bash
nvidia-smi          # should list "GeForce RTX 3090", 24GB
```

## 2. Pull the model (one time, ~5 GB download)

```bash
ollama pull aya-expanse        # the 8B version
# (aya-expanse:32b also exists and fits in 24GB quantized, but start with 8B)
```

## 3. Serve it

Ollama runs as a background service after install and listens on
`localhost:11434`. Confirm it's up:
```bash
ollama list                    # should show aya-expanse
curl http://localhost:11434/api/tags
```

That's all the server needs. Leave it running while you work.

---

# Connecting from your Windows machine

Don't expose Ollama to the open internet. Instead, open a secure **SSH tunnel**
so the server's port 11434 appears as `localhost:11434` on your machine. Your
Python code then needs no changes between local and server.

In a PowerShell window on your machine (keep it open while working):
```powershell
ssh -L 11434:localhost:11434 user@your-server-address
```

Now in a second window, run the project:
```powershell
python chat.py
```

When you're done, close the SSH window and shut the server down to stop billing.

## Quick cost-saving workflow
1. Write/edit code locally (free).
2. Start server → open SSH tunnel.
3. Test against the real model.
4. Close tunnel + stop server.
