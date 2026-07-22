# Deploying to Oracle Cloud Always Free

Target: an **Always-Free Ampere A1** VM (arm64, up to 2 OCPU / 12 GB RAM as of
June 2026 — plenty for this stack). The CI `Docker` workflow builds an **arm64**
image and pushes it to GHCR on every merge to `main`; the VM just pulls and runs.

## 1. Create the VM

1. OCI console → **Compute → Instances → Create instance**.
2. Image/shape: **Canonical Ubuntu 22.04**, shape **VM.Standard.A1.Flex**,
   set **2 OCPU / 12 GB** (or 1/6 to be safe on capacity).
3. Add your SSH public key. Create.
4. **"Out of capacity"?** The free A1 pool is oversubscribed. Retry in another
   Availability Domain, pick a large 3-AD region (Ashburn, London), or script the
   retry. This is the single most common blocker — it's capacity, not your setup.

## 2. Open the port — BOTH firewalls (the #1 gotcha)

Publishing the UI on `:8501` needs **two** independent layers opened:

**(a) Cloud firewall — VCN Security List:**
Networking → your VCN → Subnet → Security List → **Add Ingress Rule**:
Source `0.0.0.0/0`, IP Protocol `TCP`, Destination port `8501`.

**(b) OS firewall — iptables:** Oracle's Ubuntu image ships a default `REJECT`
rule that blocks everything but SSH. Opening only (a) will still look dead.

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8501 -j ACCEPT
sudo netfilter-persistent save     # persist across reboots
```

## 3. Install Docker

```bash
sudo apt-get update && sudo apt-get install -y ca-certificates curl
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker      # run docker without sudo
```

## 4. Make the GHCR image pullable

Simplest: on GitHub, open the pushed package (**Packages** → the image) →
**Package settings** → set **visibility: Public**. Then no login is needed on the
VM. (If you keep it private, `docker login ghcr.io` with a read-scoped PAT.)

## 5. Configure and run

```bash
mkdir -p ~/pm25 && cd ~/pm25
# fetch just the compose file
curl -fsSLO https://raw.githubusercontent.com/<owner>/<repo>/main/docker-compose.yml

cat > .env <<'EOF'
IMAGE=ghcr.io/<owner>/<repo>:latest
OPENAI_API_KEY=sk-...
OPENAQ_API_KEY=...
# optional tracing:
# LANGFUSE_PUBLIC_KEY=pk-lf-...
# LANGFUSE_SECRET_KEY=sk-lf-...
# LANGFUSE_HOST=https://cloud.langfuse.com
EOF
chmod 600 .env          # keys stay readable only by you; .env is never committed

docker compose pull
docker compose up -d
```

## 6. Verify

```bash
docker compose ps                 # api should show (healthy) after ~1 min warmup
docker compose logs -f api        # watch the model/Chroma load on first boot
docker compose exec api python -c \
  "import urllib.request;print(urllib.request.urlopen('http://localhost:8000/health').read())"
```

Then open **`http://<VM-public-IP>:8501`** in a browser — the Streamlit UI, which
calls the API over the internal compose network. The API itself is **not**
published (no `ports:` on the `api` service); only the UI is exposed.

## 7. Updating

After a new merge to `main` (CI rebuilds + pushes):

```bash
cd ~/pm25 && docker compose pull && docker compose up -d
```

## Notes / contingencies

- **First boot is slow** (~1 min): the API loads TensorFlow + Torch + Chroma at
  startup. The healthcheck `start-period` allows for it; `depends_on: service_healthy`
  holds the UI until the API is ready.
- **TensorFlow on arm64:** the build uses the `tensorflow-cpu` aarch64 wheel. If a
  future version fails to resolve for arm64, the fallback is to convert
  `models/best_lstm.keras` to **TFLite** and swap the three model calls in
  `src/forecasting.py` to `tflite-runtime` (tiny, rock-solid on arm64). The rest of
  the stack (Torch, chromadb, onnxruntime, pymupdf) all ship aarch64 wheels.
- **Memory:** peak resident is ~1.5–2.5 GB (API) + ~0.2 GB (UI) — comfortable on 12 GB.
