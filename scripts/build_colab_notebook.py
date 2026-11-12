"""Generate notebooks/colab_tryon_server.ipynb from the tryon_server/ source files.

The notebook embeds the server code with %%writefile cells, so it runs on Colab without the
repository being public and can never drift from the tested code: always regenerate it after
changing tryon_server/ and commit both.

    .venv\\Scripts\\python scripts\\build_colab_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
SERVER_FILES = ["__init__.py", "catvton_engine.py", "jobs.py", "mock_engine.py", "server.py", "requirements-server.txt"]
CATVTON_URL = "https://github.com/Zheng-Chong/CatVTON.git"
CATVTON_COMMIT = "7818397f25613beedb3d861a34769f607cfcf3b1"   # verified 2026-11-08


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build() -> nbf.NotebookNode:
    cells = [md(f"""
# Virtual Try-On server (CatVTON on a Colab GPU)

This notebook turns a free Colab GPU into the try-on backend for the Streamlit app on your laptop.

**Steps on Google Colab**
1. `Runtime → Change runtime type → T4 GPU → Save`.
2. `Runtime → Run all`. Each new session downloads about 6 GB of model weights (a few minutes).
3. Copy the **URL** and **API key** printed in step 4 into the app's sidebar.
4. Keep this tab open while you use the app. If Colab disconnects, run all cells again (the URL changes).

**Steps on Kaggle** (separate free GPU quota, useful when Colab says "Cannot connect to GPU backend")
1. kaggle.com → `Create → New Notebook` → `File → Import Notebook` → upload this `.ipynb`.
2. Right-hand panel → `Session options`: **Accelerator: GPU T4 x2** (or P100) and **Internet: on**
   (Internet needs a phone-verified Kaggle account).
3. `Run all`, then use the URL and key from step 4 exactly as on Colab.

The server code below is generated from `tryon_server/` in the project repository by
`scripts/build_colab_notebook.py`. Edit the `.py` files there, not here.

CatVTON (Chong et al., ICLR 2025) is used under CC BY-NC-SA 4.0 (non-commercial, academic use).
"""),
        md("## 1. Check the GPU"),
        code("""
import os, sys, torch
os.makedirs("/content", exist_ok=True)   # Colab has it; on Kaggle we create it so the same paths work
print("Python", sys.version.split()[0], "| torch", torch.__version__, "| platform:",
      "Kaggle" if os.path.exists("/kaggle") else "Colab/other")
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Colab: Runtime → Change runtime type → T4 GPU. Kaggle: Session options → Accelerator → GPU. "
                     "Then run all cells again.")
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}, {props.total_memory / 1024**3:.1f} GB, compute capability {props.major}.{props.minor}")
"""),
        md("## 2. Get CatVTON (pinned commit) and write the server code"),
        code(f"""
%cd /content
!rm -rf /content/third_party/CatVTON && mkdir -p /content/third_party
!git clone -q {CATVTON_URL} /content/third_party/CatVTON
!git -C /content/third_party/CatVTON checkout -q {CATVTON_COMMIT} && git -C /content/third_party/CatVTON log -1 --format="CatVTON at %h (%cd)"
!mkdir -p /content/vto/tryon_server
"""),
    ]
    for name in SERVER_FILES:
        source = (ROOT / "tryon_server" / name).read_text(encoding="utf-8")
        cells.append(nbf.v4.new_code_cell(f"%%writefile /content/vto/tryon_server/{name}\n{source}"))  # no strip: byte-identical
    cells += [
        code("""
!pip install -q -r /content/vto/tryon_server/requirements-server.txt
import diffusers, transformers
print("diffusers", diffusers.__version__, "| transformers", transformers.__version__)
"""),
        md("""
## 3. Start the server

Loads CatVTON + DensePose + SCHP onto the GPU. **First run: 3–8 minutes** (downloads).
`SAFETY_CHECK` keeps the Stable Diffusion NSFW filter on; set it to `False` only if it fails to load.
"""),
        code("""
import os, secrets, subprocess, time, requests

SAFETY_CHECK = True
API_KEY = secrets.token_urlsafe(16)
os.chdir("/content/vto")
log_file = open("/content/server.log", "w")
cmd = ["python", "-m", "tryon_server.server", "--catvton-repo", "/content/third_party/CatVTON",
       "--api-key", API_KEY, "--port", "8000"]
if not SAFETY_CHECK:
    cmd.append("--skip-safety-check")
server = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

start = time.time()
while True:
    time.sleep(5)
    if server.poll() is not None:
        print(open("/content/server.log").read()[-4000:])
        raise SystemExit("The server stopped while loading. The log above shows why.")
    try:
        health = requests.get("http://127.0.0.1:8000/health", timeout=5).json()
        break
    except requests.exceptions.RequestException:
        last = open("/content/server.log").read().strip().splitlines()[-1:] or [""]
        print(f"[{time.time() - start:5.0f}s] loading... {last[0][:110]}")
print("Server ready:", health)
"""),
        md("## 4. Open a public HTTPS tunnel (Cloudflare quick tunnel, no account needed)"),
        code("""
import re
!wget -q -nc -O /content/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /content/cloudflared
tunnel_log = open("/content/tunnel.log", "w")
tunnel = subprocess.Popen(["/content/cloudflared", "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8000"],
                          stdout=tunnel_log, stderr=subprocess.STDOUT)
PUBLIC_URL = None
for _ in range(60):
    time.sleep(1)
    found = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", open("/content/tunnel.log").read())
    if found:
        PUBLIC_URL = found.group(0)
        break
if PUBLIC_URL is None:
    print(open("/content/tunnel.log").read()[-3000:])
    raise SystemExit("Tunnel did not start; run this cell again.")
for _ in range(30):   # DNS for a new quick tunnel can take a few seconds
    try:
        if requests.get(PUBLIC_URL + "/health", timeout=10).ok:
            break
    except requests.exceptions.RequestException:
        pass
    time.sleep(2)
print("=" * 70)
print("  Paste these into the app sidebar:")
print(f"  URL     : {PUBLIC_URL}")
print(f"  API key : {API_KEY}")
print("=" * 70)
"""),
        md("""
## 5. Self-test: first real try-on on CatVTON's own demo images

Runs through the same HTTP API the laptop uses, and reports timing and peak GPU memory.
These numbers go into `docs/test_report.md`.
"""),
        code("""
import base64, io, glob
from PIL import Image
from IPython.display import display

def run_job(person_path, garment_path, garment_type="upper", steps=30, seed=42):
    files = {"person": open(person_path, "rb"), "garment": open(garment_path, "rb")}
    data = {"garment_type": garment_type, "steps": steps, "seed": seed}
    headers = {"X-API-Key": API_KEY}
    job = requests.post("http://127.0.0.1:8000/jobs", files=files, data=data, headers=headers).json()
    while True:
        time.sleep(1)
        status = requests.get(f"http://127.0.0.1:8000/jobs/{job['job_id']}", headers=headers).json()
        if status["status"] in ("done", "error"):
            return status

demo = "/content/third_party/CatVTON/resource/demo/example"
people = sorted(glob.glob(f"{demo}/person/*/*"))
uppers = sorted(glob.glob(f"{demo}/condition/upper/*"))
overalls = sorted(glob.glob(f"{demo}/condition/overall/*"))
tests = [(people[0], uppers[0], "upper", 30), (people[0], uppers[0], "upper", 50), (people[-1], overalls[0], "overall", 30)]
for person_path, garment_path, garment_type, steps in tests:
    status = run_job(person_path, garment_path, garment_type, steps)
    if status["status"] == "error":
        print("ERROR:", status["error"]); continue
    r = status["result"]
    print(f"{garment_type}, {steps} steps -> timings {r['timings']}, peak GPU memory {r['peak_gpu_memory_gb']} GB")
    tiles = [Image.open(person_path).convert("RGB").resize((384, 512)), Image.open(garment_path).convert("RGB").resize((384, 512)),
             Image.open(io.BytesIO(base64.b64decode(r["mask_png"]))).convert("RGB").resize((384, 512)),
             Image.open(io.BytesIO(base64.b64decode(r["result_png"]))).resize((384, 512))]
    sheet = Image.new("RGB", (384 * 4, 512), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (384 * i, 0))
    display(sheet)
"""),
        md("## 6. Server log (run any time to see what the server is doing)"),
        code("print(open('/content/server.log').read()[-5000:])"),
        md("""
## Troubleshooting
- **App says it cannot reach the server**: Colab disconnected or the tunnel URL changed. Run all cells again and paste the new URL and key.
- **GPU out of memory**: `Runtime → Restart session`, then run all again. Use 30 steps.
- **"Cannot connect to GPU backend" / "No GPU"**: the free Colab GPU quota is used up. Do **not** choose "Connect without GPU"
  (CatVTON on CPU takes many minutes per image). Use Kaggle (steps at the top) or try Colab again later.
- **Kaggle: downloads fail**: Internet is off. Turn it on in Session options (needs phone verification).
- **Safety checker fails to load**: set `SAFETY_CHECK = False` in step 3.
"""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update({
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    })
    return notebook


def main() -> None:
    out = ROOT / "notebooks" / "colab_tryon_server.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.write(build(), out)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
