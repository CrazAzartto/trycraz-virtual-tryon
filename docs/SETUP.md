# Setup guide

Trycraz runs in two places:

- **Your computer** runs the app: the interface, the photo checks, background removal, CLIP and the recommender. A normal laptop is enough, and no GPU is needed.
- **A free cloud GPU** (Kaggle or Google Colab) runs the try-on model. You start it only when you want to generate try-on images.

The Recommendations page works without the GPU server.

## 1. Install the app (once)

### Windows: automatic

Double-click `start_app.bat` in the project folder. If the Python environment does not exist yet, it asks *Set it up now? [Y/n]*. Press Enter and it will:

1. create a `.venv` folder with Python 3.12, using [uv](https://docs.astral.sh/uv/) if it is installed (uv can also download Python 3.12 by itself), and otherwise using Python 3.12 (or 3.11) from [python.org](https://www.python.org/downloads/);
2. install the exact library versions from `requirements.lock.txt`, falling back to `requirements.txt` if that fails;
3. start the app.

This takes about 5 to 15 minutes and downloads 1 to 2 GB. It happens only on the first run. If Python is missing, the window says so; install Python 3.12 (tick **Add python.exe to PATH**) and double-click the file again. To only set up the environment without starting the app, run `start_app.bat --setup-only` in a terminal.

### Manual (any system)

Requirements: Python **3.12**, and a few GB of free disk space for the libraries and the CLIP model, which is downloaded on first use.

```powershell
git clone https://github.com/CrazAzartto/trycraz-virtual-tryon.git   # or unzip the project and open a terminal in it
cd trycraz-virtual-tryon
pip install uv
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.lock.txt
.venv\Scripts\python scripts\check_env.py
```

Without uv, use the standard tools instead:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.lock.txt
```

- `requirements.lock.txt` pins the exact versions we tested on Windows 11 with Python 3.12. If it fails on another system, use `requirements.txt`, which lists the same packages with looser versions.
- On macOS or Linux, replace `.venv\Scripts\python.exe` with `.venv/bin/python`.
- `check_env.py` prints your Python version, memory, disk space and the installed package versions, so you can see that everything was installed.
- On first use the app also downloads the CLIP model (about 600 MB) and the U²-Net model for background removal (about 176 MB).

Optional: run the tests with `.venv\Scripts\python -m pytest -m "not slow"`.

## 2. Start the GPU server

The notebook `notebooks/colab_tryon_server.ipynb` works on both Kaggle and Colab. Each new session downloads about 6 GB of model weights, which takes a few minutes. The model then loads in about 1.5 minutes.

### Option A: Kaggle (recommended; about 30 free GPU hours a week)

1. Sign in at [kaggle.com](https://www.kaggle.com). GPU and Internet access require a **verified phone number** (Settings, then Phone verification).
2. Click **Create**, then **New Notebook**.
3. In the notebook, open **File**, choose **Import Notebook** and upload `notebooks/colab_tryon_server.ipynb`.
4. In the right-hand panel under **Session options**:
   - set **Accelerator** to a **T4** GPU;
   - switch **Internet** on.
5. Click **Run All**.
6. When step 4 of the notebook finishes, it prints two lines:
   ```
   URL:     https://<random-words>.trycloudflare.com
   API key: <random key>
   ```
   Keep the notebook tab open. The server stops when the session stops.
7. When you finish, **stop the session** (the power button at the top right, or Run, then Stop session). Otherwise it keeps using your weekly GPU quota.

### Option B: Google Colab

1. Open [colab.research.google.com](https://colab.research.google.com), choose **File**, then **Upload notebook**, and select `notebooks/colab_tryon_server.ipynb`.
2. Choose **Runtime**, then **Change runtime type**, select **T4 GPU** and save.
3. Choose **Runtime**, then **Run all**.
4. Copy the URL and API key printed by step 4.

If Colab says *Cannot connect to GPU backend*, your free GPU quota is used up for the day. Use Kaggle instead.

## 3. Start the app

**Windows:** double-click `start_app.bat` in the project folder (the first run sets things up, see step 1). Paste the URL and press Enter, then paste the API key and press Enter. Leave both empty to use only the Recommendations page. (In the black window, right-click pastes.) The app opens at http://localhost:8501 with both values filled in.

**Any system:**

```bash
.venv/bin/streamlit run app.py        # Windows: .venv\Scripts\streamlit run app.py
```

Then paste the URL and key into the **Try-on server** section of the sidebar and press **Check connection**. It should say *Connected*.

You can also set them as environment variables, `TRYON_SERVER_URL` and `TRYON_API_KEY`, before starting the app.

## 4. Use it

1. **Recommendations:** choose Men or Women, then the occasion, setting and weather, and press **Recommend**. Open **Why this score** to see the six factor scores. Press **Try this on** to send a garment to the try-on page.
2. **Virtual Try-On:** pick a sample photo or upload your own, pick a catalogue garment or upload one, check the photo report, and press **Generate try-on**. A result takes about 70 to 130 seconds end to end.
3. Drag the slider to compare the result with your photo, and download the result or a comparison sheet.

Every result is saved in `outputs/runs/<timestamp>/` with its inputs and settings. This folder is git-ignored, so your photos never end up in the repository.

### Command-line try-on

```powershell
.venv\Scripts\python scripts\tryon_cli.py --url <URL> --key <KEY> --person assets\sample_people\man_studio.jpg --garment data\catalogue\images\cd_002.jpg --type upper
```

## Troubleshooting

| Problem | What to do |
|---|---|
| *Cannot reach the try-on server* | The notebook session stopped, or the URL changed. The URL is new every session, so copy the latest one. |
| *The try-on server rejected the API key* | The key is also new every session. Copy it again from the notebook. |
| The first try-on is slow | The first request warms the model up. Later requests take about 62 s of GPU time each. |
| *Waiting for the GPU* for a while | Another try-on is running. Jobs run one at a time. |
| Grey patches below the garment | Your photo is probably cut off at the hips. Use a photo that shows you down to at least mid-thigh. The photo check warns about this. |
| Wrong colours leaking into the garment | Keep **Remove garment background** switched on, and photograph the garment on a plain surface. |
| CLIP could not be loaded | The first use downloads about 600 MB. Check your internet connection. Recommendations still work without CLIP. |
| *An Application Control policy has blocked this file* | Windows Smart App Control is blocking a library's DLL. The app does not need the blocked libraries (pandas, pyarrow) for its own features. If another library is blocked, turn Smart App Control off (Windows Security, then App & browser control), or allow the file. |
| `ModuleNotFoundError` | Start the app with the project's `.venv` (or `start_app.bat`), not your system Python. |
| `start_app.bat` setup stopped halfway | Double-click `start_app.bat` again. It deletes the unfinished `.venv` and starts the setup over. |
