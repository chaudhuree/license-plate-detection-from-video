# 🎥 Video License Plate Detection

A desktop app that scans a video (CCTV, dashcam, surveillance), detects every
vehicle license plate in it, reads the plate number, and lists the results in a
table that you can export to **CSV** or **JSON**.

- Detects plates with a **YOLOv9** license-plate detector (ONNX)
- Reads text with the **fast-plate-ocr** CCT model (ONNX)
- Tracks each plate across frames and keeps the most reliable reading
  (confidence-weighted voting), so blurry or partial plates improve as the car approaches
- Shows the video live with a box and the plate number drawn on every detected plate
- Tkinter UI — no command-line knowledge needed after setup

> ⚠️ The app does **not guess or reconstruct** plate numbers. It reports only what the
> OCR can read from the pixels. Low-confidence rows are marked in the table.

---

## 🧠 How It Works

1. You pick a video file (almost any format: mp4, webm, avi, mkv, mov, …).
2. Every *N*th frame is passed to the **detector**, which returns plate bounding boxes.
3. Each plate crop is passed to the **OCR**, which returns the text and per-character confidence.
4. A **tracker** matches plates between frames (IoU + position), so one physical plate = one row.
5. For each plate, the text with the highest accumulated confidence wins.
6. The live preview and the table update while the video is processed.
7. Save the table as CSV or JSON when you are done.

---

## 📁 Project Structure

```
video-license-plate-detection/
├── app.py            # Tkinter UI (video preview, table, save CSV/JSON)
├── engine.py         # Detector + OCR + multi-frame tracking
├── requirements.txt  # Python dependencies
├── code.py           # Original OpenCV/Tesseract script (kept for reference)
└── README.md
```

---

## 💻 Requirements

| Item | Version / Notes |
|------|-----------------|
| Operating system | Windows 10/11 (the instructions below are for Windows; Linux/macOS work the same way with the equivalent commands) |
| Python | **3.12** (recommended). `onnxruntime` does not yet support Python 3.13+ on all platforms, so use 3.12 |
| RAM | 4 GB or more |
| GPU | Not required. CPU works; NVIDIA GPU is optional and faster if you install `onnxruntime-gpu` |
| Disk | About 100 MB for packages and models |

The models (about 35 MB total) are downloaded automatically the first time you run the app.

---

## 🛠️ Installation (Windows, step by step)

### Step 1 — Install Python 3.12

Either download it from <https://www.python.org/downloads/release/python-31210/>
or use winget in PowerShell:

```powershell
winget install --id Python.Python.3.12 -e
```

On the Python installer screen, tick **"Add python.exe to PATH"** if you install it manually.

Verify:

```powershell
py -3.12 --version
```

### Step 2 — Get the project

Clone the repository (requires Git, <https://git-scm.com>):

```powershell
git clone https://github.com/Ahsan483/video-license-plate-detection.git
cd video-license-plate-detection
```

### Step 3 — Create a virtual environment

Creating a virtual environment keeps the project's packages separate from the rest of your PC.

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

> If PowerShell reports that scripts are disabled, run this once and try again:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
>
> Or skip activation entirely and call the venv Python directly: `.\.venv\Scripts\python`.

Your prompt should now start with `(.venv)`.

### Step 4 — Install the packages

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---------|---------|
| `open-image-models` | Loads the YOLOv9 license-plate detector |
| `fast-plate-ocr[onnx]` | Reads plate text (includes ONNX Runtime) |
| `opencv-python` | Video decoding, drawing, image processing |
| `numpy` | Array operations |
| `pillow` | Converts frames for the Tkinter preview |

Installation takes a few minutes. Check that everything imported correctly:

```powershell
python -c "import cv2, onnxruntime, open_image_models, fast_plate_ocr; print('OK', cv2.__version__)"
```

You should see `OK` followed by a version number.

### Optional — NVIDIA GPU acceleration

If you have an NVIDIA GPU with CUDA drivers installed:

```powershell
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu
```

The app automatically uses the GPU when it is available.

---

## ▶️ Running the App

From the project folder, with the virtual environment active:

```powershell
python app.py
```

To run without a console window:

```powershell
.\.venv\Scripts\pythonw.exe app.py
```

### First run

The first start downloads two models (detector ≈ 27 MB, OCR ≈ 5 MB) into your user
cache. The status bar shows "Loading models…" during this time. Later runs start quickly.

---

## 🧭 Using the App

1. Click **Browse…** and choose a video. All video formats are listed under "All files";
   the "Video" filter shows common extensions (mp4, webm, avi, mkv, mov, …).
2. Adjust the settings if needed:
   - **Detect every Nth frame** — default `2`. Use `1` to check every frame (slower and
     more thorough); raise it (e.g. `5`) to speed things up.
   - **Min. detection confidence** — default `0.25`. Lower it (e.g. `0.15`) to catch more
     faint or distant plates, at the cost of more false positives.
3. Click **Start**. The video plays in the preview with a coloured box and the plate
   text on each detected plate. The table fills in as plates are found.
4. Click **Stop** to end early, or wait for the video to finish.
5. Click **Save CSV** or **Save JSON** to export the table.
6. Click **Clear** to empty the table before processing another video.

### Table columns

| Column | Meaning |
|--------|---------|
| ID | Track number of the physical plate in the video |
| Plate | Best reading of the plate text |
| Confidence % | Average OCR confidence of the chosen reading |
| First frame | Frame number where the plate was first seen |
| Time (s) | `First frame ÷ video FPS` — the time in the video |
| Sightings | How many frames the plate was detected in |

### Export format

**CSV** (opens in Excel):

```csv
id,plate,confidence,frame,time_sec,detections
1,ABC1234,98.7,152,5.07,14
```

**JSON**:

```json
[
  {
    "id": 1,
    "plate": "ABC1234",
    "confidence": 98.7,
    "frame": 152,
    "time_sec": 5.07,
    "detections": 14
  }
]
```

---

## ⚙️ Tuning Tips

| Problem | Try |
|---------|-----|
| Missing distant or small plates | Lower *Min. detection confidence* to `0.15`; set *Detect every Nth frame* to `1` |
| Too many wrong readings | Raise confidence to `0.4`; check the Confidence % column and ignore low values |
| Processing is slow | Raise *Detect every Nth frame* to `3`–`5`; use the GPU option |
| Video does not open | Re-encode it with a standard codec (e.g. H.264 mp4) — see below |

**Re-encoding a video with ffmpeg** (only if a file will not open):

```powershell
winget install --id Gyan.FFmpeg -e
ffmpeg -i input.webm -c:v libx264 -crf 20 output.mp4
```

---

## 🧩 Troubleshooting

| Error / symptom | Fix |
|-----------------|-----|
| `python` is not recognised | Reinstall Python with "Add to PATH", or use `py -3.12` |
| `No matching distribution found for onnxruntime` | You are on Python 3.13+. Create the venv with `py -3.12` |
| `Could not open video (unsupported codec?)` | Re-encode the video (see above) |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Model download fails | Check your internet connection and run the app again |
| Window opens but is black | Wait a few seconds; the first frame needs the models loaded |

---

## 📜 Original Script

`code.py` is the original OpenCV + Tesseract prototype (contour-based, one frame at a time,
prints plates to the console). It is kept for reference; the app above replaces it.

---

## ⚠️ Limitations

- Results depend on video quality: very small, heavily blurred or motion-smeared plates
  may be missed or misread.
- Plates are read as printed text; non-Latin scripts and very unusual fonts are not guaranteed.
- Only the OCR's own confidence is shown; there is no manual verification step.
- Use this tool only on footage you are allowed to analyse.

---

## 🙏 Credits

- [open-image-models](https://pypi.org/project/open-image-models/) — YOLOv9 license-plate detector
- [fast-plate-ocr](https://pypi.org/project/fast-plate-ocr/) — plate OCR models
- OpenCV, NumPy, Pillow, Tkinter
