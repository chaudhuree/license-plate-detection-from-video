"""Tkinter UI: live video preview with plate boxes + labels, results table, CSV/JSON export."""
import csv
import json
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

COLUMNS = ("id", "plate", "confidence", "frame", "time_sec", "detections")
HEADINGS = {"id": "ID", "plate": "Plate", "confidence": "Confidence %",
            "frame": "First frame", "time_sec": "Time (s)", "detections": "Sightings"}
WIDTHS = {"id": 40, "plate": 140, "confidence": 90, "frame": 80, "time_sec": 70, "detections": 70}
DISPLAY_W, DISPLAY_H = 820, 460


def draw_label(img, box, text, color, scale):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, int(2 * scale)))
    if text:
        fs, th = 0.9 * scale, max(2, int(2 * scale))
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        ty = y1 - 6 if y1 - tht - 12 > 0 else y2 + tht + 10
        cv2.rectangle(img, (x1, ty - tht - 6), (x1 + tw + 8, ty + 6), color, -1)
        cv2.putText(img, text, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video License Plate Detection")
        self.geometry("860x860")
        self.q = queue.Queue()
        self.stop_flag = threading.Event()
        self.reader = None
        self.rows = {}  # track id -> dict

        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        self.path_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(top, text="Browse…", command=self.browse).pack(side="left", padx=4)
        self.start_btn = ttk.Button(top, text="Start", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(top, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)

        opts = ttk.Frame(self, padding=(8, 0))
        opts.pack(fill="x")
        ttk.Label(opts, text="Detect every Nth frame:").pack(side="left")
        self.step_var = tk.IntVar(value=2)
        ttk.Spinbox(opts, from_=1, to=30, width=4, textvariable=self.step_var).pack(side="left", padx=4)
        ttk.Label(opts, text="Min. detection confidence:").pack(side="left", padx=(14, 0))
        self.conf_var = tk.DoubleVar(value=0.25)
        ttk.Spinbox(opts, from_=0.05, to=0.9, increment=0.05, width=5,
                    textvariable=self.conf_var).pack(side="left", padx=4)

        self.video = tk.Label(self, bg="black", width=DISPLAY_W, height=DISPLAY_H)
        self.video.pack(padx=8, pady=6)
        self.blank = ImageTk.PhotoImage(Image.new("RGB", (DISPLAY_W, DISPLAY_H)))
        self.video.config(image=self.blank, width=DISPLAY_W, height=DISPLAY_H)

        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", padx=8)

        frm = ttk.Frame(self, padding=8)
        frm.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(frm, columns=COLUMNS, show="headings", height=8)
        for c in COLUMNS:
            self.tree.heading(c, text=HEADINGS[c])
            self.tree.column(c, width=WIDTHS[c], anchor="w")
        sb = ttk.Scrollbar(frm, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")

        bottom = ttk.Frame(self, padding=8)
        bottom.pack(fill="x")
        self.status = tk.StringVar(value="Select a video to begin.")
        ttk.Label(bottom, textvariable=self.status).pack(side="left")
        ttk.Button(bottom, text="Clear", command=self.clear).pack(side="right")
        ttk.Button(bottom, text="Save JSON", command=lambda: self.save("json")).pack(side="right", padx=4)
        ttk.Button(bottom, text="Save CSV", command=lambda: self.save("csv")).pack(side="right")

        self.after(30, self.poll)

    # ---------- actions ----------
    def browse(self):
        p = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("All files", "*.*"),
                       ("Video", "*.mp4 *.avi *.mov *.mkv *.webm *.wmv *.flv *.m4v *.mpg *.mpeg "
                                 "*.3gp *.ts *.mts *.m2ts *.ogv *.ogg *.vob *.asf *.divx *.f4v")])
        if p:
            self.path_var.set(p)

    def clear(self):
        self.rows.clear()
        self.tree.delete(*self.tree.get_children())

    def start(self):
        path = self.path_var.get().strip()
        if not os.path.isfile(path):
            messagebox.showerror("Error", "Please select a valid video file.")
            return
        self.clear()
        self.stop_flag.clear()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status.set("Loading models (first run downloads them)…")
        threading.Thread(target=self.run_detection, daemon=True, args=(
            path, max(1, int(self.step_var.get() or 1)), float(self.conf_var.get()))).start()

    def stop(self):
        self.stop_flag.set()

    # ---------- worker ----------
    def run_detection(self, path, step, det_conf):
        cap = None
        n = 0
        try:
            from engine import PlateReader
            if self.reader is None or abs(getattr(self, "_conf", -1) - det_conf) > 1e-9:
                self.reader = PlateReader(det_conf=det_conf)
                self._conf = det_conf
            reader = self.reader
            reader.reset()
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                self.q.put(("error", "Could not open video (unsupported codec?)."))
                return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            colors = [(0, 255, 0), (0, 200, 255), (255, 160, 0), (255, 0, 200), (0, 255, 220)]
            while not self.stop_flag.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                n += 1
                scale = max(1.0, frame.shape[1] / 900)
                if n % step == 0:
                    for t, box, new, changed in reader.process(frame, n):
                        if t.best_text:
                            self.q.put(("row", {
                                "id": t.id, "plate": t.best_text,
                                "confidence": round(t.best_conf * 100, 1), "frame": t.first_frame,
                                "time_sec": round(t.first_frame / fps, 2) if fps else "",
                                "detections": t.hits}))
                # draw current tracks on every frame so the preview stays smooth
                for t in reader.tracks:
                    if n - t.last_seen <= step * 2:
                        draw_label(frame, t.box, t.best_text, colors[t.id % len(colors)], scale)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hh, ww = rgb.shape[:2]
                f = min(DISPLAY_W / ww, DISPLAY_H / hh)
                rgb = cv2.resize(rgb, (int(ww * f), int(hh * f)))
                if self.q.qsize() < 3:  # drop preview frames if UI lags
                    self.q.put(("frame", rgb))
                if n % 5 == 0:
                    self.q.put(("progress", n, total))
        except Exception as e:
            self.q.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            if cap is not None:
                cap.release()
            self.q.put(("done", n, self.stop_flag.is_set()))

    # ---------- UI updates ----------
    def poll(self):
        try:
            for _ in range(40):
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "frame":
                    self._img = ImageTk.PhotoImage(Image.fromarray(msg[1]))
                    self.video.config(image=self._img)
                elif kind == "row":
                    r = msg[1]
                    vals = tuple(r[c] for c in COLUMNS)
                    iid = str(r["id"])
                    if iid in self.rows:
                        self.tree.item(iid, values=vals)
                    else:
                        self.tree.insert("", "end", iid=iid, values=vals)
                        self.tree.see(iid)
                    self.rows[iid] = r
                elif kind == "progress":
                    _, n, total = msg
                    if total:
                        self.progress.config(maximum=total, value=n)
                    self.status.set(f"Frame {n}/{total or '?'} — {len(self.rows)} plate(s) found")
                elif kind == "error":
                    messagebox.showerror("Error", msg[1])
                elif kind == "done":
                    self.start_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    self.status.set(f"{'Stopped' if msg[2] else 'Finished'}: {msg[1]} frames, "
                                    f"{len(self.rows)} plate(s) found")
        except queue.Empty:
            pass
        self.after(15, self.poll)

    def save(self, fmt):
        if not self.rows:
            messagebox.showinfo("Nothing to save", "No detected plates yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}", initialfile=f"plates.{fmt}", filetypes=[(fmt.upper(), f"*.{fmt}")])
        if not path:
            return
        data = [self.rows[k] for k in sorted(self.rows, key=int)]
        if fmt == "csv":
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS)
                w.writeheader()
                w.writerows(data)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        self.status.set(f"Saved {len(data)} rows to {path}")


if __name__ == "__main__":
    App().mainloop()
