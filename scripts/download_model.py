"""
Downloads pose_landmarker_heavy.task from Google's MediaPipe storage
and saves it to the models/ directory.

Usage:
    python scripts/download_model.py
"""

import os
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)
DEST_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
DEST_FILE = os.path.join(DEST_DIR, "pose_landmarker_heavy.task")


def download():
    os.makedirs(DEST_DIR, exist_ok=True)

    if os.path.exists(DEST_FILE):
        print(f"Model already exists at {DEST_FILE} — skipping download.")
        return

    print(f"Downloading model from:\n  {MODEL_URL}")
    print(f"Saving to:\n  {DEST_FILE}\n")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = "#" * int(pct / 2)
            print(f"\r  [{bar:<50}] {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(MODEL_URL, DEST_FILE, reporthook=progress)
    print(f"\nDone. Model saved to {DEST_FILE}")


if __name__ == "__main__":
    download()
