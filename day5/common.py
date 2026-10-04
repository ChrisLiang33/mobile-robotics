"""Shared paths, constants and camera helper for the Day 5 scripts."""

import os
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DAY5_DATA", os.path.join(HERE, "data"))  # override for tests
RAW_DIR = os.path.join(DATA_DIR, "raw")          # captured frames
LABEL_DIR = os.path.join(DATA_DIR, "labels")     # YOLO .txt labels (one per image)
DATASET_DIR = os.path.join(DATA_DIR, "dataset")  # train/val split built by train.py
DATASET_YAML = os.path.join(DATA_DIR, "dataset.yaml")
MODEL_PATH = os.environ.get("DAY5_MODEL", os.path.join(HERE, "models", "minifig.pt"))

CLASS_NAMES = ["green_minifig"]   # class 0

CAMERA_INDEX = 0                  # 0 = built-in camera
FRAME_W, FRAME_H = 1280, 720

# MQTT contract with the UNO Q app (from Prof. Rogers' MQTT Minifig Monitor)
MQTT_TOPIC = "ME193/minifig"
DRIVE_TOPIC = "ME193/minifig/drive"   # the UNO Q echoes its decisions here


def open_camera(index=None):
    """Open the first camera that actually delivers frames (macOS opens
    Continuity Cameras and permission-denied devices 'successfully' but
    never produces a frame, so each candidate must prove itself)."""
    indices = [index] if index is not None else \
        [CAMERA_INDEX] + [i for i in range(4) if i != CAMERA_INDEX]
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        for _ in range(40):
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Using camera {idx}")
                return cap
            time.sleep(0.05)
        print(f"Camera {idx} opened but produced no frames, trying next...")
        cap.release()
    raise RuntimeError("No camera delivered frames. Check System Settings -> "
                       "Privacy & Security -> Camera for your terminal app.")
