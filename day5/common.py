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
MQTT_TOPIC = "ME193/minifig/chris"   # our own topic (the shared class default is "ME193/minifig")
DRIVE_TOPIC = MQTT_TOPIC + "/drive"    # the UNO Q echoes its decisions here


def list_cameras():
    """[(index, name, is_builtin)] in the order OpenCV numbers them on macOS.

    OpenCV's macOS backend sorts cameras by their system unique ID and uses
    the position in that list as the index, so an iPhone within Continuity
    Camera range can become "camera 0" and push the laptop's own camera to 1.
    This reproduces that ordering so cameras can be chosen by name. Returns
    [] when it can't tell (not macOS, or pyobjc-framework-AVFoundation missing).
    """
    try:
        import AVFoundation as avf
    except ImportError:
        return []
    devs = list(avf.AVCaptureDevice.devicesWithMediaType_(avf.AVMediaTypeVideo) or []) + \
        list(avf.AVCaptureDevice.devicesWithMediaType_(avf.AVMediaTypeMuxed) or [])
    devs.sort(key=lambda d: str(d.uniqueID()))
    out = []
    for i, d in enumerate(devs):
        name, kind = str(d.localizedName()), str(d.deviceType())
        try:
            phone = bool(d.isContinuityCamera())
        except Exception:
            phone = "Continuity" in kind
        builtin = not phone and ("FaceTime" in name or "BuiltIn" in kind)
        out.append((i, name, builtin))
    return out


def open_camera(index=None):
    """Open a camera that actually delivers frames. With no index given, the
    laptop's built-in camera is preferred over a nearby iPhone; macOS also
    opens Continuity Cameras and permission-denied devices 'successfully'
    without ever producing a frame, so each candidate must prove itself."""
    cams = list_cameras()
    names = {i: n for i, n, _ in cams}
    if index is not None:
        indices = [index]
    else:
        builtin = [i for i, _, b in cams if b]
        indices = builtin + [i for i in [CAMERA_INDEX, 0, 1, 2, 3] if i not in builtin]
        indices = list(dict.fromkeys(indices))
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
                print(f"Using camera {idx}" + (f" ({names[idx]})" if idx in names else ""))
                return cap
            time.sleep(0.05)
        print(f"Camera {idx} opened but produced no frames, trying next...")
        cap.release()
    raise RuntimeError("No camera delivered frames. Check System Settings -> "
                       "Privacy & Security -> Camera for your terminal app.")


if __name__ == "__main__":      # python common.py  -> show the camera list
    for i, name, builtin in list_cameras() or [(None, "(camera names unavailable on this system)", False)]:
        print(f"  --camera {i}: {name}" + ("   <- built-in, used by default" if builtin else ""))
