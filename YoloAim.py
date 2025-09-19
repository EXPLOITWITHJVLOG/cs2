import win32api
import win32con
import tkinter as tk
import threading
import time
import math
import numpy as np
import cv2
import mss
import ultralytics
import torch


class Config:
    def __init__(self):
        # Screen resolution
        self.width = 1920
        self.height = 1080

        # Crosshair center
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        # Default settings (centered)
        self.offset_x = 0
        self.offset_y = 0
        self.Sensitivity = 1.20
        self.delay = 0.003
        self.MovementCoefficientX = 0.50
        self.MovementCoefficientY = 0.50
        self.FOV = 60
        self.movementSteps = 1

        # Capture region
        self.capture_size = 300
        self.capture_left = self.center_x - self.capture_size // 2
        self.capture_top = self.center_y - self.capture_size // 2
        self.region = {
            "top": self.capture_top,
            "left": self.capture_left,
            "width": self.capture_size,
            "height": self.capture_size,
        }

        # Runtime flags
        self.Running = True
        self.AimEnabled = True


config = Config()


# === UI PANEL ===
def CreateUIPanel():
    root = tk.Tk()
    root.title("N.A.S")
    root.configure(bg="black")
    root.geometry("250x200")

    tk.Label(root, text="N.A.S ASSIST", font=("Helvetica", 16, "bold"),
             fg="lime", bg="black").pack(pady=10)

    def quitProgram():
        config.AimEnabled = False
        config.Running = False
        root.quit()

    def toggleAim():
        config.AimEnabled = not config.AimEnabled
        aim_label.config(text=f"Aim Enabled: {config.AimEnabled}")

    # Aim toggle button
    aim_label = tk.Label(root, text=f"Aim Enabled: {config.AimEnabled}", fg="white", bg="black")
    aim_label.pack(pady=5)

    tk.Button(root, text="Toggle Aim", command=toggleAim,
              bg="purple", fg="white").pack(pady=5)

    # Quit button
    tk.Button(root, text="Quit", command=quitProgram,
              bg="red", fg="white").pack(pady=20)

    root.mainloop()


# === CIRCLE OVERLAY ===
def CreateOverlay():
    overlay = tk.Tk()
    overlay.overrideredirect(True)
    overlay.attributes("-topmost", True)
    overlay.attributes("-transparentcolor", "black")

    overlay.geometry(f"{config.width}x{config.height}")
    canvas = tk.Canvas(overlay, width=config.width, height=config.height,
                       bg="black", highlightthickness=0)
    canvas.pack()

    # Purple circle for FOV
    r = config.FOV
    cx, cy = config.center_x, config.center_y
    canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline="purple", width=2)

    overlay.mainloop()


# === AIM ASSIST ===
def AimAssist():
    # pick GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    # load YOLO on correct device
    model = ultralytics.YOLO("yolov8n.pt").to(device)
    screen = mss.mss()

    while config.Running:
        time.sleep(0.001)
        if not config.AimEnabled:
            continue

        frame = np.array(screen.grab(config.region))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        results = model.predict(frame, conf=0.5, classes=[0],
                                verbose=False, max_det=5, device=device)
        boxes = results[0].boxes.xyxy

        if len(boxes) == 0:
            continue

        closest = None
        min_dist = 99999
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            tx = int((x1 + x2) / 2) + config.capture_left + config.offset_x
            ty = int((y1 + y2) / 2) + config.capture_top + config.offset_y

            dist = math.hypot(tx - config.center_x, ty - config.center_y)
            if dist < min_dist:
                min_dist = dist
                closest = (tx, ty)

        if closest and min_dist < config.FOV:
            moveX = int((closest[0] - config.center_x) // config.Sensitivity)
            moveY = int((closest[1] - config.center_y) // config.Sensitivity)

            for i in range(config.movementSteps):
                win32api.mouse_event(
                    win32con.MOUSEEVENTF_MOVE,
                    int(moveX * config.MovementCoefficientX),
                    int(moveY * config.MovementCoefficientY),
                    0, 0
                )
                time.sleep(config.delay)


# === MAIN ===
if __name__ == "__main__":
    # UI Panel
    threading.Thread(target=CreateUIPanel, daemon=True).start()

    # Circle Overlay
    threading.Thread(target=CreateOverlay, daemon=True).start()

    # Aim Assist
    AimAssist()
