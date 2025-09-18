import win32api
import win32con
import tkinter as tk
import numpy as np
import ultralytics
import threading
import math
import time
import cv2
import mss


class Config:
    def __init__(self):
        self.width = 1920
        self.height = 1080

        self.center_x = self.width // 2
        self.center_y = self.height // 2

        self.capture_width = 120
        self.capture_height = 170
        self.capture_left = self.center_x - self.capture_width // 2
        self.capture_top = self.center_y - self.capture_height // 2
        self.crosshairX = self.capture_width // 2
        self.crosshairY = self.capture_height // 2

        self.region = {
            "top": self.capture_top,
            "left": self.capture_left,
            "width": self.capture_width,
            "height": self.capture_height + 100,
        }

        # defaults from your script
        self.Running = True
        self.AimToggle = False
        self.Sensitivity = 1
        self.MovementCoefficientX = 0.80
        self.MovementCoefficientY = 0.65
        self.movementSteps = 5
        self.delay = 0.007
        self.radius = 60
        self.rectC = None
        self.fovC = None


config = Config()


def CreateOverlay():
    root = tk.Tk()
    root.title("NAVALEN AIM ASSIST")
    root.geometry("250x150")
    root.configure(bg="black")

    tk.Label(
        root,
        text="⚡ NAVALEN AIM ASSIST ⚡",
        font=("Helvetica", 14, "bold"),
        fg="#9b59b6",
        bg="black",
    ).pack(pady=10)

    def toggleAimbot():
        config.AimToggle = not config.AimToggle
        if config.AimToggle:
            ToggleBtn.config(text="Deactivate Aimbot", bg="red")
        else:
            ToggleBtn.config(text="Activate Aimbot", bg="#9b59b6")

    ToggleBtn = tk.Button(
        root,
        text="Activate Aimbot",
        command=toggleAimbot,
        bg="#9b59b6",
        fg="white",
        relief="flat",
        padx=20,
        pady=10,
    )
    ToggleBtn.pack(pady=10)

    def quitProgram():
        config.AimToggle = False
        config.Running = False
        root.quit()

    tk.Button(
        root,
        text="Quit",
        command=quitProgram,
        bg="red",
        fg="white",
        relief="flat",
        padx=20,
        pady=5,
    ).pack(pady=5)

    # Overlay window
    overlay = tk.Toplevel(root)
    overlay.geometry(
        f"150x150+{config.center_x - config.radius}+{config.center_y - config.radius}"
    )
    overlay.overrideredirect(True)
    overlay.attributes("-topmost", True)
    overlay.attributes("-transparentcolor", "blue")

    canvas = tk.Canvas(
        overlay, width=150, height=150, bg="blue", bd=0, highlightthickness=0
    )
    canvas.pack()
    config.fovC = canvas.create_oval(
        0, 0, config.radius * 2, config.radius * 2, outline="#9b59b6"
    )
    config.rectC = canvas.create_rectangle(
        config.radius - 10,
        config.radius - 10,
        config.radius + 10,
        config.radius + 10,
        outline="white",
    )

    root.mainloop()


def main():
    model = ultralytics.YOLO("yolov8n.pt")
    screenCapture = mss.mss()

    overlayThread = threading.Thread(target=CreateOverlay, daemon=True)
    overlayThread.start()

    while config.Running:
        time.sleep(0.001)

        if not config.AimToggle:
            time.sleep(0.05)
            continue

        try:
            GameFrame = np.array(screenCapture.grab(config.region))
            GameFrame = cv2.cvtColor(GameFrame, cv2.COLOR_BGRA2BGR)
            results = model.predict(source=GameFrame, conf=0.5, classes=[0], verbose=False, max_det=10)

            if len(results[0].boxes) == 0:
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()

            # Find closest box
            distsm = 99999
            indexMin = 0
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                targetX = (x1 + x2) / 2
                targetY = (y1 + y2) / 2
                moveX = targetX - config.crosshairX
                moveY = targetY - config.crosshairY
                distance = math.sqrt(moveX**2 + moveY**2)
                if distance < distsm:
                    distsm = distance
                    indexMin = i

            # Move toward that target
            x1, y1, x2, y2 = boxes[indexMin]
            targetX = (x1 + x2) / 2
            targetY = (y1 + y2) / 2

            moveX = int((targetX - config.crosshairX) // config.Sensitivity)
            moveY = int((targetY - config.crosshairY) // config.Sensitivity)

            distance = math.sqrt(moveX**2 + moveY**2)
            if distance < config.radius:
                for _ in range(config.movementSteps):
                    win32api.mouse_event(
                        win32con.MOUSEEVENTF_MOVE,
                        int(moveX * config.MovementCoefficientX),
                        int(moveY * config.MovementCoefficientY),
                        0,
                        0,
                    )
                    time.sleep(config.delay)

        except Exception as e:
            print("Error in loop:", e)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
