import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading
import numpy as np

class SketchGUI:
    def __init__(self):
        self.root = None
        self.panels = []
        self.titles = ["1. Raw (Preprocessed)", "2. Gemini Sketch", "3. G-Code (Thinning)"]
        self.images = [None, None, None]
        self.thread = threading.Thread(target=self._run_gui, daemon=True)
        self.thread.start()

    def _run_gui(self):
        self.root = tk.Tk()
        self.root.title("Image Conversion Monitor")
        self.root.geometry("1200x450")
        
        # 메인 프레임 설정
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # 3개의 패널 생성 (이미지 표시용)
        for i in range(3):
            panel_frame = tk.Frame(main_frame, relief=tk.RIDGE, borderwidth=2)
            panel_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
            
            title_label = tk.Label(panel_frame, text=self.titles[i], font=("Arial", 12, "bold"))
            title_label.pack(pady=5)
            
            image_label = tk.Label(panel_frame, text="Waiting...", width=40, height=20, bg="gray80")
            image_label.pack(pady=10, fill=tk.BOTH, expand=True)
            
            self.panels.append(image_label)

        self.root.mainloop()

    def update_image(self, step, cv2_img):
        """
        step: 0(Raw), 1(Sketch), 2(G-Code)
        cv2_img: OpenCV 형식의 이미지 (BGR 또는 Gray)
        """
        if self.root is None or not self.root.winfo_exists():
            return

        try:
            # 1. OpenCV BGR -> RGB 변환
            if len(cv2_img.shape) == 2: # 그레이스케일인 경우
                rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
            else:
                rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

            # 2. 리사이징 (패널 크기에 맞춤)
            h, w = rgb_img.shape[:2]
            target_h = 300
            target_w = int(w * (target_h / h))
            resized_img = cv2.resize(rgb_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # 3. PIL Image로 변환 및 ImageTk 생성
            pil_img = Image.fromarray(resized_img)
            tk_img = ImageTk.PhotoImage(pil_img)

            # 4. GUI 업데이트 (메인 스레드에서 실행되도록 root.after 사용)
            self.root.after(0, self._set_label_image, step, tk_img)
        except Exception as e:
            print(f"[GUI 오류] 이미지 업데이트 실패: {e}")

    def _set_label_image(self, step, tk_img):
        label = self.panels[step]
        label.config(image=tk_img, text="")
        # 가비지 컬렉션 방지를 위해 참조 유지
        label.image = tk_img

    def clear_panels(self):
        """새로운 작업 시작 시 패널 초기화"""
        if self.root:
            for i, label in enumerate(self.panels):
                self.root.after(0, lambda idx=i: self.panels[idx].config(image="", text="Processing..."))
