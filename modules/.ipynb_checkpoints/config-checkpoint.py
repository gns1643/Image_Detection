import os

# 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "raw")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")

INPUT_FILENAME = "input.png"
OUTPUT_NC_FILENAME = "drawing.nc"
OUTPUT_SVG_FILENAME = "output.svg"

# Canny 엣지 설정
FINAL_TH1 = 50
FINAL_TH2 = 100

# G-코드(로봇) 설정
Z_SAFE = 10.0       # 이동 높이 (mm)
Z_DRAW = 0.0        # 그리기 높이 (mm)
FEED_RATE = 1000    # 속도 (mm/min)
SCALE = 0.2         # 크기 조절 (1픽셀 = 0.2mm)