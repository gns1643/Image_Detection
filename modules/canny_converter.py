import cv2
import os
import svgwrite
import numpy as np
from typing import Tuple, List
from .config import FINAL_TH1, FINAL_TH2, Z_SAFE, Z_DRAW, FEED_RATE, SCALE

def trim_contour_to_midpoint(approx: np.ndarray) -> np.ndarray:
    """
    갔다가 되돌아오는 contour에서 첫 절반(전진 경로)만 추출.
    시작점에서 가장 멀리 떨어진 인덱스까지만 사용.
    """
    if len(approx) < 2:
        return approx

    start = approx[0][0].astype(float)
    
    # 각 점에서 시작점까지 거리 계산
    distances = [
        np.linalg.norm(approx[i][0].astype(float) - start)
        for i in range(len(approx))
    ]
    
    # 가장 먼 점의 인덱스 (= 턴어라운드 포인트)
    turn_idx = int(np.argmax(distances))
    
    # 턴어라운드까지만 사용 (전진 경로만)
    return approx[:turn_idx + 1]

def generate_files_canny(img_blurr: np.ndarray, nc_filepath: str, svg_filepath: str) -> bool:
    """Canny Edge Detection 방식을 사용하여 NC와 SVG 파일을 생성합니다."""
    print(f">> [2단계] 파일 생성 시작 (Threshold: {FINAL_TH1}, {FINAL_TH2})")
    
    # 1. 엣지 검출 (Canny)
    edges = cv2.Canny(img_blurr, FINAL_TH1, FINAL_TH2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # [최적화] 로봇 이동 거리 단축을 위해 윤곽선을 위에서 아래로(Y축 기준) 정렬
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    # 2. SVG 준비
    height, width = img_blurr.shape

    os.makedirs(os.path.dirname(svg_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(nc_filepath), exist_ok=True)
    
    dwg = svgwrite.Drawing(svg_filepath, profile='tiny', size=(width, height), viewBox=f"0 0 {width} {height}")
    
    count = 0
    
    try:
        # 3. G-코드 파일 열기
        with open(nc_filepath, 'w') as f:
            f.write("%\n")
            f.write("G21 (Units: mm)\n")
            f.write("G90 (Absolute)\n")
            f.write(f"G0 Z{Z_SAFE}\n")
            
            for i, contour in enumerate(contours):
                if cv2.arcLength(contour, closed=False) < 15:
                    continue
                
                epsilon = 0.002 * cv2.arcLength(contour, closed=False)
                approx = cv2.approxPolyDP(contour, epsilon, closed=False)
                
                # [NEW] 되돌아오는 경로를 중간점에서 잘라냄
                approx = trim_contour_to_midpoint(approx)
                
                if len(approx) < 2: 
                    continue

                path = [(float(p[0][0]), float(p[0][1])) for p in approx]
                
                # --- [SVG 저장] ---
                dwg.add(dwg.polyline(path, stroke='black', fill='none', stroke_width=2))
                
                # --- [G-코드 저장] ---
                start_p = path[0]
                sx = start_p[0] * SCALE
                sy = start_p[1] * SCALE
                
                f.write(f"G0 X{sx:.3f} Y{-sy:.3f}\n") 
                f.write(f"G1 Z{Z_DRAW} F{FEED_RATE}\n")
                
                for j in range(1, len(path)):
                    px = path[j][0] * SCALE
                    py = path[j][1] * SCALE
                    f.write(f"G1 X{px:.3f} Y{-py:.3f}\n")
                
                f.write(f"G0 Z{Z_SAFE}\n")
                count += 1
            
            f.write("G0 X0 Y0\n")
            f.write("M2\n")
            f.write("%\n")
            
        dwg.save()
        
        print(f">> 생성 완료! 총 {count}개의 획")
        print(f"   1. SVG 파일: {svg_filepath}")
        print(f"   2. NC  파일: {nc_filepath}")
        return True
        
    except Exception as e:
        print(f"[오류] 파일 생성 중 오류 발생: {e}")
        return False