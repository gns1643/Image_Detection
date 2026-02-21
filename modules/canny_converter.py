import cv2
import os
import svgwrite
import numpy as np
from typing import Tuple, List
from .config import FINAL_TH1, FINAL_TH2, Z_SAFE, Z_DRAW, FEED_RATE, SCALE

def generate_files_canny(img_blurr: np.ndarray, nc_filepath: str, svg_filepath: str) -> bool:
    """Canny Edge Detection 방식을 사용하여 NC와 SVG 파일을 생성합니다."""
    print(f">> [2단계] 파일 생성 시작 (Threshold: {FINAL_TH1}, {FINAL_TH2})")
    
    # 1. 엣지 검출 (Canny)
    edges = cv2.Canny(img_blurr, FINAL_TH1, FINAL_TH2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # [최적화] 로봇 이동 거리 단축을 위해 윤곽선을 위에서 아래로(Y축 기준) 정렬
    # bounding rect: (x, y, w, h) -> y값(c[1])을 기준으로 정렬
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    # 2. SVG 준비
    height, width = img_blurr.shape

    # 출력 폴더가 없으면 생성 (안전장치)
    os.makedirs(os.path.dirname(svg_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(nc_filepath), exist_ok=True)
    
    dwg = svgwrite.Drawing(svg_filepath, profile='tiny', size=(width, height), viewBox=f"0 0 {width} {height}")
    
    count = 0
    
    try:
        # 3. G-코드 파일 열기
        with open(nc_filepath, 'w') as f:
            # 헤더 작성
            f.write("%\n")
            f.write("G21 (Units: mm)\n")
            f.write("G90 (Absolute)\n")
            f.write(f"G0 Z{Z_SAFE}\n") # 안전 높이로 들기
            
            for i, contour in enumerate(contours):
                # 잡티 제거 (너무 짧은 선은 무시)
                if cv2.arcLength(contour, closed=False) < 15:
                    continue
                
                # 단순화 (점 개수 줄이기)
                epsilon = 0.002 * cv2.arcLength(contour, closed=False)
                approx = cv2.approxPolyDP(contour, epsilon, closed=False)
                
                if len(approx) < 2: continue

                # --- [수정된 부분: 되돌아오는 중복 경로 제거 로직] ---
                path = []
                for p in approx:
                    px, py = float(p[0][0]), float(p[0][1])
                    
                    # 현재 획(path)에서 이미 지나간 점들 중, 
                    # 반경 2픽셀 이내로 겹치는 곳(되돌아오는 길)이 있다면 그리지 않음
                    is_duplicate = False
                    for (ex, ey) in path:
                        # 두 점 사이의 거리의 제곱이 4(반경 2픽셀)보다 작으면 중복으로 간주
                        if (px - ex)**2 + (py - ey)**2 < 4.0:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        path.append((px, py))
                
                # 중복을 제거하고 남은 점이 2개 미만이면 선을 그을 수 없으므로 무시
                if len(path) < 2:
                    continue
                # ----------------------------------------------------

                # --- [SVG 저장] ---
                dwg.add(dwg.polyline(path, stroke='black', fill='none', stroke_width=2))
                
                # --- [G-코드 저장] ---
                start_p = path[0]
                sx = start_p[0] * SCALE
                sy = start_p[1] * SCALE
                
                f.write(f"G0 X{sx:.3f} Y{-sy:.3f}\n") 
                f.write(f"G1 Z{Z_DRAW} F{FEED_RATE}\n") # 펜 내리기 (G1)
                
                # 경로 따라 그리기 (G1)
                for j in range(1, len(path)):
                    px = path[j][0] * SCALE
                    py = path[j][1] * SCALE
                    f.write(f"G1 X{px:.3f} Y{-py:.3f}\n")
                
                # 획 끝나면 펜 들기 (G0)
                f.write(f"G0 Z{Z_SAFE}\n")
                count += 1
            
            # 푸터 작성
            f.write("G0 X0 Y0\n")
            f.write("M2\n")
            f.write("%\n")
            
        # SVG 파일 저장
        dwg.save()
        
        print(f">> 생성 완료! 총 {count}개의 획")
        print(f"   1. SVG 파일: {svg_filepath}")
        print(f"   2. NC  파일: {nc_filepath}")
        return True
        
    except Exception as e:
        print(f"[오류] 파일 생성 중 오류 발생: {e}")
        return False