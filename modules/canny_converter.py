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
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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

                
                
                approx = trim_contour_to_midpoint(approx)
                
                if len(approx) < 2: continue

                # --- [SVG 저장] ---
                # 웹브라우저 확인용 (원본 좌표 사용)
                points = [(float(p[0][0]), float(p[0][1])) for p in approx]
                dwg.add(dwg.polyline(points, stroke='black', fill='none', stroke_width=2))
                
                # --- [G-코드 저장] ---
                # 시작점으로 이동 (G0)
                start_p = approx[0][0]
                # 좌표 변환: Y축 반전 (이미지 좌표계 -> 로봇 좌표계)
                # 필요시 SCALE 앞에 - 부호를 붙이거나 떼세요. 여기선 요청하신 대로 Y에 - 붙임.
                sx = float(start_p[0]) * SCALE
                sy = float(start_p[1]) * SCALE
                
                f.write(f"G0 X{sx:.3f} Y{-sy:.3f}\n") 
                f.write(f"G1 Z{Z_DRAW} F{FEED_RATE}\n") # 펜 내리기 (G1)
                
                # 경로 따라 그리기 (G1)
                for i in range(1, len(approx)):
                    p = approx[i][0]
                    px = float(p[0]) * SCALE
                    py = float(p[1]) * SCALE
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