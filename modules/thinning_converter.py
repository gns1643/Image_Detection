import cv2
import numpy as np
import os
import svgwrite
from .config import Z_SAFE, Z_DRAW, FEED_RATE, SCALE

def generate_files_thinning(image: np.ndarray, nc_filepath: str, svg_filepath: str) -> bool:
    """
    세선화(Thinning)된 이미지의 픽셀을 직접 추적하여 NC와 SVG 파일을 생성합니다.
    findContours보다 스케치/일러스트에 더 적합하고, 한 번만 경로를 그리도록 보장합니다.
    """
    print(">> [2단계] 파일 생성 시작 (세선화 방식, 픽셀 추적)")

    # 1. 이미지 전처리 (그레이스케일 및 이진화)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # [수정] 이미지 대비를 극대화하여 연한 선을 진하게 만듦
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # [수정] 임계값을 128에서 230으로 대폭 올려 연한 회색 선도 모두 잡아냄
    _, binary_image = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    
    # 2. 세선화(뼈대 추출)
    thinned = cv2.ximgproc.thinning(binary_image)
    
    # 3. 픽셀 추적을 통해 선분(Path) 추출
    h, w = thinned.shape
    visited = np.zeros((h, w), dtype=bool)
    paths = []
    
    # 8방향 탐색 우선순위 (대각선 포함)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for y in range(h):
        for x in range(w):
            if thinned[y, x] == 255 and not visited[y, x]:
                # 새로운 획(path) 시작
                path = []
                cx, cy = x, y
                
                while (cx, cy) != (-1, -1):
                    visited[cy, cx] = True
                    path.append((float(cx), float(cy)))
                    
                    next_pixel = (-1, -1)
                    # 현재 픽셀 주변의 방문하지 않은 흰색 픽셀 찾기
                    for dx, dy in dirs:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h and thinned[ny, nx] == 255 and not visited[ny, nx]:
                            next_pixel = (nx, ny)
                            break
                    
                    cx, cy = next_pixel

                 # 점이 너무 적은 자잘한 노이즈는 버림
                if len(path) > 3: # 너무 짧은 선은 무시하도록 기준을 살짝 올림
                    # 1. 수집된 픽셀 경로를 OpenCV가 계산할 수 있는 Numpy 배열로 변환
                    path_np = np.array(path, dtype=np.float32)
                    
                    # 2. 경로 단순화 (스무딩)
                    # epsilon 값이 커질수록 선이 둥글고 단순해집니다. (보통 1.0 ~ 2.0 사이가 좋습니다)
                    epsilon = 1.5 
                    approx = cv2.approxPolyDP(path_np, epsilon, closed=False)
                    
                    # 3. 단순화된 경로를 다시 리스트 형태로 변환
                    smoothed_path = [(float(p[0][0]), float(p[0][1])) for p in approx]
                    
                    if len(smoothed_path) > 1:
                        paths.append(smoothed_path)

    # 4. SVG 및 G-코드 생성
    os.makedirs(os.path.dirname(svg_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(nc_filepath), exist_ok=True)
    
    dwg = svgwrite.Drawing(svg_filepath, profile='tiny', size=(w, h), viewBox=f"0 0 {w} {h}")
    count = 0
    
    try:
        with open(nc_filepath, 'w') as f:
            f.write("%\n")
            f.write("G21 (Units: mm)\n")
            f.write("G90 (Absolute)\n")
            f.write(f"G0 Z{Z_SAFE}\n")
            
            for path in paths:
                dwg.add(dwg.polyline(path, stroke='black', fill='none', stroke_width=1))
                
                start_p = path[0]
                sx, sy = start_p[0] * SCALE, start_p[1] * SCALE
                
                f.write(f"G0 X{sx:.3f} Y{-sy:.3f}\n")
                f.write(f"G1 Z{Z_DRAW} F{FEED_RATE}\n")
                
                for j in range(1, len(path)):
                    px, py = path[j][0] * SCALE, path[j][1] * SCALE
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
