import cv2
import numpy as np
import os
import svgwrite
from scipy.interpolate import splprep, splev
from .config import Z_SAFE, Z_DRAW, FEED_RATE, SCALE

def smooth_path_spline(path, num_points_multiplier=3, smooth_factor=2.0):
    """
    울퉁불퉁한 픽셀 경로를 수학적인 B-Spline(베지어) 곡선으로 부드럽게 변환합니다.
    이 과정을 거치면 픽셀의 계단 현상이 사라지고 완벽한 벡터 곡선이 됩니다.
    """
    pts = np.array(path)
    
    # 중복 점 제거 (Scipy 계산 에러 방지)
    _, idx = np.unique(pts, axis=0, return_index=True)
    pts = pts[np.sort(idx)]
    
    # 점이 4개 미만이면 곡선 생성이 불가능하므로 원본을 반환
    if len(pts) < 4:
        return [(float(p[0]), float(p[1])) for p in pts]

    x = pts[:, 0]
    y = pts[:, 1]

    try:
        # s(smooth_factor): 값이 클수록 선이 더 둥글고 매끄러워집니다.
        # tck: 곡선의 수학적 방정식, u: 매개변수
        tck, u = splprep([x, y], s=smooth_factor)
        
        # 곡선을 그릴 점의 개수를 원래 픽셀보다 늘려서 훨씬 부드럽게 쪼갭니다.
        num_points = max(10, len(pts) * num_points_multiplier)
        u_new = np.linspace(u.min(), u.max(), num_points)
        
        # 새로운 곡선 좌표 추출
        x_new, y_new = splev(u_new, tck)
        return list(zip(x_new, y_new))
    except Exception as e:
        # 곡선화 실패 시 안전하게 원본 경로 반환
        return [(float(p[0]), float(p[1])) for p in pts]


def generate_files_thinning(image: np.ndarray, nc_filepath: str, svg_filepath: str) -> bool:
    """
    세선화(Thinning)된 픽셀을 추출한 뒤, 벡터(Spline) 곡선으로 변환하여 NC와 SVG를 생성합니다.
    """
    print(">> [2단계] 파일 생성 시작 (벡터 곡선 스무딩 방식)")

    # 1. 이미지 전처리 (그레이스케일 및 이진화)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, binary_image = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    
    # 2. 세선화(뼈대 추출)
    thinned = cv2.ximgproc.thinning(binary_image)
    
    # 3. 픽셀 추적을 통해 선분(Path) 추출
    h, w = thinned.shape
    visited = np.zeros((h, w), dtype=bool)
    paths = []
    
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for y in range(h):
        for x in range(w):
            if thinned[y, x] == 255 and not visited[y, x]:
                path = []
                cx, cy = x, y
                
                while (cx, cy) != (-1, -1):
                    visited[cy, cx] = True
                    path.append((float(cx), float(cy)))
                    
                    next_pixel = (-1, -1)
                    for dx, dy in dirs:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h and thinned[ny, nx] == 255 and not visited[ny, nx]:
                            next_pixel = (nx, ny)
                            break
                    
                    cx, cy = next_pixel

                # 너무 짧은 노이즈 점들은 버림 (길이 3 이하)
                if len(path) > 3: 
                    # 1. 원본 픽셀 경로 배열화
                    path_np = np.array(path, dtype=np.float32)
                    
                    # 2. 1차 단순화 (자잘한 지그재그를 펴줌)
                    epsilon = 0.5
                    approx = cv2.approxPolyDP(path_np, epsilon, closed=False)
                    approx_path = [(float(p[0][0]), float(p[0][1])) for p in approx]
                    
                    # 3. [핵심] 스플라인(Spline) 알고리즘으로 매끄러운 곡선 변환!
                    if len(approx_path) >= 4:
                        # smooth_factor를 조절하여 부드러운 정도를 변경할 수 있습니다.
                        smoothed_path = smooth_path_spline(approx_path, smooth_factor=3.0)
                    else:
                        smoothed_path = approx_path
                    
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
                # SVG 저장 (부드러운 곡선 적용)
                dwg.add(dwg.polyline(path, stroke='black', fill='none', stroke_width=1))
                
                # G-코드 저장
                start_p = path[0]
                sx, sy = start_p[0] * SCALE, start_p[1] * SCALE
                
                f.write(f"G0 X{sx:.3f} Y{-sy:.3f}\n")
                f.write(f"G1 Z{Z_DRAW} F{FEED_RATE}\n")
                
                # 수많은 짧은 직선을 이어붙여 로봇에게 완벽한 곡선처럼 움직이게 함
                for j in range(1, len(path)):
                    px, py = path[j][0] * SCALE, path[j][1] * SCALE
                    f.write(f"G1 X{px:.3f} Y{-py:.3f}\n")
                
                f.write(f"G0 Z{Z_SAFE}\n")
                count += 1
            
            f.write("G0 X0 Y0\n")
            f.write("M2\n")
            f.write("%\n")
            
        dwg.save()
        
        print(f">> 생성 완료! 총 {count}개의 부드러운 벡터 획")
        print(f"   1. SVG 파일: {svg_filepath}")
        print(f"   2. NC  파일: {nc_filepath}")
        return True
        
    except Exception as e:
        print(f"[오류] 파일 생성 중 오류 발생: {e}")
        return False