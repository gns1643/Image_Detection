import cv2
import numpy as np
import os
import svgwrite
from scipy.interpolate import splprep, splev  # B-Spline을 위한 SciPy 라이브러리 추가
from .config import Z_SAFE, Z_DRAW, FEED_RATE, SCALE

def generate_files_thinning(binary_image: np.ndarray, nc_filepath: str, svg_filepath: str) -> bool:
    """
    이미 이진화된 이미지를 입력받아 세선화(Thinning)하고, 
    B-Spline 보간법을 적용하여 매우 부드러운 NC/SVG 파일을 생성합니다.
    """
    print(">> [2단계] 파일 생성 시작 (세선화 + B-Spline 곡선 최적화)")

    # 1. 흑백 반전 및 세선화
    inverted_binary = cv2.bitwise_not(binary_image)
    thinned = cv2.ximgproc.thinning(inverted_binary)
    
    # 2. 픽셀 추적(DFS)을 통해 선분(Path) 추출
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

                 # 3. [핵심] 수집된 픽셀 경로에 B-Spline 적용
                if len(path) > 5: 
                    path_np = np.array(path, dtype=np.float32)
                    
                    # 3차 스플라인 곡선을 그리려면 최소 4개의 점이 필요합니다.
                    if len(path_np) > 4:
                        try:
                            # x, y 좌표 분리
                            px = path_np[:, 0]
                            py = path_np[:, 1]

                            # B-Spline 계산
                            # s(평활도): 숫자가 커질수록 픽셀의 각진 부분을 무시하고 부드러운 곡선이 됩니다. (보통 2.0 ~ 5.0 사이가 좋습니다)
                            tck, u = splprep([px, py], s=3.0, k=3)

                            # 부드러워진 곡선을 따라 G코드를 생성할 점들을 새로 찍습니다.
                            # 점의 개수를 원래 픽셀 수의 40%로 줄여 G코드 용량 최적화
                            num_points = max(5, int(len(path_np) * 0.4))
                            u_new = np.linspace(0, 1, num_points)
                            x_new, y_new = splev(u_new, tck)

                            smoothed_path = [(float(nx), float(ny)) for nx, ny in zip(x_new, y_new)]
                            paths.append(smoothed_path)
                            
                        except Exception as e:
                            # 곡선 연산에 실패할 경우(점이 너무 겹쳐있는 등) 원본 경로 유지
                            paths.append([(float(p[0]), float(p[1])) for p in path_np])
                    else:
                        paths.append([(float(p[0]), float(p[1])) for p in path_np])

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
        
        print(f">> 생성 완료! 총 {count}개의 부드러운 획")
        return True
        
    except Exception as e:
        print(f"[오류] 파일 생성 중 오류 발생: {e}")
        return False