import cv2
import numpy as np
import os
import svgwrite
from .config import Z_SAFE, Z_DRAW, FEED_RATE, SCALE

def generate_files_thinning(image: np.ndarray, nc_filepath: str, svg_filepath: str) -> bool:
    """
    세선화(Thinning) 방식을 사용하여 NC와 SVG 파일을 생성합니다.
    주로 스케치나 일러스트와 같은 선화 이미지에 적합합니다.
    """
    print(">> [2단계] 파일 생성 시작 (세선화 방식)")

    # 1. 이미지 전처리 (그레이스케일 및 이진화)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 배경이 흰색(255)이고 선이 검은색(0)이므로, 이를 반전시켜 thinning에 적합한 형태로 만듭니다.
    # 즉, 객체(선)를 흰색(255)으로, 배경을 검은색(0)으로 변경합니다.
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # 2. 세선화(뼈대 추출)
    # 2. 세선화(뼈대 추출)
    thinned = cv2.ximgproc.thinning(binary_image)
    
    # -------------------------------------------------------------
    # [수정된 부분] 3. 직접 픽셀을 추적하여 선분(Path) 추출
    # -------------------------------------------------------------
    h, w = thinned.shape
    visited = np.zeros((h, w), dtype=bool)
    paths = []
    
    # 8방향 탐색
    dirs = [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)]
    
    def count_all_neighbors(cx, cy):
        count = 0
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and thinned[ny, nx] == 255:
                count += 1
        return count

    # 끝점(이웃이 1개인 픽셀)을 최우선 시작점으로 찾기
    start_points = []
    for y in range(h):
        for x in range(w):
            if thinned[y, x] == 255 and count_all_neighbors(x, y) == 1:
                start_points.append((x, y))
                
    # 원 모양(폐곡선)처럼 끝점이 없는 경우를 위해 모든 흰색 픽셀도 후보로 추가
    for y in range(h):
        for x in range(w):
            if thinned[y, x] == 255:
                start_points.append((x, y))

    # 픽셀 따라가기 (선 추적)
    for sx, sy in start_points:
        if visited[sy, sx]:
            continue
            
        path = []
        cx, cy = sx, sy
        
        while True:
            path.append((float(cx), float(cy)))
            visited[cy, cx] = True
            
            # 방문하지 않은 이웃 픽셀 찾기
            next_pixel = None
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if thinned[ny, nx] == 255 and not visited[ny, nx]:
                        next_pixel = (nx, ny)
                        break # 첫 번째 발견된 연결 픽셀로 이동
            
            if next_pixel:
                cx, cy = next_pixel
            else:
                break # 더 이상 연결된 길이 없으면 현재 획 종료
                
        # 노이즈 필터링: 점이 너무 적은 자잘한 노이즈는 버림 (원하는 수치로 조절)
        if len(path) > 3:
            paths.append(path)

    # -------------------------------------------------------------
    # 4. SVG 및 G-코드 생성 준비
    # -------------------------------------------------------------
    height, width = image.shape[:2]
    os.makedirs(os.path.dirname(svg_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(nc_filepath), exist_ok=True)
    
    dwg = svgwrite.Drawing(svg_filepath, profile='tiny', size=(width, height), viewBox=f"0 0 {width} {height}")
    
    count = 0
    
    try:
        with open(nc_filepath, 'w') as f:
            f.write("%\n")
            f.write("G21 (Units: mm)\n")
            f.write("G90 (Absolute)\n")
            f.write(f"G0 Z{Z_SAFE}\n")
            
            for path in paths:
                # [삭제됨] 이전의 '되돌아오는 중복 경로 제거 로직'은 이제 필요 없습니다!
                
                # SVG 저장
                dwg.add(dwg.polyline(path, stroke='black', fill='none', stroke_width=1))
                
                # G-코드 저장
                start_p = path[0]
                sx, sy = start_p[0] * SCALE, start_p[1] * SCALE
                
                # 시작점으로 안전하게 이동
                f.write(f"G0 X{sx:.3f} Y{-sy:.3f}\n")
                # 펜 내리기
                f.write(f"G1 Z{Z_DRAW} F{FEED_RATE}\n")
                
                # 선 그리기
                for j in range(1, len(path)):
                    px, py = path[j][0] * SCALE, path[j][1] * SCALE
                    f.write(f"G1 X{px:.3f} Y{-py:.3f}\n")
                
                # 펜 올리기 (다음 획을 위해)
                f.write(f"G0 Z{Z_SAFE}\n")
                count += 1
            
            f.write("G0 X0 Y0\n")
            f.write("M2\n")
            f.write("%\n")
            
        dwg.save()
        # 이하 기존 코드와 동일
        
        print(f">> 생성 완료! 총 {count}개의 획")
        print(f"   1. SVG 파일: {svg_filepath}")
        print(f"   2. NC  파일: {nc_filepath}")
        return True
        
    except Exception as e:
        print(f"[오류] 파일 생성 중 오류 발생: {e}")
        return False
