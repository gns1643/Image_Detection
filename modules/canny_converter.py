import cv2
import os
import svgwrite
import numpy as np
from typing import Tuple, List
from .config import FINAL_TH1, FINAL_TH2, Z_SAFE, Z_DRAW, FEED_RATE, SCALE

def generate_files_binary(binary_image: np.ndarray, nc_filepath: str, svg_filepath: str) -> bool:
    """이미 이진화된 이미지를 사용하여 NC와 SVG 파일을 생성합니다."""
    print(f">> [외곽선 추출] 이진화 이미지 기반으로 경로를 생성합니다.")

    # 눈으로 보기엔 흰 바탕/검은 선이 좋지만, OpenCV는 검은 바탕/흰 선을 객체로 인식합니다.
    # 따라서 경로를 추출하기 직전에 메모리 상에서만 흑백을 반전시킵니다.
    inverted_binary = cv2.bitwise_not(binary_image)

    # [수정 1] RETR_EXTERNAL -> RETR_LIST로 변경하여 내부 디테일(눈, 코, 입 등) 모두 추출
    contours, _ = cv2.findContours(inverted_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 윤곽선을 Y축 기준으로 정렬
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    height, width = binary_image.shape[:2]
    
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

            for i, contour in enumerate(contours):
                # [수정 2] 짧은 디테일도 살리기 위해 노이즈 제한을 15에서 5로 하향
                if cv2.arcLength(contour, closed=False) < 5:
                    continue

                # [수정 3] 곡선을 더 정교하게 따기 위해 epsilon 계수를 0.002에서 0.001로 하향
                epsilon = 0.001 * cv2.arcLength(contour, closed=False)
                approx = cv2.approxPolyDP(contour, epsilon, closed=False)

                if len(approx) < 2:
                    continue

                # [수정 4] 이진화 기반에서는 선을 반토막 내는 trim_contour_to_midpoint() 삭제

                # SVG 저장
                points = [(float(p[0][0]), float(p[0][1])) for p in approx]
                dwg.add(dwg.polyline(points, stroke='black', fill='none', stroke_width=2))

                # G-코드 저장
                start_p = approx[0][0]
                sx = float(start_p[0]) * SCALE
                sy = float(start_p[1]) * SCALE

                f.write(f"G0 X{sx:.3f} Y{-sy:.3f}\n")
                f.write(f"G1 Z{Z_DRAW} F{FEED_RATE}\n")

                for j in range(1, len(approx)):
                    p = approx[j][0]
                    px = float(p[0]) * SCALE
                    py = float(p[1]) * SCALE
                    f.write(f"G1 X{px:.3f} Y{-py:.3f}\n")

                f.write(f"G0 Z{Z_SAFE}\n")
                count += 1

            f.write("G0 X0 Y0\n")
            f.write("M2\n")
            f.write("%\n")

        dwg.save()
        print(f">> 생성 완료! 총 {count}개의 획")
        return True

    except Exception as e:
        print(f"[오류] 파일 생성 중 오류 발생: {e}")
        return False

def trim_contour_to_midpoint(approx: np.ndarray) -> np.ndarray:
    """
    갔다가 되돌아오는 contour에서 첫 절반(전진 경로)만 추출.
    시작점에서 가장 멀리 떨어진 인덱스까지만 사용.
    """
    if len(approx) < 2:
        return approx

    start = approx[0][0].astype(float)

    # numpy 벡터 연산으로 한 번에 거리 계산 (성능 개선)
    points = approx[:, 0, :].astype(float)  # shape: (N, 2)
    distances = np.linalg.norm(points - start, axis=1)

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
    height, width = img_blurr.shape[:2]  # 컬러/그레이 모두 안전하게 처리

    # 출력 폴더가 없으면 생성
    os.makedirs(os.path.dirname(svg_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(nc_filepath), exist_ok=True)

    dwg = svgwrite.Drawing(svg_filepath, profile='tiny', size=(width, height), viewBox=f"0 0 {width} {height}")

    count = 0

    try:
        with open(nc_filepath, 'w') as f:
            # 헤더 작성
            f.write("%\n")
            f.write("G21 (Units: mm)\n")
            f.write("G90 (Absolute)\n")
            f.write(f"G0 Z{Z_SAFE}\n")

            for i, contour in enumerate(contours):
                # 잡티 제거 (너무 짧은 선은 무시)
                if cv2.arcLength(contour, closed=False) < 15:
                    continue

                # 단순화 (점 개수 줄이기)
                epsilon = 0.002 * cv2.arcLength(contour, closed=False)
                approx = cv2.approxPolyDP(contour, epsilon, closed=False)

                if len(approx) < 2:
                    continue

                # 항상 trim 적용 → 갔다 되돌아오는 두겹 경로 제거
                approx = trim_contour_to_midpoint(approx)

                if len(approx) < 2:
                    continue

                # --- [SVG 저장] ---
                points = [(float(p[0][0]), float(p[0][1])) for p in approx]
                dwg.add(dwg.polyline(points, stroke='black', fill='none', stroke_width=2))

                # --- [G-코드 저장] ---
                start_p = approx[0][0]
                sx = float(start_p[0]) * SCALE
                sy = float(start_p[1]) * SCALE

                f.write(f"G0 X{sx:.3f} Y{-sy:.3f}\n")
                f.write(f"G1 Z{Z_DRAW} F{FEED_RATE}\n")

                # 경로 따라 그리기 (루프 변수 j로 수정 → 외부 i와 충돌 방지)
                for j in range(1, len(approx)):
                    p = approx[j][0]
                    px = float(p[0]) * SCALE
                    py = float(p[1]) * SCALE
                    f.write(f"G1 X{px:.3f} Y{-py:.3f}\n")

                f.write(f"G0 Z{Z_SAFE}\n")
                count += 1

            # 푸터 작성
            f.write("G0 X0 Y0\n")
            f.write("M2\n")
            f.write("%\n")

        dwg.save()

        print(f">> 생성 완료! 총 {count}개의 획")
        print(f"   1. SVG 파일: {svg_filepath}")
        print(f"   2. NC  파일: {nc_filepath}")
        return True

    except IOError as e:
        print(f"[오류] 파일 입출력 오류: {e}")
        return False
    except cv2.error as e:
        print(f"[오류] OpenCV 처리 오류: {e}")
        return False