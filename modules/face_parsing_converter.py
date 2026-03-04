# modules/face_parsing_converter.py
import cv2
import numpy as np
import os
import svgwrite
import onnxruntime as ort
from scipy.interpolate import splprep, splev
from .config import Z_SAFE, Z_DRAW, FEED_RATE, SCALE

def smooth_contour_spline(contour, num_points=100, smooth_factor=2.0):
    """
    거친 픽셀 외곽선(Contour)을 B-Spline을 사용해 아주 매끄러운 곡선으로 변환합니다.
    로봇의 모터 움직임을 부드럽게 만들어 줍니다.
    """
    pts = contour.reshape(-1, 2)
    
    # 점이 너무 적으면 보간하지 않고 그냥 반환
    if len(pts) < 5:
        return [(float(p[0]), float(p[1])) for p in pts]
    
    # 중복 점 제거 (Scipy splprep 에러 방지)
    _, idx = np.unique(pts, axis=0, return_index=True)
    pts = pts[np.sort(idx)]
    
    if len(pts) < 4:
        return [(float(p[0]), float(p[1])) for p in pts]

    x = pts[:, 0]
    y = pts[:, 1]

    try:
        # 폐곡선(닫힌 선)으로 가정하고 스플라인 곡선 계산 (per=True)
        tck, u = splprep([x, y], s=smooth_factor, per=True)
        u_new = np.linspace(u.min(), u.max(), num_points)
        x_new, y_new = splev(u_new, tck)
        return list(zip(x_new, y_new))
    except Exception as e:
        # 스플라인 계산 실패 시 원본 단순화 좌표 반환
        print(f"   [경고] 곡선 보간 실패 (원본 사용): {e}")
        return [(float(p[0]), float(p[1])) for p in pts]

def generate_files_face_parsing(image_bgr: np.ndarray, nc_filepath: str, svg_filepath: str, model_path: str = "models/bisenet.onnx") -> bool:
    """
    BiSeNet 얼굴 파싱 모델을 사용하여 부위별 외곽선을 추출하고 매끄럽게 보간하여 G-code를 생성합니다.
    """
    if not os.path.exists(model_path):
        print(f"[오류] 파싱 모델을 찾을 수 없습니다: {model_path}")
        return False

    print(">> [2단계] Face Parsing 및 스플라인 곡선 생성 시작")
    
    try:
        # 1. 모델 추론 준비
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # BiSeNet은 보통 512x512 해상도를 사용합니다.
        input_size = 512
        img_resized = cv2.resize(image_bgr, (input_size, input_size))
        
        # 전처리: 정규화 (ImageNet 평균/표준편차 사용)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_rgb / 255.0 - mean) / std
        
        # CHW 포맷 및 배치 차원 추가
        img_tensor = np.transpose(img_norm, (2, 0, 1)).astype(np.float32)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        
        # 2. 모델 실행
        input_name = session.get_inputs()[0].name
        out = session.run(None, {input_name: img_tensor})[0]
        
        # (1, 19, 512, 512) -> 클래스 인덱스로 변환 (512, 512)
        parsing_map = np.argmax(out[0], axis=0).astype(np.uint8)
        
        # 3. 로봇이 그릴 핵심 부위 클래스 정의
        # 1: 피부(얼굴윤곽), 2/3: 눈썹, 4/5: 눈, 10: 코, 11/12/13: 입술, 17: 머리카락
        target_classes = {
            "Face_Outline": [1],
            "Hair": [17],
            "Left_Eyebrow": [2],
            "Right_Eyebrow": [3],
            "Left_Eye": [4],
            "Right_Eye": [5],
            "Nose": [10],
            "Lips": [11, 12, 13]
        }
        
        paths = []
        
        # 4. 각 부위별 마스크에서 외곽선 추출 및 보간
        for part_name, class_indices in target_classes.items():
            # 해당 부위만 흰색(255), 나머지는 검은색(0)인 마스크 생성
            mask = np.isin(parsing_map, class_indices).astype(np.uint8) * 255
            
            # 노이즈 제거 (자잘한 구멍 메우기)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 외곽선 추출 (오직 가장 바깥쪽 외곽선만: RETR_EXTERNAL)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # 너무 작은 노이즈 점들은 무시 (면적 및 길이 기준)
                if cv2.contourArea(contour) < 50 or cv2.arcLength(contour, closed=True) < 20:
                    continue
                
                # 외곽선 단순화 (불필요한 점 줄이기)
                epsilon = 0.005 * cv2.arcLength(contour, closed=True)
                approx = cv2.approxPolyDP(contour, epsilon, closed=True)
                
                # 스플라인 곡선으로 부드럽게 만들기
                smoothed_path = smooth_contour_spline(approx, num_points=max(50, len(approx)*3))
                paths.append(smoothed_path)

        # 5. 파일 생성 (SVG & G-code)
        os.makedirs(os.path.dirname(svg_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(nc_filepath), exist_ok=True)
        
        dwg = svgwrite.Drawing(svg_filepath, profile='tiny', size=(input_size, input_size), viewBox=f"0 0 {input_size} {input_size}")
        count = 0
        
        with open(nc_filepath, 'w') as f:
            f.write("%\n")
            f.write("G21 (Units: mm)\n")
            f.write("G90 (Absolute)\n")
            f.write(f"G0 Z{Z_SAFE}\n")
            
            for path in paths:
                if len(path) < 2: continue
                
                # SVG 저장
                dwg.add(dwg.polyline(path, stroke='black', fill='none', stroke_width=2))
                
                # G-code 저장
                start_p = path[0]
                sx, sy = start_p[0] * SCALE, start_p[1] * SCALE
                f.write(f"G0 X{sx:.3f} Y{-sy:.3f}\n")
                f.write(f"G1 Z{Z_DRAW} F{FEED_RATE}\n")
                
                for j in range(1, len(path)):
                    px, py = path[j][0] * SCALE, path[j][1] * SCALE
                    f.write(f"G1 X{px:.3f} Y{-py:.3f}\n")
                
                # 폐곡선이므로 마지막에 다시 시작점으로 돌아오기
                f.write(f"G1 X{sx:.3f} Y{-sy:.3f}\n")
                
                f.write(f"G0 Z{Z_SAFE}\n")
                count += 1
            
            f.write("G0 X0 Y0\n")
            f.write("M2\n")
            f.write("%\n")
            
        dwg.save()
        
        print(f">> 생성 완료! 총 {count}개의 파싱 획(스플라인)")
        print(f"   1. SVG 파일: {svg_filepath}")
        print(f"   2. NC  파일: {nc_filepath}")
        return True
        
    except Exception as e:
        print(f"[오류] Face Parsing 처리 중 오류 발생: {e}")
        return False