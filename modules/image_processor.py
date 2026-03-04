import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image
import os

# ONNX Runtime의 불필요한 로그(에러 메시지)를 줄이기 위한 환경 변수 설정
os.environ["ORT_LOGGING_LEVEL"] = "3" 

def image_processor(image_bgr: np.ndarray):
    """
    이미지(OpenCV)를 입력받아 배경을 제거하고 부드럽게(Blur) 만듭니다.
    """
    print(">> [2단계] 이미지 전처리 (배경 제거 및 블러)")
    
    try:
        # 1. PIL 포맷으로 변환 (BGR -> RGB)
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_img_pil = Image.fromarray(rgb_image)
        
        # [수정] CPU 전용 세션을 생성하여 CUDA 관련 빨간 에러 메시지 방지
        # 모델명은 기본값인 'u2net'을 사용합니다.
        session = new_session("u2net", providers=['CPUExecutionProvider'])
        output_img_pil = remove(input_img_pil, session=session)
        
        # 2. 다시 OpenCV 포맷으로 변환
        img_np = np.array(output_img_pil)
        
        # 투명한 배경(RGBA)을 흰색(RGB)으로 변경
        if img_np.shape[2] == 4:
            alpha = img_np[:, :, 3]
            img_rgb = img_np[:, :, :3]
            bg = np.ones_like(img_rgb, dtype=np.uint8) * 255
            alpha_factor = alpha[:, :, np.newaxis] / 255.0
            # RGB 이미지에 알파 채널을 곱하고, 흰색 배경에 (1-알파)를 곱하여 더합니다.
            img_rgb = (img_rgb * alpha_factor + bg * (1 - alpha_factor)).astype(np.uint8)
        else:
            img_rgb = img_np
            
        # OpenCV는 BGR을 기본으로 사용하므로 최종 변환
        img_bgr_processed = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
        # 3. 그레이스케일 변환 및 블러링 (노이즈 제거용)
        gray = cv2.cvtColor(img_bgr_processed, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blur
        
    except Exception as e:
        print(f"[오류] 이미지 처리 실패: {e}")
        return None