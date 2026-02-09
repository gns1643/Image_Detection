import cv2
import numpy as np
from rembg import remove
from PIL import Image

def load_and_preprocess(filepath):
    """이미지를 불러와서 배경을 제거하고 부드럽게(Blur) 만듭니다."""
    print(f"🔄 [1단계] 이미지 로드 및 배경 제거: {filepath}")
    
    try:
        # 1. PIL로 열어서 배경 제거
        input_img = Image.open(filepath)
        output_img = remove(input_img)
        
        # 2. OpenCV 포맷으로 변환 (RGB -> BGR)
        img_np = np.array(output_img)
        
        # 투명한 배경을 흰색으로 변경
        if img_np.shape[2] == 4:
            alpha = img_np[:, :, 3]
            img_rgb = img_np[:, :, :3]
            bg = np.ones_like(img_rgb, dtype=np.uint8) * 255
            alpha_factor = alpha[:, :, np.newaxis] / 255.0
            img_bgr = (img_rgb * alpha_factor + bg * (1 - alpha_factor)).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
        # 3. 그레이스케일 변환 및 블러링 (노이즈 제거용)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blur
        
    except Exception as e:
        print(f"❌ 이미지 처리 실패: {e}")
        return None