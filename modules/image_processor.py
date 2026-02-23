import cv2
import numpy as np
from rembg import remove
from PIL import Image

def image_processor(image_bgr: np.ndarray):
    """
    이미지(OpenCV)를 입력받아 배경을 제거하고 부드럽게(Blur) 만듭니다.
    
    :param image_bgr: 배경을 제거할 OpenCV 이미지 객체 (BGR 형식)
    :return: 전처리된 이미지 (Grayscale, blurred)
    """
    print(">> [2단계] 이미지 전처리 (배경 제거 및 블러)")
    
    try:
        # 1. PIL 포맷으로 변환 (BGR -> RGB) 후 배경 제거
        # rembg는 PIL 이미지를 입력으로 받습니다.
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_img_pil = Image.fromarray(rgb_image)
        output_img_pil = remove(input_img_pil)
        
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