import cv2
import numpy as np
from rembg import remove
from PIL import Image

def image_processor(image: np.ndarray) -> np.ndarray:
    """
    이미지(BGR)를 입력받아 배경 제거, 그레이스케일, 블러 처리를 순차적으로 수행합니다.
    Canny Edge Detection을 위한 전처리 과정입니다.
    
    :param image: 처리할 원본 BGR 이미지
    :return: 모든 전처리가 완료된 이미지 (Canny 입력용)
    """
    processed_image = remove_background(image)
    if processed_image is None:
        return None # 배경 제거 실패
    
    final_image = grayscale_and_blur(processed_image)
    if final_image is None:
        return None # 그레이스케일/블러 실패
        
    return final_image

def remove_background(image_bgr: np.ndarray):
    """
    이미지(OpenCV BGR)를 입력받아 배경을 제거하고 흰색 배경의 BGR 이미지를 반환합니다.
    
    :param image_bgr: 배경을 제거할 OpenCV 이미지 객체 (BGR 형식)
    :return: 배경이 제거된 BGR 이미지
    """
    print(">> [전처리 1/2] 배경 제거 중...")
    
    try:
        # 1. PIL 포맷으로 변환 (BGR -> RGB) 후 배경 제거
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_img_pil = Image.fromarray(rgb_image)
        output_img_pil = remove(input_img_pil)
        
        # 2. 다시 OpenCV 포맷으로 변환
        img_np = np.array(output_img_pil)
        
        # 3. 투명한 배경(RGBA)을 흰색(BGR)으로 변경
        if img_np.shape[2] == 4:
            alpha = img_np[:, :, 3]
            img_rgb = img_np[:, :, :3]
            bg = np.ones_like(img_rgb, dtype=np.uint8) * 255
            alpha_factor = alpha[:, :, np.newaxis] / 255.0
            img_rgb = (img_rgb * alpha_factor + bg * (1 - alpha_factor)).astype(np.uint8)
        else:
            img_rgb = img_np
            
        # OpenCV는 BGR을 기본으로 사용하므로 최종 변환
        img_bgr_processed = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        print("   - 배경 제거 완료")
        return img_bgr_processed
        
    except Exception as e:
        print(f"[오류] 배경 제거 실패: {e}")
        return None

def grayscale_and_blur(image_bgr: np.ndarray):
    """
    이미지(BGR)를 그레이스케일로 변환하고 블러링합니다. Canny 입력용입니다.
    
    :param image_bgr: BGR 이미지
    :return: 그레이스케일 블러 처리된 이미지
    """
    print(">> [전처리 2/2] 그레이스케일 변환 및 블러링...")
    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        print("   - 그레이스케일 및 블러 완료")
        return blur
    except Exception as e:
        print(f"[오류] 그레이스케일 변환/블러 실패: {e}")
        return None