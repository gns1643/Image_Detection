import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from .model import Generator 

# --- 모델 경로 설정 (Anime 스타일 단일 사용) ---
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_WEIGHTS_PATH = os.path.join(_MODEL_DIR, "anime_style", "netG_A_latest.pth")

# --- 딥러닝 스케치 변환 함수 ---
def generate_sketch(image_bgr: np.ndarray) -> np.ndarray:
    """
    OpenCV 이미지를 입력받아 Anime 스타일의 스케치 이미지 1장을 반환합니다.
    """
    print(">> [스케치 변환] Anime 화풍으로 선화 추출을 시작합니다.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 이미지 전처리
    if len(image_bgr.shape) == 3:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    
    h, w = image_rgb.shape[:2]
    max_size = 512
    scale = max_size / max(h, w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        new_w = new_w - (new_w % 2)
        new_h = new_h - (new_h % 2)
        image_resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        image_resized = image_rgb

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(image_resized).unsqueeze(0).to(device)

    # 2. 모델 로드 및 추론
    model = Generator(3, 1, 3).to(device)

    if not os.path.exists(_WEIGHTS_PATH):
        print(f"   [오류] 가중치 파일이 없습니다: {_WEIGHTS_PATH}")
        return None
        
    model.load_state_dict(torch.load(_WEIGHTS_PATH, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    # 3. 후처리
    output_image = output_tensor.squeeze(0).cpu().numpy()
    output_image = (output_image * 0.5 + 0.5) * 255.0 
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    if len(output_image.shape) == 3:
        output_image = output_image[0]

    final_sketch = cv2.resize(output_image, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return final_sketch