import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from .model import Generator 

# --- 1. 모델 경로 설정 ---
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_WEIGHTS_PATH = os.path.join(_MODEL_DIR, "anime_style", "netG_A_latest.pth")

# --- 2. 전역 변수 (캐싱 및 디바이스 설정) ---
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None

# --- 3. 모델 초기화 함수 (Lazy Loading) ---
def _get_model() -> Generator:
    """
    모델을 한 번만 로드하여 메모리에 올려두고 재사용합니다.
    """
    global _model
    if _model is None:
        if not os.path.exists(_WEIGHTS_PATH):
            raise FileNotFoundError(f"[오류] 가중치 파일이 없습니다: {_WEIGHTS_PATH}")
            
        print(f">> [초기화] Anime 모델을 {_device} 메모리에 로드합니다...")
        _model = Generator(3, 1, 3).to(_device)
        _model.load_state_dict(torch.load(_WEIGHTS_PATH, map_location=_device, weights_only=True))
        _model.eval()
        
    return _model

# --- 4. 메인 스케치 변환 함수 ---
# 💡 threshold_value를 200에서 235로 올렸습니다. (선이 더 많이 남게 됩니다)
def generate_sketch(image_bgr: np.ndarray, threshold_value: int = 210) -> np.ndarray:
    """
    OpenCV 이미지를 입력받아 Anime 스타일의 흑백(이진화) 스케치 이미지를 반환합니다.
    """
    print(">> [스케치 변환] Anime 화풍으로 선화 추출 및 이진화를 시작합니다.")
    
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
        interpolation = cv2.INTER_AREA
    else:
        new_w, new_h = w, h
        interpolation = cv2.INTER_LINEAR

    new_w = new_w - (new_w % 16)
    new_h = new_h - (new_h % 16)
    
    image_resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=interpolation)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    input_tensor = transform(image_resized).unsqueeze(0).to(_device)

    # 2. 모델 추론
    model = _get_model()

    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    # 3. 텐서 후처리
    output_image = output_tensor.squeeze(0).cpu().numpy()
    output_image = (output_image * 0.5 + 0.5) * 255.0 
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    if len(output_image.shape) == 3:
        output_image = output_image[0]

    # 원본 해상도로 복구
    final_sketch = cv2.resize(output_image, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 4. 이진화 (Binarization)
    # threshold_value(기본 235) 이하의 픽셀은 0(검정)으로, 초과하는 픽셀은 255(하양)로 만듭니다.
    _, binarized_sketch = cv2.threshold(final_sketch, threshold_value, 255, cv2.THRESH_BINARY)
    
    return binarized_sketch