import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from .model import APDrawingGenerator

# --- 모델 경로 설정 ---
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
# 새로 다운로드받은 가중치 파일 경로
_WEIGHTS_PATH = os.path.join(_MODEL_DIR, "checkpoints", "checkpoints", "apdrawinggan++_author", "150_net_gen.pt")

# 전역 변수로 모델 캐싱 (성능 최적화)
_cached_ap_model = None

def get_ap_model(device):
    global _cached_ap_model
    if _cached_ap_model is None:
        if not os.path.exists(_WEIGHTS_PATH):
            print(f"   [오류] APDrawingGAN2 가중치 파일이 없습니다: {_WEIGHTS_PATH}")
            return None
            
        print(f">> [모델 로드] APDrawingGAN2 모델을 로드합니다: {os.path.basename(_WEIGHTS_PATH)}")
        # APDrawingGenerator가 GlobalGenerator2를 직접 상속하므로 구조가 단순해짐
        _cached_ap_model = APDrawingGenerator(input_nc=3, output_nc=1).to(device)
        state_dict = torch.load(_WEIGHTS_PATH, map_location=device, weights_only=True)
        
        # 1. 'G' 키 내부의 실제 생성기 가중치 추출
        if 'G' in state_dict:
            g_state_dict = state_dict['G']
        else:
            g_state_dict = state_dict

        # 2. 키 정제 (DataParallel 접두어 제거)
        new_state_dict = {}
        for k, v in g_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        # strict=True로 설정하여 모델 구조와 가중치가 완벽히 일치하는지 확인
        _cached_ap_model.load_state_dict(new_state_dict, strict=True)
        _cached_ap_model.eval()
    return _cached_ap_model

def generate_ap_sketch(image_input: np.ndarray) -> np.ndarray:
    """
    OpenCV 이미지를 입력받아 APDrawingGAN2 스타일의 고해상도 예술 선화를 반환합니다.
    격자무늬 노이즈 제거를 위해 양방향 필터(Bilateral Filter)를 적용합니다.
    """
    print(">> [스케치 변환] APDrawingGAN2 예술 선화 추출을 시작합니다.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_ap_model(device)
    if model is None: return None

    # 1. 이미지 전처리
    h, w = image_input.shape[:2]
    
    # 입력이 그레이스케일이면 RGB로, BGR이면 RGB로 변환 (모델은 3채널 RGB 기대)
    if len(image_input.shape) == 2 or image_input.shape[2] == 1:
        image_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
    
    input_size = 512
    image_resized = cv2.resize(image_rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)

    # 3채널 정규화 ([-1, 1] 범위)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(image_resized).unsqueeze(0).to(device)

    # 2. 모델 추론
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    # 3. 후처리
    output_image = output_tensor.squeeze(0).cpu().numpy()
    if output_image.shape[0] == 1: # Single channel output
        output_image = output_image[0]
        
    # Tanh [-1, 1] -> [0, 255] 변환
    output_image = (output_image * 0.5 + 0.5) * 255.0
    
    # 결과물이 너무 밝은 경우(모든 픽셀이 200 이상 등)를 대비한 자동 대비 보정
    # 최소값과 최대값의 차이가 어느 정도 있는 경우에만 수행
    if output_image.max() - output_image.min() > 10:
        output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min()) * 255.0

    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    # 원본 크기로 복구
    final_sketch = cv2.resize(output_image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    # --- [노이즈 제거] 격자무늬(Checkerboard) 및 GAN 노이즈 억제 ---
    final_sketch = cv2.bilateralFilter(final_sketch, d=9, sigmaColor=75, sigmaSpace=75)
    final_sketch = cv2.GaussianBlur(final_sketch, (3, 3), 0)
    
    # 대비 향상 (선명도 보정)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    final_sketch = clahe.apply(final_sketch)

    return final_sketch
