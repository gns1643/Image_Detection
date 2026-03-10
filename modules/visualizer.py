import cv2
import numpy as np
import os

def create_pipeline_diagram(image_paths, labels, output_path):
    """
    이미지 경로 리스트를 받아 가로로 합성하고 라벨을 추가하여 파이프라인 다이어그램 생성
    """
    images = []
    target_height = 400
    
    for path in image_paths:
        if os.path.exists(path):
            img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                # 가로 세로 비율 유지하며 높이 맞춤
                h, w = img.shape[:2]
                new_w = int(w * (target_height / h))
                img_res = cv2.resize(img, (new_w, target_height))
                images.append(img_res)
    
    if not images:
        return

    # 이미지들 사이에 넣을 화살표 영역 생성
    arrow_w = 60
    arrow_img = np.full((target_height, arrow_w, 3), 255, dtype=np.uint8)
    cv2.putText(arrow_img, ">>", (10, target_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # 전체 캔버스 계산 (이미지들 + 화살표들 + 상단 텍스트 영역)
    total_width = sum(img.shape[1] for img in images) + arrow_w * (len(images) - 1)
    canvas_height = target_height + 100
    canvas = np.full((canvas_height, total_width, 3), 255, dtype=np.uint8)

    current_x = 0
    for i, img in enumerate(images):
        # 이미지 배치
        canvas[80:80+target_height, current_x:current_x+img.shape[1]] = img
        
        # 라벨 배치
        label = labels[i] if i < len(labels) else ""
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = current_x + (img.shape[1] - text_size[0]) // 2
        cv2.putText(canvas, label, (max(0, text_x), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
        
        current_x += img.shape[1]
        
        # 다음 이미지 사이 화살표
        if i < len(images) - 1:
            canvas[80:80+target_height, current_x:current_x+arrow_w] = arrow_img
            current_x += arrow_w

    # 결과 저장
    cv2.imencode(".png", canvas)[1].tofile(output_path)
    print(f"\n[시각화] 파이프라인 다이어그램이 생성되었습니다: {output_path}")
