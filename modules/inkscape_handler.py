import subprocess
import os
import shutil
import platform
import cv2
import tempfile
import base64

def get_inkscape_path():
    """시스템에서 Inkscape 실행 파일 경로를 찾습니다."""
    if platform.system() == "Windows":
        # 1.4+ 버전에서는 .com이 터미널 출력에 더 유리함
        common_paths = [
            r"C:\Program Files\Inkscape\bin\inkscape.com",
            r"C:\Program Files\Inkscape\bin\inkscape.exe",
            r"C:\Program Files\Inkscape\inkscape.exe"
        ]
        for p in common_paths:
            if os.path.exists(p): return p
    return shutil.which("inkscape")

def is_inkscape_installed():
    return get_inkscape_path() is not None

def run_inkscape_trace(input_path, output_svg_path, options=None):
    """
    Reddit 추천 방식(object-bitmap-trace)을 사용하여 팝업창 없이 벡터화를 수행합니다.
    """
    inkscape_exe = get_inkscape_path()
    if not inkscape_exe:
        print("[오류] Inkscape를 찾을 수 없습니다.")
        return False

    # 1. 이미지 읽기 및 Base64 인코딩 (안전한 데이터 전달)
    import numpy as np
    try:
        img_array = np.fromfile(input_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None: raise ValueError("이미지 디코딩 실패")
        h, w = img.shape[:2]
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"[오류] 이미지 처리 실패: {e}")
        return False

    # 2. 임시 폴더에서 작업
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_wrapper = os.path.join(tmpdir, "wrapper.svg")
        tmp_output = os.path.join(tmpdir, "output.svg")

        # 3. 데이터 내장형 SVG 생성 (팝업 방지)
        wrapper_content = f'''<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">
  <image href="data:image/png;base64,{img_base64}" width="{w}" height="{h}" />
</svg>'''
        with open(tmp_wrapper, "w", encoding="utf-8") as f:
            f.write(wrapper_content)

        # 4. Reddit 추천 Inkscape 1.4+ 액션 시퀀스
        # object-bitmap-trace에 임계값(threshold) 인자를 직접 전달합니다.
        # 인자 형식이 버전마다 다를 수 있으므로, 가장 보편적인 콜론 방식을 사용합니다.
        # 0.7 정도로 높게 설정하여 옅은 선들도 확실히 벡터화하도록 유도합니다.
        actions = (
            "select-all; "
            "object-bitmap-trace:threshold=0.7; "
            f"export-filename:{tmp_output}; "
            "export-do; "
            "file-close"
        )
        
        command = [
            inkscape_exe,
            tmp_wrapper,
            "--batch-process",
            f"--actions={actions}"
        ]

        try:
            print(f">> [Inkscape 1.4+] Reddit 추천 방식으로 벡터화 진행 중...")
            startupinfo = None
            if platform.system() == "Windows":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            # 실행
            result = subprocess.run(command, capture_output=True, text=True, startupinfo=startupinfo)
            
            if os.path.exists(tmp_output):
                shutil.copy(tmp_output, output_svg_path)
                print(f">> [Inkscape] 변환 성공!")
                return True
            else:
                print(f"[오류] 변환 실패 (코드: {result.returncode})")
                if result.stderr:
                    print(f"--- Inkscape 로그 ---\n{result.stderr.strip()}\n---------------------")
                return False
                
        except Exception as e:
            print(f"[오류] Inkscape 실행 중 크래시: {e}")
            return False
