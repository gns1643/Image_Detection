import os
import svgwrite
from .config import Z_SAFE, Z_DRAW, FEED_RATE, SCALE

def generate_gcode_from_paths(paths, nc_filepath, svg_filepath, width=1000, height=1000):
    """
    점(Point)들의 리스트(Path들)를 입력받아 G-code와 SVG 파일을 생성합니다.
    paths: List[List[Tuple[float, float]]] - 각 경로는 (x, y) 좌표들의 리스트입니다.
    """
    os.makedirs(os.path.dirname(svg_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(nc_filepath), exist_ok=True)

    dwg = svgwrite.Drawing(svg_filepath, profile='tiny', size=(width, height), viewBox=f"0 0 {width} {height}")
    count = 0

    try:
        with open(nc_filepath, 'w') as f:
            f.write("%\n")
            f.write("G21 (Units: mm)\n")
            f.write("G90 (Absolute)\n")
            f.write(f"G0 Z{Z_SAFE}\n")

            for path in paths:
                if len(path) < 2:
                    continue

                # SVG 저장
                dwg.add(dwg.polyline(path, stroke='black', fill='none', stroke_width=2))

                # G-코드 저장
                start_p = path[0]
                sx = float(start_p[0]) * SCALE
                sy = float(start_p[1]) * SCALE

                f.write(f"G0 X{sx:.3f} Y{-sy:.3f}\n")
                f.write(f"G1 Z{Z_DRAW} F{FEED_RATE}\n")

                for i in range(1, len(path)):
                    px, py = float(path[i][0]) * SCALE, float(path[i][1]) * SCALE
                    f.write(f"G1 X{px:.3f} Y{-py:.3f}\n")

                f.write(f"G0 Z{Z_SAFE}\n")
                count += 1

            f.write("G0 X0 Y0\n")
            f.write("M2\n")
            f.write("%\n")

        dwg.save()
        print(f">> G-code 생성 완료! 총 {count}개의 획")
        return True

    except Exception as e:
        print(f"[오류] G-code 파일 생성 중 오류 발생: {e}")
        return False
