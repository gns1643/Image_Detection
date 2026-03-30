import re
import xml.etree.ElementTree as ET
import numpy as np

def parse_svg_paths(svg_path):
    """
    SVG 파일에서 모든 <path> 태그의 'd' 속성을 찾아 점들의 리스트로 변환합니다.
    네임스페이스와 계층 구조에 상관없이 모든 경로를 추출합니다.
    """
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        paths = []
        
        # 1. 모든 path 태그를 재귀적으로 찾기 (네임스페이스 무시 방식)
        # 태그 이름에 'path'가 포함된 모든 요소를 찾습니다.
        for elem in root.iter():
            # 태그명에서 네임스페이스 제거 ({...} 제거)
            tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            
            if tag_name.lower() == 'path':
                d = elem.get('d')
                if not d:
                    continue
                
                # 'd' 속성 파싱
                points = parse_d_attribute(d)
                if points and len(points) >= 2:
                    paths.append(points)
        
        if not paths:
            print(f"[알림] SVG 파일 '{svg_path}'에서 유효한 <path>를 찾지 못했습니다.")
            
        return paths
    except Exception as e:
        print(f"[오류] SVG 파싱 중 문제 발생: {e}")
        return []

def parse_d_attribute(d):
    """
    'd' 속성 문자열을 좌표 리스트로 변환합니다.
    M, L, C, Z 명령어를 기본적으로 처리합니다.
    """
    # 명령어와 숫자를 분리하기 위한 정규표현식
    # 소문자(상대좌표)와 대문자(절대좌표) 모두 대응
    tokens = re.findall(r'([a-zA-Z])|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', d)
    
    current_path = []
    all_paths = [] # 여러 개의 M 명령이 있을 경우 대비
    last_point = (0.0, 0.0)
    
    i = 0
    while i < len(tokens):
        cmd = tokens[i][0]
        if cmd:
            i += 1
            cmd_upper = cmd.upper()
            
            # Move To (새로운 획 시작)
            if cmd_upper == 'M':
                # 기존 획이 있다면 저장 (보통 한 path 내에 여러 M이 있을 수 있음)
                # 여기서는 단순화를 위해 하나의 연속된 리스트로 합치거나 
                # 파서 구조에 따라 처리 방식을 결정해야 함
                try:
                    x, y = float(tokens[i][1]), float(tokens[i+1][1])
                    current_path.append((x, y))
                    last_point = (x, y)
                    i += 2
                except (IndexError, ValueError): pass
                
            # Line To
            elif cmd_upper == 'L':
                try:
                    x, y = float(tokens[i][1]), float(tokens[i+1][1])
                    current_path.append((x, y))
                    last_point = (x, y)
                    i += 2
                except (IndexError, ValueError): pass

            # Cubic Bezier (곡선 보간)
            elif cmd_upper == 'C':
                try:
                    x1, y1 = float(tokens[i][1]), float(tokens[i+1][1])
                    x2, y2 = float(tokens[i+2][1]), float(tokens[i+3][1])
                    x, y = float(tokens[i+4][1]), float(tokens[i+5][1])
                    
                    # 4분할 선형 보간으로 곡선 근사
                    for t in [0.25, 0.5, 0.75, 1.0]:
                        nx, ny = cubic_bezier(last_point, (x1, y1), (x2, y2), (x, y), t)
                        current_path.append((nx, ny))
                    
                    last_point = (x, y)
                    i += 6
                except (IndexError, ValueError): pass
                
            # Close Path
            elif cmd_upper == 'Z':
                if current_path:
                    current_path.append(current_path[0])
            
            # 기타 (H, V, Q, S, T, A 등) - 현재는 단순화를 위해 끝점만 추출하거나 건너뜀
            # 실제 고품질 G-code를 위해서는 이 부분의 보강이 필요함
        else:
            # 명령어 없이 좌표만 나열되는 경우 (암시적 L)
            try:
                x, y = float(tokens[i][1]), float(tokens[i+1][1])
                current_path.append((x, y))
                last_point = (x, y)
                i += 2
            except (IndexError, ValueError):
                i += 1
            
    return current_path

def cubic_bezier(p0, p1, p2, p3, t):
    """3차 베지어 곡선 좌표 계산"""
    cx = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
    cy = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
    return cx, cy
