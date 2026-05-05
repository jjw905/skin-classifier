"""
모든 모델에서 공통으로 쓰이는 클래스 정보 상수

상수:
    NUM_CLASSES(클래스 수)       분류 대상 질환 수 (6가지)
    CLASS_NAMES(클래스 이름)     클래스 이름 목록 (영문)
    CLASS_NAMES_KR(클래스 이름) 클래스 이름 목록 (한국어)
"""

import torch
import torch.nn as nn

NUM_CLASSES    = 6
CLASS_NAMES    = ["psoriasis", "atopy", "acne", "normal", "rosacea", "seborrheic"]
CLASS_NAMES_KR = ["건선", "아토피", "여드름", "정상", "주사", "지루"]
