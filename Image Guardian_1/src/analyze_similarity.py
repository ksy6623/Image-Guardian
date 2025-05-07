from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

def analyze_similarity(original_path, protected_path):
    try:
        # 이미지 열기
        original = Image.open(original_path).convert('L')  # Grayscale
        protected = Image.open(protected_path).convert('L')  # Grayscale

        # 이미지 크기 맞추기
        if original.size != protected.size:
            protected = protected.resize(original.size)

        # numpy 배열로 변환
        original_np = np.array(original)
        protected_np = np.array(protected)

        # SSIM 계산
        similarity_score, _ = ssim(original_np, protected_np, full=True)

        print(f"✅ 이미지 유사도 (SSIM): {similarity_score:.4f}")
        return similarity_score

    except Exception as e:
        print(f"❗ 유사도 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None
