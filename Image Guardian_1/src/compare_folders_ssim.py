from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import os

def compare_folders(original_folder, protected_folder):
    total_ssim = 0
    count = 0

    for filename in os.listdir(original_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            original_path = os.path.join(original_folder, filename)
            protected_path = os.path.join(protected_folder, filename)

            if not os.path.exists(protected_path):
                print(f"❗ 보호된 이미지가 없습니다: {filename}")
                continue

            try:
                # 이미지 열기 (PIL)
                original = Image.open(original_path).convert("L")  # Grayscale
                protected = Image.open(protected_path).convert("L")

                # 크기 맞추기
                protected = protected.resize(original.size)

                # numpy 배열로 변환
                original_np = np.array(original)
                protected_np = np.array(protected)

                # SSIM 계산
                score, _ = ssim(original_np, protected_np, full=True)
                print(f"✅ {filename} SSIM: {score:.4f}")

                total_ssim += score
                count += 1

            except Exception as e:
                print(f"❗ 이미지 비교 실패: {filename} ({e})")
                continue

    if count > 0:
        avg_ssim = total_ssim / count
        print(f"\n✅ 전체 평균 SSIM: {avg_ssim:.4f}")
    else:
        print("❗ 비교할 이미지가 없습니다.")

if __name__ == "__main__":
    original_folder = input("원본 이미지 폴더 경로를 입력하세요: ").strip()
    protected_folder = input("보호된 이미지 폴더 경로를 입력하세요: ").strip()
    compare_folders(original_folder, protected_folder)
