import os
import shutil
from src.watermark import add_watermark
from src.adversarial import add_adversarial_noise
from src.analyze_similarity import analyze_similarity
from src.utils import is_valid_image

def main():
    image_path = input("보호할 이미지 파일 경로를 입력하세요 (png, jpg, jpeg, bmp): ")

    if not is_valid_image(image_path):
        print("❗ 지원하지 않는 파일 형식입니다. png, jpg, jpeg, bmp 파일만 가능합니다.")
        return

    # 원본 복사해서 따로 저장 (data/original_uploaded.png)
    os.makedirs("data", exist_ok=True)
    original_copy_path = "data/original_uploaded.png"
    shutil.copy(image_path, original_copy_path)

    watermark_text = input("워터마크에 사용할 문구를 입력하세요: ")

    # 워터마크 삽입
    watermarked_path = add_watermark(image_path, text=watermark_text)
    print(f"✅ 워터마크 이미지 저장 완료: {watermarked_path}")

    # Adversarial Noise 삽입
    adversarial_path = add_adversarial_noise(image_path)
    print(f"✅ Adversarial 노이즈 이미지 저장 완료: {adversarial_path}")

    # 유사도 분석
    similarity = analyze_similarity(original_copy_path, watermarked_path)
    if similarity is not None:
        print(f"✅ 유사도 분석 결과 (SSIM): {similarity:.4f}")

if __name__ == "__main__":
    main()
