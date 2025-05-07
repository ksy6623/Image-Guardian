import os
import shutil
from watermark import add_watermark
from utils import is_valid_image

def create_dataset(input_dir="input_images", output_dir="dataset/train", watermark_text="워터마크"):
    os.makedirs(os.path.join(output_dir, "original"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "protected"), exist_ok=True)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        # 이미지 파일만 처리
        if not is_valid_image(input_path):
            continue

        # 원본 복사
        shutil.copy(input_path, os.path.join(output_dir, "original", filename))

        # 워터마크 삽입
        watermarked_output_path = os.path.join(output_dir, "protected", filename)
        add_watermark(input_path, text=watermark_text, output_path=watermarked_output_path)

    print("✅ 데이터셋 생성 완료!")

if __name__ == "__main__":
    create_dataset()
