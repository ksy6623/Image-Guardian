from PIL import Image, ImageDraw, ImageFont
import os
import math


def add_watermark(image_path, text="워터마크", output_path="output/watermarked.png", opacity=0.2):
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # 이미지 열기
        original = Image.open(image_path)

        # RGBA 모드로 변환 (투명도 지원)
        if original.mode != 'RGBA':
            image = original.convert('RGBA')
        else:
            image = original.copy()

        width, height = image.size

        # 워터마크용 빈 레이어 생성
        watermark = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)

        # 폰트 크기 설정 - 이미지 크기에 비례
        fontsize = int(min(width, height) / 15)
        fontsize = max(15, min(fontsize, 40))  # 최소 15, 최대 40

        # 여러 폰트 경로 시도
        font = None
        font_paths = [
            "C:/Windows/Fonts/malgun.ttf",  # 한글 폰트 (윈도우)
            "C:/Windows/Fonts/Arial.ttf",  # 영문 폰트 (윈도우)
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # 한글 폰트 (리눅스)
            "/System/Library/Fonts/AppleSDGothicNeo.ttc"  # 한글 폰트 (맥)
        ]

        for path in font_paths:
            try:
                font = ImageFont.truetype(path, fontsize)
                break
            except OSError:
                continue

        if font is None:
            font = ImageFont.load_default()
            print("적합한 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")

        # 텍스트 크기 측정
        textbbox = draw.textbbox((0, 0), text, font=font)
        text_width = textbbox[2] - textbbox[0]
        text_height = textbbox[3] - textbbox[1]

        # 반복 간격 설정 (텍스트 크기의 3배)
        spacing_x = text_width * 2
        spacing_y = text_height * 2

        # 반복 횟수 계산
        repeat_x = math.ceil(width / spacing_x) + 1
        repeat_y = math.ceil(height / spacing_y) + 1

        # 시작 위치 (약간의 오프셋 추가)
        offset_x = spacing_x / 4
        offset_y = spacing_y / 4

        # 워터마크 반복 그리기
        for y in range(repeat_y):
            # 짝수/홀수 행에 따라 오프셋 조정 (벌집 모양으로)
            row_offset = offset_x if y % 2 == 0 else 0

            for x in range(repeat_x):
                pos_x = x * spacing_x + row_offset
                pos_y = y * spacing_y + offset_y

                # 텍스트 그리기
                draw.text((pos_x, pos_y), text, font=font, fill=(255, 255, 255, int(255 * opacity)))

        # 이미지와 워터마크 합성
        result = Image.alpha_composite(image, watermark)

        # 저장 (RGB로 변환)
        if original.mode == 'RGB':
            result = result.convert('RGB')

        result.save(output_path)
        print(f"워터마크가 적용된 이미지 저장 완료: {output_path}")
        return output_path

    except Exception as e:
        print(f"워터마크 적용 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None