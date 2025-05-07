import os
import torch
from torchvision.models import vgg16
from torchvision import transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np

# .webp 포함하여 이미지 열기
def open_image_convert_rgb(path):
    return Image.open(path).convert('RGB')

# 보호 함수 (VGG16 특징 기반 + Nightshade 스타일 노이즈)
def protect_image(input_tensor, epsilon=0.1, nightshade_strength=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vgg16(weights="IMAGENET1K_V1").features.to(device).eval()

    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad = True

    feature = model(input_tensor)
    loss = -feature.norm()
    loss.backward()

    adv_noise = epsilon * input_tensor.grad.sign()
    random_noise = nightshade_strength * torch.randn_like(input_tensor)
    perturbed = input_tensor + adv_noise + random_noise
    perturbed = torch.clamp(perturbed, 0, 1)

    return perturbed

# 한 장 보호
def protect_single_image(input_path, output_path):
    image = open_image_convert_rgb(input_path)
    transform = transforms.ToTensor()
    input_tensor = transform(image).unsqueeze(0)

    protected_tensor = protect_image(input_tensor)
    protected_image = transforms.ToPILImage()(protected_tensor.squeeze(0).cpu())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    protected_image.save(output_path, quality=95)
    print(f"✅ 보호 완료: {output_path}")

# 폴더 전체 보호 (.webp 포함)
def protect_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            input_path = os.path.join(input_folder, filename)
            image = open_image_convert_rgb(input_path)
            input_tensor = transforms.ToTensor()(image).unsqueeze(0)

            protected_tensor = protect_image(input_tensor)
            protected_image = transforms.ToPILImage()(protected_tensor.squeeze(0).cpu())

            output_path = os.path.join(output_folder, filename)
            protected_image.save(output_path, quality=95)
            print(f"✅ 보호 완료: {filename}")

# 실행 진입점
if __name__ == "__main__":
    choice = input("한 장 보호 [1], 폴더 전체 보호 [2] 중 선택하세요: ").strip()

    if choice == "1":
        input_path = input("보호할 원본 이미지 파일 경로를 입력하세요: ").strip()
        output_path = input("보호된 이미지를 저장할 경로를 입력하세요 (예: output/protected.png): ").strip()
        protect_single_image(input_path, output_path)

    elif choice == "2":
        input_folder = input("보호할 원본 이미지 폴더 경로를 입력하세요: ").strip()
        output_folder = input("보호된 이미지를 저장할 폴더 경로를 입력하세요: ").strip()
        protect_folder(input_folder, output_folder)

    else:
        print("❗ 잘못된 입력입니다. 1 또는 2를 선택하세요.")
