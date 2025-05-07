import os
import torch
import torchvision.transforms as transforms
from torchvision.models import vgg16
from PIL import Image

# .webp 포함 이미지 로딩 함수
def open_image_convert_rgb(path):
    return Image.open(path).convert('RGB')

# 스타일 블록형 보호 함수
def protect_image_with_style_block(input_tensor, epsilon=0.1, block_size=16, noise_std=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vgg16(weights="IMAGENET1K_V1").features.to(device).eval()

    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad = True

    features = model(input_tensor)
    loss = -features.norm()
    loss.backward()

    adv_noise = epsilon * input_tensor.grad.sign()
    adv_img = input_tensor + adv_noise

    # 블록 스타일 노이즈 추가
    _, _, h, w = adv_img.shape
    noise = torch.zeros_like(adv_img)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block_noise = torch.randn(1, 3, 1, 1).to(device) * noise_std
            noise[:, :, i:i+block_size, j:j+block_size] = block_noise

    perturbed = adv_img + noise
    perturbed = torch.clamp(perturbed, 0, 1)
    return perturbed

# 한 장 보호
def protect_single_image(input_path, output_path):
    transform = transforms.ToTensor()
    image = open_image_convert_rgb(input_path)
    input_tensor = transform(image).unsqueeze(0)

    protected_tensor = protect_image_with_style_block(input_tensor)
    protected_image = transforms.ToPILImage()(protected_tensor.squeeze(0).cpu())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    protected_image.save(output_path)
    print(f"✅ 보호 완료: {output_path}")

# 폴더 전체 보호 (.webp 포함)
def protect_folder(input_folder, output_folder):
    transform = transforms.ToTensor()
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            input_path = os.path.join(input_folder, filename)
            image = open_image_convert_rgb(input_path)
            input_tensor = transform(image).unsqueeze(0)

            protected_tensor = protect_image_with_style_block(input_tensor)
            protected_image = transforms.ToPILImage()(protected_tensor.squeeze(0).cpu())

            output_path = os.path.join(output_folder, filename)
            protected_image.save(output_path)
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
