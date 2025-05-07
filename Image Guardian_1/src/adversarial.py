import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def add_adversarial_noise(image_path, output_path="output/adversarial.png", epsilon=0.02):
    # output 폴더 없으면 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 이미지 로드
    image = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)  # (1, 3, H, W)

    # 단순 adversarial noise 생성
    noise = torch.randn_like(image_tensor) * epsilon
    adv_image = torch.clamp(image_tensor + noise, 0, 1)

    # 저장
    adv_image_np = adv_image.squeeze().permute(1, 2, 0).detach().numpy()
    adv_image_np = (adv_image_np * 255).astype(np.uint8)
    adv_image_pil = Image.fromarray(adv_image_np)
    adv_image_pil.save(output_path)

    return output_path
