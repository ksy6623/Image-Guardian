import os
import torch
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision import transforms
from PIL import Image

def feature_attack(image_path, output_path="output/feature_attacked.png", epsilon=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vgg16(pretrained=True).features.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    feature = model(input_tensor)
    loss = -feature.norm()
    loss.backward()

    perturbed = input_tensor + epsilon * input_tensor.grad.sign()
    perturbed = torch.clamp(perturbed, 0, 1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    transforms.ToPILImage()(perturbed.squeeze(0).cpu()).save(output_path)

    print(f"✅ Feature attack 이미지 저장 완료: {output_path}")

if __name__ == "__main__":
    input_path = input("Feature 공격할 이미지 파일 경로를 입력하세요: ")
    feature_attack(input_path)
