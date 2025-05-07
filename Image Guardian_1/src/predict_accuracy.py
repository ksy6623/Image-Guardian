import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from train import StrongerCNN

def predict_folder(folder_path, model_path="models/stronger_cnn.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    model = StrongerCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ✅ 사용자에게 라벨 직접 입력받기
    label = int(input("이 폴더의 이미지 정답 라벨을 직접 입력하세요 (0=원본, 1=보호): ").strip())

    total = 0
    correct = 0

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(folder_path, filename)

        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()

        if pred == label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"📂 {os.path.basename(folder_path)} 폴더 예측 정확도: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    input_folder = input("예측할 이미지 폴더 경로를 입력하세요: ").strip()
    predict_folder(input_folder)
