import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from train import ImprovedCNN
def predict_folder(folder_path, model_path="models/improved_cnn.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    model = ImprovedCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total = 0
    correct = 0

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(folder_path, filename)

        # 추론
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # 예측 결과 처리
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()

        # 💡 여기가 수정 포인트!
        if "input_images" in folder_path.lower():
            label = 0
        elif "output_strong" in folder_path.lower():
            label = 1
        else:
            print(f"❗ 폴더 이름에서 라벨을 판단할 수 없습니다: {folder_path}")
            continue

        if pred == label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"📂 {os.path.basename(folder_path)} 폴더 예측 정확도: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    input_folder = input("예측할 이미지 폴더 경로를 입력하세요: ").strip()
    predict_folder(input_folder)
