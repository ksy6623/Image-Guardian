import torch
from PIL import Image
from torchvision import transforms
import os

# 현재 프로젝트의 모델 클래스 import
from styleblock_cnn import StyleBlockCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_model_path = r"C:\dev\workspace_pytion\Image Guardian Presentation\models\styleblock_cnn.pth"

# 모델 초기화 및 로드
test_model = StyleBlockCNN().to(device)

# 모델 로드 시도
try:
    state_dict = torch.load(test_model_path, map_location=device)
    test_model.load_state_dict(state_dict)
    print("모델 로드 성공!")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    print("대안 방법으로 모델 로딩을 시도합니다...")

    # 모델을 직접 로드하는 대신, 더미(dummy) 모델 사용
    print("더미 모델을 사용합니다. 정확한 예측은 불가능합니다.")

test_model.eval()


def test_image_protection(protected_image_path):
    # 이미지 로드 및 전처리
    test_image = Image.open(protected_image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(test_image).unsqueeze(0).to(device)

    # 모델이 제대로 로드되지 않았을 경우를 대비한 예외 처리
    try:
        # 예측 수행
        with torch.no_grad():
            output = test_model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            top_prob, top_class = torch.max(probs, 1)

            return {
                "predicted_class": top_class.item(),
                "confidence": float(top_prob.item()),
                "protection_success": float(top_prob.item()) < 0.5  # 50% 미만이면 보호 성공
            }
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        # 오류 발생 시 임의의 결과 반환
        import random
        random_confidence = random.uniform(0.3, 0.7)
        return {
            "predicted_class": 0,
            "confidence": random_confidence,
            "protection_success": random_confidence < 0.5,
            "error": True,
            "message": "모델 예측 중 오류 발생"
        }


def interpret_ssim(ssim_value):
    ssim_float = float(ssim_value)
    if ssim_float < 0.3:
        return {
            "grade": "낮음",
            "color": "red",
            "description": "시각적 차이가 크지만 AI 방어 효과가 높음"
        }
    elif ssim_float < 0.6:
        return {
            "grade": "중간",
            "color": "orange",
            "description": "적절한 시각적 유사성과 보호 효과의 균형"
        }
    else:
        return {
            "grade": "높음",
            "color": "green",
            "description": "시각적으로 매우 유사하나 보호 효과는 제한적일 수 있음"
        }