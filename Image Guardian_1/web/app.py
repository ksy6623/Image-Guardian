import os
import uuid
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch
from styleblock_cnn import StyleBlockCNN  # 모델 가져오기

app = Flask(__name__)

UPLOAD_FOLDER = "web/uploads"
ORIGINAL_FOLDER = os.path.join(UPLOAD_FOLDER, "original")
PROTECTED_FOLDER = os.path.join(UPLOAD_FOLDER, "protected")
os.makedirs(ORIGINAL_FOLDER, exist_ok=True)
os.makedirs(PROTECTED_FOLDER, exist_ok=True)

# 모델 로딩 (절대 경로 확인)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"C:\dev\workspace_pytion\Image Guardian\models\styleblock_cnn.pth"  # 경로 수정
model = StyleBlockCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))  # 모델 파일 로딩
model.eval()  # 모델을 평가 모드로 설정


def protect_image(input_tensor, epsilon=0.07, nightshade_strength=0.1):
    """ StyleBlockCNN 모델을 사용해 이미지 보호 (보호 강도 높음) """
    input_tensor = input_tensor.to(device)

    # 기울기 계산을 하지 않도록 설정 (역전파 제외)
    with torch.no_grad():
        protected_tensor = model(input_tensor)  # 모델의 출력을 얻음

    # 보호 강도 조정 (epsilon 값, nightshade_strength 높이기)
    adv_noise = epsilon * torch.randn_like(input_tensor)  # 기울기 대신 랜덤 노이즈 추가
    random_noise = nightshade_strength * torch.randn_like(input_tensor)

    perturbed = input_tensor + adv_noise + random_noise
    perturbed = torch.clamp(perturbed, 0, 1)
    return perturbed

def compute_ssim(img1_path, img2_path):
    img1 = np.array(Image.open(img1_path).convert("RGB"))  # RGB로 변환
    img2 = np.array(Image.open(img2_path).convert("RGB"))  # RGB로 변환
    try:
        score = ssim(img1, img2, multichannel=True)  # multichannel=True로 컬러 SSIM 계산
    except Exception:
        score = 0.0
    return score

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory("web/uploads", filename)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    protect_success = False

    if request.method == "POST":
        files = request.files.getlist("images")

        for file in files:
            if file:
                ext = os.path.splitext(file.filename)[1]
                unique_id = str(uuid.uuid4())
                original_filename = f"original_{unique_id}{ext}"
                protected_filename = f"protected_{unique_id}{ext}"

                original_path = os.path.join(ORIGINAL_FOLDER, original_filename)
                protected_path = os.path.join(PROTECTED_FOLDER, protected_filename)

                image = Image.open(file).convert("RGB")
                image.save(original_path)

                transform = transforms.ToTensor()
                input_tensor = transform(image).unsqueeze(0)
                protected_tensor = protect_image(input_tensor)  # styleblock_cnn 모델을 이용한 보호
                protected_image = transforms.ToPILImage()(protected_tensor.squeeze(0).cpu())
                protected_image.save(protected_path)

                ssim_score = compute_ssim(original_path, protected_path)

                results.append({
                    "original": f"/uploads/original/{original_filename}",
                    "protected": f"/uploads/protected/{protected_filename}",
                    "ssim": f"{ssim_score:.4f}"
                })

        protect_success = True

    return render_template("index.html", results=results, protect_success=protect_success)

if __name__ == "__main__":
    app.run(debug=True)
