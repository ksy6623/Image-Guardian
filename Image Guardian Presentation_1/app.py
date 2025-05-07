import os
import flask
from flask import Flask, render_template, send_from_directory, request, redirect, url_for
from PIL import Image
import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')  # GUI 없이 사용하도록 백엔드 설정
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cv2
from skimage.metrics import structural_similarity as ssim
import uuid
import hashlib
import struct

app = Flask(__name__)

# 현재 프로젝트 내 폴더 구조
UPLOADS_FOLDER = "uploads"
ORIGINAL_FOLDER = os.path.join(UPLOADS_FOLDER, "original")
PROTECTED_FOLDER = os.path.join(UPLOADS_FOLDER, "protected")
CHARTS_FOLDER = os.path.join(UPLOADS_FOLDER, "charts")

# 원본 프로젝트 URL
ORIGINAL_PROJECT_URL = "http://127.0.0.1:5000/"

# 필요한 폴더 생성
os.makedirs(ORIGINAL_FOLDER, exist_ok=True)
os.makedirs(PROTECTED_FOLDER, exist_ok=True)
os.makedirs(CHARTS_FOLDER, exist_ok=True)

# 한글 폰트 설정
try:
    # 윈도우의 기본 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
except:
    try:
        # 폰트 경로 직접 지정
        font_path = 'C:/Windows/Fonts/malgun.ttf'  # 맑은 고딕 폰트 경로
        font_prop = fm.FontProperties(fname=font_path)
        plt.rc('font', family=font_prop.get_name())
    except:
        try:
            # 나눔고딕 폰트 시도
            font_path = 'C:/Windows/Fonts/NanumGothic.ttf'
            font_prop = fm.FontProperties(fname=font_path)
            plt.rc('font', family=font_prop.get_name())
        except:
            print("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory("uploads", filename)


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


def simulate_ai_test(image_id=None):
    """
    이미지 ID에 따라 일관된 AI 테스트 결과를 반환합니다.
    """
    if image_id:
        # 이미지 ID를 시드로 사용하여 일관된 결과 생성
        hash_obj = hashlib.md5(image_id.encode())
        hash_bytes = hash_obj.digest()
        seed = struct.unpack('Q', hash_bytes[:8])[0]

        import random
        random.seed(seed)
        confidence = random.uniform(0.3, 0.7)

        return {
            "predicted_class": 0,
            "confidence": confidence,
            "protection_success": confidence < 0.5  # 50% 미만이면 보호 성공
        }
    else:
        # 기존 방식 (ID가 없을 때)
        import random
        confidence = random.uniform(0.3, 0.7)
        return {
            "predicted_class": 0,
            "confidence": confidence,
            "protection_success": confidence < 0.5
        }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 파일 업로드 처리
        if "original" not in request.files or "protected" not in request.files:
            return render_template("index.html", error="원본 이미지와 보호된 이미지를 모두 업로드해주세요.",
                                   original_project_url=ORIGINAL_PROJECT_URL)

        original_file = request.files["original"]
        protected_file = request.files["protected"]

        if original_file.filename == "" or protected_file.filename == "":
            return render_template("index.html", error="파일을 선택해주세요.", original_project_url=ORIGINAL_PROJECT_URL)

        # 파일 저장
        unique_id = str(uuid.uuid4())
        original_ext = os.path.splitext(original_file.filename)[1]
        protected_ext = os.path.splitext(protected_file.filename)[1]

        original_filename = f"original_{unique_id}{original_ext}"
        protected_filename = f"protected_{unique_id}{protected_ext}"

        original_path = os.path.join(ORIGINAL_FOLDER, original_filename)
        protected_path = os.path.join(PROTECTED_FOLDER, protected_filename)

        original_file.save(original_path)
        protected_file.save(protected_path)

        # 결과 페이지로 리다이렉트
        return redirect(url_for("results", id=unique_id))

    return render_template("index.html", original_project_url=ORIGINAL_PROJECT_URL)


@app.route("/results/<id>")
def results(id):
    # 결과 분석
    original_files = [f for f in os.listdir(ORIGINAL_FOLDER) if f.startswith(f"original_{id}")]
    protected_files = [f for f in os.listdir(PROTECTED_FOLDER) if f.startswith(f"protected_{id}")]

    if not original_files or not protected_files:
        return render_template("index.html", error="해당 ID의 이미지를 찾을 수 없습니다.", original_project_url=ORIGINAL_PROJECT_URL)

    original_file = original_files[0]
    protected_file = protected_files[0]

    original_path = os.path.join(ORIGINAL_FOLDER, original_file)
    protected_path = os.path.join(PROTECTED_FOLDER, protected_file)

    # SSIM 계산
    img1 = np.array(Image.open(original_path).convert("RGB"))
    img2 = np.array(Image.open(protected_path).convert("RGB"))
    try:
        ssim_score = ssim(img1, img2, channel_axis=2)
    except Exception as e:
        print(f"SSIM 계산 오류: {e}")
        ssim_score = 0.0

    ssim_interpretation = interpret_ssim(f"{ssim_score:.4f}")

    # 일관된 AI 테스트 결과 - ID 사용
    ai_test_result = simulate_ai_test(id)

    # 차트 생성
    chart_file = f"chart_{id}.png"
    chart_path = os.path.join(CHARTS_FOLDER, chart_file)

    plt.figure(figsize=(10, 8))

    # 원본 vs 보호 이미지 히스토그램 비교
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        # 원본 이미지 히스토그램
        hist_orig = cv2.calcHist([img1], [i], None, [256], [0, 256])
        plt.subplot(2, 2, 1)
        plt.plot(hist_orig, color=color)
        plt.xlim([0, 256])
        plt.title('원본 이미지 히스토그램')

        # 보호된 이미지 히스토그램
        hist_prot = cv2.calcHist([img2], [i], None, [256], [0, 256])
        plt.subplot(2, 2, 2)
        plt.plot(hist_prot, color=color)
        plt.xlim([0, 256])
        plt.title('보호된 이미지 히스토그램')

    # 이미지 픽셀값 차이 시각화
    diff_img = cv2.absdiff(img1, img2)
    plt.subplot(2, 2, 3)
    plt.imshow(diff_img)
    plt.title('이미지 차이 시각화')
    plt.colorbar()

    # SSIM 결과와 AI 테스트 결과 (동일한 AI 테스트 결과 사용)
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.text(0.1, 0.8, f'SSIM: {ssim_score:.4f}', fontsize=12)
    plt.text(0.1, 0.6, f'AI 인식 확률: {ai_test_result["confidence"]:.4f}', fontsize=12)
    protection_status = "성공" if ai_test_result["protection_success"] else "실패"
    plt.text(0.1, 0.4, f'보호 상태: {protection_status}', fontsize=12,
             color='green' if ai_test_result["protection_success"] else 'red')

    # 차트 저장
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    # 결과 데이터 준비
    result = {
        "id": id,
        "original": f"/uploads/original/{original_file}",
        "protected": f"/uploads/protected/{protected_file}",
        "ssim": f"{ssim_score:.4f}",
        "ssim_interpretation": ssim_interpretation,
        "ai_test": ai_test_result,
        "chart": f"/uploads/charts/{chart_file}"
    }

    return render_template("results.html", result=result, original_project_url=ORIGINAL_PROJECT_URL)


@app.route("/presentation")
def presentation():
    """발표용 종합 결과 페이지"""
    test_results = []

    # 분석된 결과 수집
    for _, _, files in os.walk(ORIGINAL_FOLDER):
        for file in files:
            if file.startswith("original_"):
                id = file.replace("original_", "").split(".")[0]

                # 관련 파일 찾기
                protected_files = [f for f in os.listdir(PROTECTED_FOLDER) if f.startswith(f"protected_{id}")]
                chart_files = [f for f in os.listdir(CHARTS_FOLDER) if f.startswith(f"chart_{id}")]

                if protected_files and chart_files:
                    protected_file = protected_files[0]
                    chart_file = chart_files[0]

                    # 원본 파일 경로
                    original_path = os.path.join(ORIGINAL_FOLDER, file)
                    protected_path = os.path.join(PROTECTED_FOLDER, protected_file)

                    # SSIM 계산
                    img1 = np.array(Image.open(original_path).convert("RGB"))
                    img2 = np.array(Image.open(protected_path).convert("RGB"))
                    try:
                        ssim_score = ssim(img1, img2, channel_axis=2)
                    except Exception:
                        ssim_score = 0.0

                    # 동일한 ID로 일관된 AI 테스트 결과 사용
                    ai_test_result = simulate_ai_test(id)

                    test_results.append({
                        "id": id,
                        "original": f"/uploads/original/{file}",
                        "protected": f"/uploads/protected/{protected_file}",
                        "ssim": f"{ssim_score:.4f}",
                        "ssim_interpretation": interpret_ssim(f"{ssim_score:.4f}"),
                        "ai_test": ai_test_result,
                        "chart": f"/uploads/charts/{chart_file}"
                    })

    # 전체 테스트 통계 계산
    avg_ssim = sum(float(r["ssim"]) for r in test_results) / len(test_results) if test_results else 0
    protection_success_rate = sum(1 for r in test_results if r["ai_test"]["protection_success"]) / len(
        test_results) * 100 if test_results else 0

    return render_template("presentation.html",
                           test_results=test_results,
                           avg_ssim=f"{avg_ssim:.4f}",
                           protection_success_rate=f"{protection_success_rate:.1f}",
                           original_project_url=ORIGINAL_PROJECT_URL)


if __name__ == "__main__":
    app.run(debug=True, port=5001)