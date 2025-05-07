# Image-Guardian

# 🛡️ Image Guardian

> 생성형 AI의 무단 이미지 학습을 방지하는 보호 시스템

---

## 📌 프로젝트 소개

**Image Guardian**은 드로잉, 일러스트 등의 이미지가 생성형 AI에 무단으로 학습되지 않도록 보호하는 시스템입니다.  
사람 눈에는 원본과 유사하게 보이지만, AI 모델은 인식하기 어려운 형태로 이미지가 변형됩니다.

---

## 🛠️ 사용 기술

### 💻 개발 환경
- Python 3.x
- Flask (웹 프레임워크)
- Anaconda (conda 환경: `style`)

### 🖼️ 이미지 처리 및 분석
- PIL / Pillow: 이미지 로딩 및 처리
- OpenCV: 히스토그램 분석, 이미지 차이 시각화
- scikit-image: SSIM 계산

### 📊 데이터 시각화
- Matplotlib: 이미지 분석 차트
- NumPy: 배열 및 픽셀 처리

### 🤖 딥러닝
- PyTorch
- StyleBlockCNN (Adversarial Noise 적용)

### 🌐 웹 인터페이스
- HTML / CSS
- Jinja2 템플릿 (Flask 내장)

---

## 🧪 구현 기능 요약

- ✅ 이미지 보호 기능 (AI 학습 방지용 노이즈 삽입)
- ✅ SSIM 계산 (시각적 유사도 평가)
- ✅ AI 인식률 분석 (50% 미만이면 방어 성공)
- ✅ RGB 채널 히스토그램 시각화
- ✅ 원본 vs 보호 이미지 비교
- ✅ 대시보드 형태의 분석 결과 제공

---

## 📈 예시 결과

- 평균 SSIM: **0.49**
- AI 방어 성공률: **42.9%**

> 🤖 *AI가 이미지의 의미를 인식하지 못한 비율입니다.*

---


---
