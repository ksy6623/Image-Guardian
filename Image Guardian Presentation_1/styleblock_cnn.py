import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleBlockCNN(nn.Module):
    def __init__(self):
        super(StyleBlockCNN, self).__init__()

        # 원본 모델의 정확한 구조로 수정
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # features.0 - 16 채널
            nn.ReLU(inplace=True),  # features.1
            nn.MaxPool2d(2, 2),  # features.2
            nn.Conv2d(16, 32, 3, padding=1),  # features.3 (현재 모델) / features.4 (원본 모델)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),  # features.6 (현재 모델) / features.8 (원본 모델)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # 원본 모델의 분류기 구조로 수정
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 128),  # classifier.1 - [128, 1024]
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 2)  # classifier.4 - [2, 128] (2 클래스 분류)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)  # 1024 크기로 변경
        x = self.classifier(x)
        return x