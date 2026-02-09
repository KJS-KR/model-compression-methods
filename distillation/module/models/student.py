"""
Student Models for Knowledge Distillation

Student 모델은 Teacher보다 작고 가벼운 모델입니다.
Teacher의 지식을 전달받아 더 나은 성능을 얻는 것이 목표입니다.
"""

import torch
import torch.nn as nn


class LightNN(nn.Module):
    """
    경량 CNN Student 모델 (~267K parameters)
    
    구조:
    - 2개의 Convolutional layers (16 -> 16 filters)
    - 2개의 MaxPooling layers
    - Fully connected classifier (1024 -> 256 -> 10)
    
    Teacher 대비:
    - 파라미터 수: ~22% (267K vs 1.18M)
    - Conv layers: 2개 vs 4개
    - Filters: 16 vs 128/64/32
    """
    
    def __init__(self, num_classes: int = 10):
        super(LightNN, self).__init__()
        
        # Feature extractor: 더 적은 필터와 레이어
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
        )
        
        # Classifier: 더 작은 hidden layer
        # Input: 16 * 8 * 8 = 1024
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten: (batch, 16, 8, 8) -> (batch, 1024)
        x = self.classifier(x)
        return x


class ModifiedLightNNCosine(nn.Module):
    """
    Cosine Similarity Loss를 위한 Student 모델
    
    forward()가 (logits, flattened_hidden_representation)을 반환합니다.
    Teacher의 hidden representation과 유사해지도록 학습합니다.
    """
    
    def __init__(self, num_classes: int = 10):
        super(ModifiedLightNNCosine, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        flattened_conv_output = torch.flatten(x, 1)  # (batch, 1024)
        x = self.classifier(flattened_conv_output)
        return x, flattened_conv_output


class ModifiedLightNNRegressor(nn.Module):
    """
    Feature Map Regression을 위한 Student 모델 (FitNets 스타일)
    
    추가된 regressor layer가 Student의 feature map을 
    Teacher의 feature map 차원으로 변환합니다.
    
    핵심 아이디어:
    - Student: 16 filters -> Regressor: 32 filters (Teacher와 동일)
    - MSE Loss로 Teacher의 feature map을 모방
    """
    
    def __init__(self, num_classes: int = 10):
        super(ModifiedLightNNRegressor, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Regressor: Student feature map -> Teacher feature map 차원 변환
        # 16 channels -> 32 channels (Teacher의 마지막 conv 출력과 동일)
        self.regressor = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        regressor_output = self.regressor(x)  # (batch, 32, 8, 8)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, regressor_output
