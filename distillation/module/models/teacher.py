"""
Teacher Models for Knowledge Distillation

Teacher 모델은 더 많은 파라미터와 복잡한 구조를 가진 모델입니다.
Student 모델에게 "지식"을 전달하는 역할을 합니다.
"""

import torch
import torch.nn as nn


class DeepNN(nn.Module):
    """
    깊은 CNN Teacher 모델 (~1.18M parameters)
    
    구조:
    - 4개의 Convolutional layers (128 -> 64 -> 64 -> 32 filters)
    - 2개의 MaxPooling layers
    - Fully connected classifier (2048 -> 512 -> 10)
    """
    
    def __init__(self, num_classes: int = 10):
        super(DeepNN, self).__init__()
        
        # Feature extractor: Convolutional layers
        self.features = nn.Sequential(
            # Block 1: 3 -> 128 -> 64
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            
            # Block 2: 64 -> 64 -> 32
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
        )
        
        # Classifier: Fully connected layers
        # Input: 32 * 8 * 8 = 2048
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten: (batch, 32, 8, 8) -> (batch, 2048)
        x = self.classifier(x)
        return x


class ModifiedDeepNNCosine(nn.Module):
    """
    Cosine Similarity Loss를 위한 Teacher 모델
    
    forward()가 (logits, flattened_hidden_representation)을 반환합니다.
    Student와 차원을 맞추기 위해 avg_pool1d를 적용합니다.
    """
    
    def __init__(self, num_classes: int = 10):
        super(ModifiedDeepNNCosine, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        flattened_conv_output = torch.flatten(x, 1)  # (batch, 2048)
        x = self.classifier(flattened_conv_output)
        
        # Student와 차원 맞추기: 2048 -> 1024
        flattened_conv_output_after_pooling = torch.nn.functional.avg_pool1d(
            flattened_conv_output, kernel_size=2
        )
        
        return x, flattened_conv_output_after_pooling


class ModifiedDeepNNRegressor(nn.Module):
    """
    Feature Map Regression을 위한 Teacher 모델
    
    forward()가 (logits, conv_feature_map)을 반환합니다.
    Student의 regressor가 이 feature map을 모방하도록 학습합니다.
    """
    
    def __init__(self, num_classes: int = 10):
        super(ModifiedDeepNNRegressor, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        conv_feature_map = x  # (batch, 32, 8, 8) - 마지막 conv layer 출력
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, conv_feature_map
