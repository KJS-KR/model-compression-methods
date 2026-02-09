"""
Part 1: Knowledge Distillation 개념 및 Baseline 학습

이 스크립트에서 다루는 내용:
1. Knowledge Distillation의 핵심 개념
2. CIFAR-10 데이터셋 로드
3. Teacher 모델 (DeepNN) 학습
4. Student 모델 (LightNN) baseline 학습
5. 두 모델의 성능 비교

Knowledge Distillation 핵심 아이디어 (Hinton et al., 2015):
- 큰 Teacher 모델의 "soft targets"을 작은 Student 모델에게 전달
- Soft targets = softmax(logits / T), T는 temperature
- T가 클수록 확률 분포가 부드러워짐 (soft)
- 이를 통해 Teacher가 학습한 클래스 간 관계 정보를 전달

Loss 함수:
L = α * KD_loss + (1-α) * CE_loss

여기서:
- KD_loss: Teacher의 soft targets와 Student의 soft predictions 간 KL divergence
- CE_loss: Student의 predictions과 ground truth labels 간 Cross Entropy
- α: 두 loss의 가중치 (0~1)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from models import DeepNN, LightNN


# =============================================================================
# 1. 환경 설정
# =============================================================================

def get_device() -> str:
    """사용 가능한 가속기 확인"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()
print(f"Using device: {DEVICE}")

# 하이퍼파라미터
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 10

# 재현성을 위한 시드 설정
SEED = 42
torch.manual_seed(SEED)


# =============================================================================
# 2. 데이터 로드 및 전처리
# =============================================================================

def get_data_loaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """
    CIFAR-10 데이터셋 로드
    
    정규화 값 설명:
    - mean=[0.485, 0.456, 0.406]: ImageNet 기준 RGB 채널별 평균
    - std=[0.229, 0.224, 0.225]: ImageNet 기준 RGB 채널별 표준편차
    - CIFAR-10에서도 일반적으로 이 값을 사용
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    return train_loader, test_loader


# =============================================================================
# 3. 학습 및 평가 함수
# =============================================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str
) -> list[float]:
    """
    일반적인 Cross-Entropy 학습
    
    Args:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        epochs: 학습 에포크 수
        learning_rate: 학습률
        device: 학습 디바이스
    
    Returns:
        각 에포크의 평균 loss 리스트
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    model.train()
    
    epoch_losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return epoch_losses


def test(model: nn.Module, test_loader: DataLoader, device: str) -> float:
    """
    모델 평가
    
    Args:
        model: 평가할 모델
        test_loader: 테스트 데이터 로더
        device: 평가 디바이스
    
    Returns:
        테스트 정확도 (%)
    """
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def count_parameters(model: nn.Module) -> int:
    """모델의 총 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# 4. 메인 실행
# =============================================================================

def main():
    print("=" * 60)
    print("Part 1: Knowledge Distillation - Baseline 학습")
    print("=" * 60)
    
    # 데이터 로드
    print("\n[1] CIFAR-10 데이터셋 로드 중...")
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    print(f"    - 학습 데이터: {len(train_loader.dataset):,} 샘플")
    print(f"    - 테스트 데이터: {len(test_loader.dataset):,} 샘플")
    
    # Teacher 모델 학습
    print("\n[2] Teacher 모델 (DeepNN) 학습")
    print("-" * 40)
    
    torch.manual_seed(SEED)
    teacher = DeepNN(num_classes=NUM_CLASSES).to(DEVICE)
    print(f"    파라미터 수: {count_parameters(teacher):,}")
    
    teacher_losses = train(teacher, train_loader, EPOCHS, LEARNING_RATE, DEVICE)
    teacher_accuracy = test(teacher, test_loader, DEVICE)
    
    # Teacher 모델 저장
    torch.save(teacher.state_dict(), 'teacher_model.pth')
    print("    Teacher 모델 저장: teacher_model.pth")
    
    # Student 모델 학습 (Baseline - KD 없이)
    print("\n[3] Student 모델 (LightNN) Baseline 학습")
    print("-" * 40)
    
    torch.manual_seed(SEED)
    student = LightNN(num_classes=NUM_CLASSES).to(DEVICE)
    print(f"    파라미터 수: {count_parameters(student):,}")
    print(f"    Teacher 대비: {count_parameters(student) / count_parameters(teacher) * 100:.1f}%")
    
    student_losses = train(student, train_loader, EPOCHS, LEARNING_RATE, DEVICE)
    student_accuracy = test(student, test_loader, DEVICE)
    
    # Student baseline 모델 저장
    torch.save(student.state_dict(), 'student_baseline.pth')
    print("    Student baseline 모델 저장: student_baseline.pth")
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"{'모델':<20} {'파라미터':<15} {'정확도':<10}")
    print("-" * 60)
    print(f"{'Teacher (DeepNN)':<20} {count_parameters(teacher):>12,} {teacher_accuracy:>8.2f}%")
    print(f"{'Student (LightNN)':<20} {count_parameters(student):>12,} {student_accuracy:>8.2f}%")
    print("-" * 60)
    print(f"정확도 차이: {teacher_accuracy - student_accuracy:.2f}%")
    print("\n이 차이를 Knowledge Distillation으로 줄이는 것이 목표입니다!")
    
    return {
        'teacher_accuracy': teacher_accuracy,
        'student_baseline_accuracy': student_accuracy,
        'teacher_losses': teacher_losses,
        'student_losses': student_losses
    }


if __name__ == "__main__":
    results = main()
