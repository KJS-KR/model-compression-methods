"""
Part 3: Feature-based Distillation 구현

이 스크립트에서 다루는 내용:
1. Cosine Similarity Loss - Hidden Representation 매칭
2. Feature Map + Regressor (FitNets 스타일) - MSE Loss
3. 각 방법의 장단점 비교

Feature-based Distillation 개념:
================================

Soft Target KD는 출력층(logits)만 사용하지만,
Feature-based는 중간 레이어의 representation도 활용합니다.

방법 1: Cosine Similarity Loss
-----------------------------
- Teacher와 Student의 hidden representation 유사도 최대화
- Loss = 1 - cosine_similarity(teacher_hidden, student_hidden)
- 장점: 방향(direction)만 맞추면 됨, 크기는 무관
- 단점: 차원이 다르면 맞춰야 함 (avg_pool 등)

방법 2: Feature Map + Regressor (FitNets, Romero et al. 2015)
------------------------------------------------------------
- Student에 추가 레이어(regressor)를 붙여서 Teacher feature map 모방
- Loss = MSE(regressor(student_feature), teacher_feature)
- 장점: 학습 가능한 레이어가 차원 변환 담당
- 단점: 추가 파라미터 필요 (학습 시에만)

핵심 통찰:
- Hidden representation에는 출력층보다 더 풍부한 정보가 있음
- 하지만 1:1 매칭이 항상 최선은 아님 (다른 구조 = 다른 representation)
- Regressor를 통해 "변환"을 학습하게 하는 것이 더 유연함
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from models import (
    DeepNN,
    LightNN,
    ModifiedDeepNNCosine,
    ModifiedLightNNCosine,
    ModifiedDeepNNRegressor,
    ModifiedLightNNRegressor
)


# =============================================================================
# 1. 환경 설정
# =============================================================================

def get_device() -> str:
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

# Feature Distillation 하이퍼파라미터
HIDDEN_REP_LOSS_WEIGHT = 0.25
CE_LOSS_WEIGHT = 0.75

SEED = 42


# =============================================================================
# 2. 데이터 로드
# =============================================================================

def get_data_loaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=2, pin_memory=True if DEVICE == "cuda" else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=2, pin_memory=True if DEVICE == "cuda" else False)
    
    return train_loader, test_loader


# =============================================================================
# 3. Cosine Similarity Loss 학습
# =============================================================================

def train_cosine_loss(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    hidden_rep_loss_weight: float,
    ce_loss_weight: float,
    device: str
) -> list[float]:
    """
    Cosine Similarity Loss를 사용한 Knowledge Distillation
    
    Teacher와 Student의 hidden representation이 
    같은 방향을 가리키도록 학습합니다.
    
    CosineEmbeddingLoss:
    - target=1: 유사도 최대화 (같은 방향)
    - target=-1: 유사도 최소화 (반대 방향)
    
    Loss = hidden_rep_loss_weight * cosine_loss + ce_loss_weight * ce_loss
    """
    ce_loss = nn.CrossEntropyLoss()
    cosine_loss = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    
    teacher.to(device)
    student.to(device)
    teacher.eval()
    student.train()
    
    epoch_losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Teacher forward (frozen)
            with torch.no_grad():
                _, teacher_hidden = teacher(inputs)
            
            # Student forward
            student_logits, student_hidden = student(inputs)
            
            # Cosine Similarity Loss
            # target = 1 (유사도 최대화)
            target = torch.ones(inputs.size(0)).to(device)
            hidden_rep_loss = cosine_loss(student_hidden, teacher_hidden, target)
            
            # CE Loss
            label_loss = ce_loss(student_logits, labels)
            
            # Combined Loss
            loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return epoch_losses


# =============================================================================
# 4. MSE Feature Map Loss 학습 (FitNets 스타일)
# =============================================================================

def train_mse_loss(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    feature_map_weight: float,
    ce_loss_weight: float,
    device: str
) -> list[float]:
    """
    MSE Loss를 사용한 Feature Map Distillation (FitNets)
    
    Student의 regressor가 Teacher의 feature map을 모방하도록 학습합니다.
    
    핵심:
    - Student feature map: (batch, 16, 8, 8)
    - Student regressor output: (batch, 32, 8, 8) -> Teacher와 동일
    - Teacher feature map: (batch, 32, 8, 8)
    
    Loss = feature_map_weight * MSE + ce_loss_weight * CE
    """
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    
    teacher.to(device)
    student.to(device)
    teacher.eval()
    student.train()
    
    epoch_losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Teacher forward (frozen)
            with torch.no_grad():
                _, teacher_feature_map = teacher(inputs)
            
            # Student forward
            student_logits, regressor_feature_map = student(inputs)
            
            # MSE Loss between feature maps
            hidden_rep_loss = mse_loss(regressor_feature_map, teacher_feature_map)
            
            # CE Loss
            label_loss = ce_loss(student_logits, labels)
            
            # Combined Loss
            loss = feature_map_weight * hidden_rep_loss + ce_loss_weight * label_loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return epoch_losses


# =============================================================================
# 5. 테스트 함수 (multiple outputs)
# =============================================================================

def test_multiple_outputs(model: nn.Module, test_loader: DataLoader, device: str) -> float:
    """모델 평가 (logits만 사용, hidden representation 무시)"""
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)  # 두 번째 출력(hidden) 무시
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def test(model: nn.Module, test_loader: DataLoader, device: str) -> float:
    """일반 모델 평가"""
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


def train_baseline(model, train_loader, epochs, learning_rate, device):
    """일반 CE 학습"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    model.train()
    
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
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")


# =============================================================================
# 6. Feature Map 차원 확인
# =============================================================================

def check_feature_dimensions(device: str):
    """Teacher와 Student의 feature map 차원 확인"""
    print("\n[Feature Map 차원 확인]")
    
    sample_input = torch.randn(1, 3, 32, 32).to(device)
    
    # 일반 모델
    teacher = DeepNN().to(device)
    student = LightNN().to(device)
    
    teacher_features = teacher.features(sample_input)
    student_features = student.features(sample_input)
    
    print(f"Teacher feature map: {teacher_features.shape}")  # (1, 32, 8, 8)
    print(f"Student feature map: {student_features.shape}")  # (1, 16, 8, 8)
    
    # Modified 모델
    teacher_cosine = ModifiedDeepNNCosine().to(device)
    student_cosine = ModifiedLightNNCosine().to(device)
    
    _, teacher_hidden = teacher_cosine(sample_input)
    _, student_hidden = student_cosine(sample_input)
    
    print(f"\n[Cosine Loss용]")
    print(f"Teacher hidden (after avg_pool): {teacher_hidden.shape}")  # (1, 1024)
    print(f"Student hidden (flattened): {student_hidden.shape}")  # (1, 1024)
    
    teacher_reg = ModifiedDeepNNRegressor().to(device)
    student_reg = ModifiedLightNNRegressor().to(device)
    
    _, teacher_fm = teacher_reg(sample_input)
    _, student_fm = student_reg(sample_input)
    
    print(f"\n[MSE Loss용 (FitNets)]")
    print(f"Teacher feature map: {teacher_fm.shape}")  # (1, 32, 8, 8)
    print(f"Student regressor output: {student_fm.shape}")  # (1, 32, 8, 8)


# =============================================================================
# 7. 메인 실행
# =============================================================================

def main():
    print("=" * 60)
    print("Part 3: Feature-based Distillation")
    print("=" * 60)
    
    # 데이터 로드
    print("\n[1] 데이터 로드 중...")
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    
    # Feature Map 차원 확인
    check_feature_dimensions(DEVICE)
    
    # Teacher 모델 준비
    print("\n[2] Teacher 모델 준비...")
    teacher_base = DeepNN(num_classes=NUM_CLASSES).to(DEVICE)
    
    try:
        teacher_base.load_state_dict(torch.load('teacher_model.pth', map_location=DEVICE, weights_only=True))
        print("    저장된 Teacher 모델 로드 완료")
    except FileNotFoundError:
        print("    Teacher 모델 학습 중...")
        torch.manual_seed(SEED)
        train_baseline(teacher_base, train_loader, EPOCHS, LEARNING_RATE, DEVICE)
        torch.save(teacher_base.state_dict(), 'teacher_model.pth')
    
    teacher_accuracy = test(teacher_base, test_loader, DEVICE)
    
    # Student baseline
    print("\n[3] Student Baseline 학습...")
    torch.manual_seed(SEED)
    student_baseline = LightNN(num_classes=NUM_CLASSES).to(DEVICE)
    train_baseline(student_baseline, train_loader, EPOCHS, LEARNING_RATE, DEVICE)
    baseline_accuracy = test(student_baseline, test_loader, DEVICE)
    
    # =============================================
    # 방법 1: Cosine Similarity Loss
    # =============================================
    print("\n[4] Cosine Similarity Loss 학습...")
    print("-" * 40)
    
    # Teacher를 Modified 버전으로 변환 (같은 weights 사용)
    modified_teacher_cosine = ModifiedDeepNNCosine(num_classes=NUM_CLASSES).to(DEVICE)
    modified_teacher_cosine.load_state_dict(teacher_base.state_dict())
    
    torch.manual_seed(SEED)
    modified_student_cosine = ModifiedLightNNCosine(num_classes=NUM_CLASSES).to(DEVICE)
    
    cosine_losses = train_cosine_loss(
        teacher=modified_teacher_cosine,
        student=modified_student_cosine,
        train_loader=train_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        hidden_rep_loss_weight=HIDDEN_REP_LOSS_WEIGHT,
        ce_loss_weight=CE_LOSS_WEIGHT,
        device=DEVICE
    )
    
    cosine_accuracy = test_multiple_outputs(modified_student_cosine, test_loader, DEVICE)
    torch.save(modified_student_cosine.state_dict(), 'student_cosine.pth')
    
    # =============================================
    # 방법 2: MSE Feature Map Loss (FitNets)
    # =============================================
    print("\n[5] MSE Feature Map Loss 학습 (FitNets)...")
    print("-" * 40)
    
    modified_teacher_reg = ModifiedDeepNNRegressor(num_classes=NUM_CLASSES).to(DEVICE)
    modified_teacher_reg.load_state_dict(teacher_base.state_dict())
    
    torch.manual_seed(SEED)
    modified_student_reg = ModifiedLightNNRegressor(num_classes=NUM_CLASSES).to(DEVICE)
    
    mse_losses = train_mse_loss(
        teacher=modified_teacher_reg,
        student=modified_student_reg,
        train_loader=train_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        feature_map_weight=HIDDEN_REP_LOSS_WEIGHT,
        ce_loss_weight=CE_LOSS_WEIGHT,
        device=DEVICE
    )
    
    mse_accuracy = test_multiple_outputs(modified_student_reg, test_loader, DEVICE)
    torch.save(modified_student_reg.state_dict(), 'student_mse_regressor.pth')
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약: Feature-based Distillation")
    print("=" * 60)
    print(f"{'방법':<35} {'정확도':<10} {'vs Baseline':<12}")
    print("-" * 60)
    print(f"{'Teacher (DeepNN)':<35} {teacher_accuracy:>8.2f}%")
    print(f"{'Student Baseline (CE only)':<35} {baseline_accuracy:>8.2f}%")
    print(f"{'Student + Cosine Loss':<35} {cosine_accuracy:>8.2f}% {cosine_accuracy - baseline_accuracy:>+10.2f}%")
    print(f"{'Student + MSE Regressor (FitNets)':<35} {mse_accuracy:>8.2f}% {mse_accuracy - baseline_accuracy:>+10.2f}%")
    print("-" * 60)
    
    print("\n분석:")
    if mse_accuracy > cosine_accuracy:
        print("- MSE Regressor가 더 좋은 성능을 보임")
        print("- Regressor가 차원 변환을 유연하게 학습하기 때문")
    else:
        print("- Cosine Loss가 더 좋은 성능을 보임")
        print("- 방향만 맞추는 것이 이 데이터셋에 더 적합함")
    
    return {
        'teacher_accuracy': teacher_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'cosine_accuracy': cosine_accuracy,
        'mse_accuracy': mse_accuracy,
        'cosine_losses': cosine_losses,
        'mse_losses': mse_losses
    }


if __name__ == "__main__":
    results = main()
