"""
Part 1: Pruning 개념 및 Baseline 학습

이 스크립트에서 다루는 내용:
1. Pruning의 핵심 개념
2. CIFAR-10 데이터셋 로드
3. CNN 모델 학습 (Baseline)
4. 가중치 분포 시각화
5. 모델 파라미터 분석

Pruning 핵심 아이디어:
- 학습된 모델에서 중요도가 낮은 가중치/뉴런/필터를 제거
- "Lottery Ticket Hypothesis" (Frankle & Carlin, 2019):
  큰 네트워크 안에 성능을 유지하는 작은 서브네트워크가 존재
- 핵심 가정: 모든 파라미터가 동일하게 중요하지 않다

Pruning 분류:
=============
1. Unstructured Pruning (비정형 프루닝)
   - 개별 가중치(weight) 단위로 제거
   - 높은 압축률 가능, 하지만 실제 속도 향상은 제한적
   - 예: 크기가 작은 가중치를 0으로 설정

2. Structured Pruning (정형 프루닝)
   - 필터/채널/레이어 단위로 제거
   - 실제 연산량 감소 및 속도 향상 가능
   - 예: L1 norm이 작은 필터 전체를 제거

중요도 기준:
- Magnitude-based: |w| 크기 기반 (가장 일반적)
- Gradient-based: gradient 크기 기반
- Taylor expansion: 가중치 제거 시 loss 변화량 추정
"""

import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'module'))
from models import CNN


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
# 4. 가중치 분포 시각화
# =============================================================================

def visualize_weight_distribution(model: nn.Module, save_path: str = 'weight_distribution.png'):
    """
    모델의 가중치 분포 시각화

    Pruning 전에 가중치 분포를 확인하면:
    - 0 근처의 가중치가 많다 = pruning에 적합
    - 분포가 넓다 = 중요한 가중치와 불필요한 가중치 구분 가능
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Weight Distribution by Layer (Before Pruning)', fontsize=14)

    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))

    for idx, (name, layer) in enumerate(conv_layers):
        ax = axes[idx // 2][idx % 2]
        weights = layer.weight.data.cpu().numpy().flatten()

        ax.hist(weights, bins=100, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax.set_title(f'{name}\n(shape: {list(layer.weight.shape)}, total: {weights.size:,})')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')

        # 통계 정보 표시
        stats_text = f'mean: {weights.mean():.4f}\nstd: {weights.std():.4f}\n|w|<0.01: {(np.abs(weights) < 0.01).sum():,}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"가중치 분포 시각화 저장: {save_path}")


def analyze_layer_parameters(model: nn.Module):
    """레이어별 파라미터 분석"""
    print("\n[레이어별 파라미터 분석]")
    print(f"{'Layer':<30} {'Shape':<25} {'Parameters':<15} {'비율':<10}")
    print("-" * 80)

    total_params = 0
    layer_info = []

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        layer_info.append((name, list(param.shape), num_params))

    for name, shape, num_params in layer_info:
        ratio = num_params / total_params * 100
        print(f"{name:<30} {str(shape):<25} {num_params:>12,} {ratio:>8.1f}%")

    print("-" * 80)
    print(f"{'Total':<30} {'':<25} {total_params:>12,} {'100.0%':>10}")

    return total_params


# =============================================================================
# 5. 메인 실행
# =============================================================================

def main():
    print("=" * 60)
    print("Part 1: Pruning - 개념 및 Baseline 학습")
    print("=" * 60)

    # 데이터 로드
    print("\n[1] CIFAR-10 데이터셋 로드 중...")
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    print(f"    - 학습 데이터: {len(train_loader.dataset):,} 샘플")
    print(f"    - 테스트 데이터: {len(test_loader.dataset):,} 샘플")

    # 모델 학습
    print("\n[2] CNN 모델 학습 (Baseline)")
    print("-" * 40)

    torch.manual_seed(SEED)
    model = CNN(num_classes=NUM_CLASSES).to(DEVICE)
    print(f"    파라미터 수: {count_parameters(model):,}")

    train_losses = train(model, train_loader, EPOCHS, LEARNING_RATE, DEVICE)
    baseline_accuracy = test(model, test_loader, DEVICE)

    # 모델 저장
    torch.save(model.state_dict(), 'baseline_model.pth')
    print("    Baseline 모델 저장: baseline_model.pth")

    # 레이어별 파라미터 분석
    print("\n[3] 레이어별 파라미터 분석")
    print("-" * 40)
    total_params = analyze_layer_parameters(model)

    # 가중치 분포 시각화
    print("\n[4] 가중치 분포 시각화")
    print("-" * 40)
    visualize_weight_distribution(model)

    # Pruning 전 가중치 통계
    print("\n[5] Pruning 관련 가중치 통계")
    print("-" * 40)

    total_weights = 0
    near_zero_weights = 0
    thresholds = [0.001, 0.01, 0.05, 0.1]

    for name, param in model.named_parameters():
        if 'weight' in name:
            weights = param.data.abs().cpu()
            total_weights += weights.numel()
            near_zero_weights += (weights < 0.01).sum().item()

    print(f"    총 가중치 수: {total_weights:,}")
    print(f"    |w| < 0.01인 가중치: {near_zero_weights:,} ({near_zero_weights/total_weights*100:.1f}%)")

    print(f"\n    Threshold별 제거 가능한 가중치 비율:")
    for threshold in thresholds:
        count = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                count += (param.data.abs().cpu() < threshold).sum().item()
        print(f"    |w| < {threshold}: {count:,} ({count/total_weights*100:.1f}%)")

    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"    모델: CNN ({total_params:,} parameters)")
    print(f"    Baseline 정확도: {baseline_accuracy:.2f}%")
    print(f"\n    이 모델에 다양한 Pruning 기법을 적용하여")
    print(f"    정확도를 최대한 유지하면서 모델을 압축하는 것이 목표입니다!")

    return {
        'baseline_accuracy': baseline_accuracy,
        'train_losses': train_losses,
        'total_params': total_params
    }


if __name__ == "__main__":
    results = main()
