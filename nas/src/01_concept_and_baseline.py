"""
Part 1: NAS 개념 및 Baseline 학습

이 스크립트에서 다루는 내용:
1. Neural Architecture Search (NAS)의 핵심 개념
2. 탐색 공간(Search Space) 정의 및 분석
3. CNN 모델 학습 (Baseline)
4. 샘플 아키텍처 시각화 및 비교

NAS 핵심 아이디어:
- 사람이 설계하는 대신, 알고리즘이 최적의 아키텍처를 자동으로 탐색
- "Architecture Engineering" → "Architecture Search"로 패러다임 전환

NAS의 3가지 핵심 요소:
=======================
1. Search Space (탐색 공간)
   - 어떤 아키텍처를 탐색할 것인가?
   - 예: 필터 수, 커널 크기, 레이어 수, 연결 방식 등
   - 탐색 공간이 클수록 좋은 아키텍처를 찾을 가능성 ↑, 탐색 비용 ↑

2. Search Strategy (탐색 전략)
   - 어떻게 탐색할 것인가?
   - Random Search: 무작위 샘플링 (가장 간단, 놀라울 정도로 효과적)
   - Evolutionary: 유전 알고리즘 기반 (mutation, selection, crossover)
   - RL-based: 강화학습 기반 (Zoph & Le, 2017)
   - Gradient-based: 미분 가능한 탐색 (DARTS, Liu et al., 2019)

3. Performance Estimation (성능 추정)
   - 후보 아키텍처의 성능을 어떻게 빠르게 평가할 것인가?
   - Full training: 정확하지만 매우 느림
   - Reduced training: 적은 에포크로 근사 (이 튜토리얼에서 사용)
   - Weight sharing: 파라미터를 공유하여 학습 비용 절감

참고 논문:
- Zoph & Le, 2017: Neural Architecture Search with RL
- Real et al., 2019: Regularized Evolution for Image Classifier Architecture Search
- Li & Talwalkar, 2020: Random Search and Reproducibility for NAS
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
from models import (
    CNN, FlexibleCNN, SEARCH_SPACE,
    sample_architecture, get_search_space_size, architecture_to_string
)


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
# 4. 탐색 공간 분석
# =============================================================================

def analyze_search_space():
    """
    탐색 공간의 구성 요소와 크기를 분석

    탐색 공간이 클수록:
    - 더 좋은 아키텍처를 찾을 가능성이 높음
    - 탐색에 필요한 시간/비용이 증가
    - 효율적인 탐색 전략이 더 중요해짐
    """
    print("\n[탐색 공간 분석]")
    print(f"{'구성 요소':<25} {'후보값':<35} {'선택 수'}")
    print("-" * 70)

    n_filters = len(SEARCH_SPACE['filters'])
    n_kernels = len(SEARCH_SPACE['kernel_sizes'])
    n_fc = len(SEARCH_SPACE['fc_hidden'])
    n_conv = SEARCH_SPACE['num_conv_layers']

    print(f"{'Conv 필터 수':<25} {str(SEARCH_SPACE['filters']):<35} {n_filters}")
    print(f"{'Conv 커널 크기':<25} {str(SEARCH_SPACE['kernel_sizes']):<35} {n_kernels}")
    print(f"{'FC Hidden Units':<25} {str(SEARCH_SPACE['fc_hidden']):<35} {n_fc}")
    print(f"{'Conv 레이어 수':<25} {str(n_conv) + ' (고정)':<35}")
    print(f"{'MaxPool 위치':<25} {str(SEARCH_SPACE['pool_after']) + ' (고정, 0-indexed)':<35}")
    print("-" * 70)

    total_size = get_search_space_size()
    print(f"\n전체 탐색 공간 크기: {total_size:,} 가지 아키텍처")
    print(f"  = (필터 {n_filters} x 커널 {n_kernels})^{n_conv} x FC {n_fc}")
    print(f"  = {n_filters * n_kernels}^{n_conv} x {n_fc}")
    print(f"  = {(n_filters * n_kernels) ** n_conv:,} x {n_fc}")

    return total_size


# =============================================================================
# 5. 샘플 아키텍처 시각화
# =============================================================================

def visualize_sample_architectures(num_samples: int = 10,
                                    save_path: str = 'results/sample_architectures.png'):
    """
    탐색 공간에서 샘플링한 아키텍처들의 파라미터 수 비교 및 필터 구성 시각화

    다양한 아키텍처가 얼마나 다른 크기를 가지는지 보여줍니다.
    """
    # 샘플 아키텍처 생성
    architectures = []
    for i in range(num_samples):
        arch = sample_architecture(seed=SEED + i)
        model = FlexibleCNN(arch, num_classes=NUM_CLASSES)
        params = count_parameters(model)
        architectures.append({
            'arch': arch,
            'params': params,
            'name': architecture_to_string(arch)
        })

    # 파라미터 수 기준 정렬
    architectures.sort(key=lambda x: x['params'])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Plot 1: 파라미터 수 비교 (수평 바 차트) ---
    ax1 = axes[0]
    names = [f"Arch {i+1}" for i in range(num_samples)]
    params_list = [a['params'] for a in architectures]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, num_samples))

    bars = ax1.barh(names, params_list, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Number of Parameters')
    ax1.set_title(f'Parameter Count of {num_samples} Random Architectures')

    # 바 옆에 파라미터 수 표시
    for bar, params in zip(bars, params_list):
        ax1.text(bar.get_width() + max(params_list) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{params:,}', va='center', fontsize=8)

    # Baseline 표시
    baseline_params = count_parameters(CNN(num_classes=NUM_CLASSES))
    ax1.axvline(x=baseline_params, color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline_params:,})')
    ax1.legend(fontsize=9)

    # --- Plot 2: 필터 구성 히트맵 ---
    ax2 = axes[1]
    filter_matrix = np.array([a['arch']['filters'] for a in architectures])
    im = ax2.imshow(filter_matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_xlabel('Conv Layer Index')
    ax2.set_ylabel('Architecture')
    ax2.set_title('Filter Configuration Heatmap')
    ax2.set_xticks(range(SEARCH_SPACE['num_conv_layers']))
    ax2.set_xticklabels([f'Conv {i+1}' for i in range(SEARCH_SPACE['num_conv_layers'])])
    ax2.set_yticks(range(num_samples))
    ax2.set_yticklabels(names)

    # 히트맵 셀에 값 표시
    for i in range(num_samples):
        for j in range(SEARCH_SPACE['num_conv_layers']):
            ax2.text(j, i, str(filter_matrix[i, j]),
                     ha='center', va='center', fontsize=9, fontweight='bold')

    plt.colorbar(im, ax=ax2, label='Filter Count')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"샘플 아키텍처 시각화 저장: {save_path}")

    # 통계 출력
    params_array = np.array(params_list)
    print(f"\n[샘플 아키텍처 통계]")
    print(f"    최소 파라미터: {params_array.min():,}")
    print(f"    최대 파라미터: {params_array.max():,}")
    print(f"    평균 파라미터: {params_array.mean():,.0f}")
    print(f"    Baseline 파라미터: {baseline_params:,}")

    return architectures


# =============================================================================
# 6. 메인 실행
# =============================================================================

def main():
    print("=" * 60)
    print("Part 1: NAS - 개념 및 Baseline 학습")
    print("=" * 60)

    # 1. 탐색 공간 분석
    print("\n[1] 탐색 공간(Search Space) 분석")
    print("-" * 40)
    total_size = analyze_search_space()

    # 2. 데이터 로드
    print("\n[2] CIFAR-10 데이터셋 로드 중...")
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    print(f"    - 학습 데이터: {len(train_loader.dataset):,} 샘플")
    print(f"    - 테스트 데이터: {len(test_loader.dataset):,} 샘플")

    # 3. Baseline 모델 학습
    print("\n[3] CNN 모델 학습 (Baseline - 사람이 설계한 고정 아키텍처)")
    print("-" * 40)

    torch.manual_seed(SEED)
    model = CNN(num_classes=NUM_CLASSES).to(DEVICE)
    baseline_params = count_parameters(model)
    print(f"    파라미터 수: {baseline_params:,}")

    train_losses = train(model, train_loader, EPOCHS, LEARNING_RATE, DEVICE)
    baseline_accuracy = test(model, test_loader, DEVICE)

    # 모델 저장
    os.makedirs('data/trained_models', exist_ok=True)
    torch.save(model.state_dict(), 'data/trained_models/baseline_model.pth')
    print("    Baseline 모델 저장: data/trained_models/baseline_model.pth")

    # 4. 샘플 아키텍처 시각화
    print("\n[4] 탐색 공간 샘플 아키텍처 시각화")
    print("-" * 40)
    sample_archs = visualize_sample_architectures()

    # 5. FlexibleCNN 동작 확인
    print("\n[5] FlexibleCNN 동작 확인")
    print("-" * 40)
    test_arch = sample_architecture(seed=SEED)
    test_model = FlexibleCNN(test_arch, num_classes=NUM_CLASSES)
    test_input = torch.randn(1, 3, 32, 32)
    test_output = test_model(test_input)
    print(f"    아키텍처: {architecture_to_string(test_arch)}")
    print(f"    파라미터 수: {count_parameters(test_model):,}")
    print(f"    입력 shape: {list(test_input.shape)}")
    print(f"    출력 shape: {list(test_output.shape)}")

    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"    탐색 공간 크기: {total_size:,} 가지 아키텍처")
    print(f"    Baseline 모델: CNN ({baseline_params:,} parameters)")
    print(f"    Baseline 정확도: {baseline_accuracy:.2f}%")
    print(f"\n    다음 단계:")
    print(f"    - Part 2: Random Search로 탐색 공간에서 최적 아키텍처 탐색")
    print(f"    - Part 3: Evolutionary Search로 더 효율적인 탐색")
    print(f"    - Part 4: 전체 실험 결과 종합 분석")

    return {
        'baseline_accuracy': baseline_accuracy,
        'train_losses': train_losses,
        'total_params': baseline_params,
        'search_space_size': total_size
    }


if __name__ == "__main__":
    results = main()
