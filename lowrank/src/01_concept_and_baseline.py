"""
Part 1: Low-Rank Approximation 개념 및 Baseline 학습

이 스크립트에서 다루는 내용:
1. Low-Rank Approximation의 핵심 개념
2. CIFAR-10 데이터셋 로드
3. CNN 모델 학습 (Baseline)
4. 레이어별 특이값(Singular Value) 분석
5. Rank vs 복원 오차 곡선

Low-Rank Approximation 핵심 아이디어:
- 가중치 행렬 W ∈ ℝ^(m×n)을 더 작은 행렬의 곱으로 분해
- W ≈ A @ B, where A ∈ ℝ^(m×r), B ∈ ℝ^(r×n)
- 원본 파라미터: m × n → 분해 후: r × (m + n)
- r < mn/(m+n) 이면 압축 효과

SVD (Singular Value Decomposition):
====================================
  W = U @ Σ @ V^T

여기서:
- U: (m × m) 좌특이벡터 행렬
- Σ: (m × n) 특이값 대각행렬, σ₁ ≥ σ₂ ≥ ... ≥ 0
- V: (n × n) 우특이벡터 행렬

Rank-r 근사:
  W_r = U_r @ Σ_r @ V_r^T (상위 r개 특이값만 사용)

Eckart–Young 정리:
  "Frobenius norm에서 최적의 rank-r 근사는 SVD truncation"

참고 논문:
- Denton et al., 2014 - Exploiting Linear Structure Within CNNs
- Jaderberg et al., 2014 - Speeding up CNNs with Low Rank Expansions
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
from models import CNN, get_reconstruction_error


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
    """CIFAR-10 데이터셋 로드"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True if DEVICE == "cuda" else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True if DEVICE == "cuda" else False
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
    """일반적인 Cross-Entropy 학습"""
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
    """모델 평가"""
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
# 4. 특이값 분석
# =============================================================================

def analyze_singular_values(model: nn.Module,
                             save_path: str = 'results/singular_value_analysis.png'):
    """
    모델의 주요 레이어별 특이값 분석

    특이값이 빠르게 감소하는 레이어 = low-rank 분해에 적합
    특이값이 천천히 감소하는 레이어 = 높은 rank가 필요
    """
    layers_to_analyze = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data.cpu()
            layers_to_analyze.append((name, W, 'Linear'))
        elif isinstance(module, nn.Conv2d):
            W = module.weight.data.cpu()
            out_c, in_c, k_h, k_w = W.shape
            W_reshaped = W.reshape(out_c * k_h * k_w, in_c)
            layers_to_analyze.append((name, W_reshaped, 'Conv2d'))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Singular Value Analysis by Layer', fontsize=14)

    for idx, (name, W, layer_type) in enumerate(layers_to_analyze):
        row, col = idx // 3, idx % 3
        ax = axes[row][col]

        _, S, _ = torch.linalg.svd(W, full_matrices=False)
        S = S.numpy()

        # 특이값 분포 (log scale)
        ax.semilogy(S, 'b-', linewidth=1.5)
        ax.set_title(f'{name} ({layer_type})\nshape: {list(W.shape)}')
        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('Singular Value (log)')
        ax.grid(True, alpha=0.3)

        # 에너지 90%, 95%, 99% 기준선
        energy = np.cumsum(S ** 2) / np.sum(S ** 2)
        for threshold, color in [(0.90, 'green'), (0.95, 'orange'), (0.99, 'red')]:
            rank_at = np.searchsorted(energy, threshold) + 1
            ax.axvline(x=rank_at, color=color, linestyle='--', alpha=0.7,
                       label=f'{threshold*100:.0f}% energy: rank={rank_at}')

        ax.legend(fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"특이값 분석 시각화 저장: {save_path}")


def analyze_cumulative_energy(model: nn.Module,
                               save_path: str = 'results/cumulative_energy.png'):
    """누적 에너지 곡선 시각화"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data.cpu()
        elif isinstance(module, nn.Conv2d):
            W = module.weight.data.cpu()
            out_c, in_c, k_h, k_w = W.shape
            W = W.reshape(out_c * k_h * k_w, in_c)
        else:
            continue

        _, S, _ = torch.linalg.svd(W, full_matrices=False)
        S = S.numpy()
        energy = np.cumsum(S ** 2) / np.sum(S ** 2)
        ranks_normalized = np.arange(1, len(S) + 1) / len(S)

        ax.plot(ranks_normalized, energy, linewidth=2,
                label=f'{name} (max_rank={len(S)})')

    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% energy')
    ax.set_xlabel('Rank / Max Rank')
    ax.set_ylabel('Cumulative Energy')
    ax.set_title('Cumulative Energy Ratio by Layer')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"누적 에너지 곡선 저장: {save_path}")


def analyze_rank_vs_error(model: nn.Module,
                           save_path: str = 'results/rank_vs_error.png'):
    """Rank vs 복원 오차 곡선"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear 레이어
    ax1 = axes[0]
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data.cpu()
            max_rank = min(W.shape)
            ranks = list(range(1, max_rank + 1, max(1, max_rank // 50)))
            if max_rank not in ranks:
                ranks.append(max_rank)
            errors = [get_reconstruction_error(W, r) for r in ranks]
            ax1.plot(ranks, errors, linewidth=2, label=f'{name} ({W.shape[0]}×{W.shape[1]})')

    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Relative Reconstruction Error')
    ax1.set_title('Linear Layers: Rank vs Error')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Conv 레이어
    ax2 = axes[1]
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            W = module.weight.data.cpu()
            out_c, in_c, k_h, k_w = W.shape
            W_reshaped = W.reshape(out_c * k_h * k_w, in_c)
            max_rank = min(W_reshaped.shape)
            ranks = list(range(1, max_rank + 1, max(1, max_rank // 50)))
            if max_rank not in ranks:
                ranks.append(max_rank)
            errors = [get_reconstruction_error(W_reshaped, r) for r in ranks]
            ax2.plot(ranks, errors, linewidth=2,
                     label=f'{name} ({out_c}×{in_c}×{k_h}×{k_w})')

    ax2.set_xlabel('Rank')
    ax2.set_ylabel('Relative Reconstruction Error')
    ax2.set_title('Conv Layers: Rank vs Error')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Rank vs 복원 오차 곡선 저장: {save_path}")


# =============================================================================
# 5. 메인 실행
# =============================================================================

def main():
    print("=" * 60)
    print("Part 1: Low-Rank Approximation - 개념 및 Baseline 학습")
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
    total_params = count_parameters(model)
    print(f"    파라미터 수: {total_params:,}")

    train_losses = train(model, train_loader, EPOCHS, LEARNING_RATE, DEVICE)
    baseline_accuracy = test(model, test_loader, DEVICE)

    # 모델 저장
    os.makedirs('data/trained_models', exist_ok=True)
    torch.save(model.state_dict(), 'data/trained_models/baseline_model.pth')
    print("    Baseline 모델 저장: data/trained_models/baseline_model.pth")

    # 레이어별 파라미터 분석
    print("\n[3] 레이어별 파라미터 분석 (Low-Rank 관점)")
    print("-" * 40)
    print(f"{'Layer':<25} {'Shape':<25} {'Params':<12} {'비율':<8}")
    print("-" * 70)

    for name, param in model.named_parameters():
        if 'weight' in name:
            num_params = param.numel()
            ratio = num_params / total_params * 100
            print(f"{name:<25} {str(list(param.shape)):<25} {num_params:>10,} {ratio:>6.1f}%")

    print("-" * 70)
    print(f"{'Total':<25} {'':<25} {total_params:>10,} {'100.0%':>8}")
    print(f"\n    → classifier.0.weight (2048×512)이 전체의 88.4%를 차지")
    print(f"    → Low-Rank 분해의 핵심 타겟!")

    # 특이값 분석
    print("\n[4] 특이값(Singular Value) 분석")
    print("-" * 40)
    model_cpu = model.cpu()
    analyze_singular_values(model_cpu)
    analyze_cumulative_energy(model_cpu)
    analyze_rank_vs_error(model_cpu)

    # 에너지 기반 rank 분석
    print("\n[5] 에너지 기반 Rank 분석")
    print("-" * 40)
    print(f"{'Layer':<25} {'Shape':<20} {'95% Rank':<10} {'99% Rank':<10} {'Max Rank':<10}")
    print("-" * 75)

    for name, module in model_cpu.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data
            _, S, _ = torch.linalg.svd(W, full_matrices=False)
            energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
            rank_95 = (energy < 0.95).sum().item() + 1
            rank_99 = (energy < 0.99).sum().item() + 1
            max_rank = len(S)
            print(f"{name + '.weight':<25} {str(list(W.shape)):<20} {rank_95:<10} {rank_99:<10} {max_rank:<10}")
        elif isinstance(module, nn.Conv2d):
            W = module.weight.data
            out_c, in_c, k_h, k_w = W.shape
            W_reshaped = W.reshape(out_c * k_h * k_w, in_c)
            _, S, _ = torch.linalg.svd(W_reshaped, full_matrices=False)
            energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
            rank_95 = (energy < 0.95).sum().item() + 1
            rank_99 = (energy < 0.99).sum().item() + 1
            max_rank = len(S)
            print(f"{name + '.weight':<25} {str(list(W.shape)):<20} {rank_95:<10} {rank_99:<10} {max_rank:<10}")

    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"    모델: CNN ({total_params:,} parameters)")
    print(f"    Baseline 정확도: {baseline_accuracy:.2f}%")
    print(f"\n    핵심 관찰:")
    print(f"    - classifier.0 (Linear 2048→512)이 파라미터의 88.4%")
    print(f"    - SVD로 이 레이어를 분해하면 큰 압축 효과")
    print(f"    - 특이값 감소 패턴으로 적절한 rank를 선택 가능")
    print(f"\n    다음 단계:")
    print(f"    - Part 2: Linear 레이어 SVD 분해")
    print(f"    - Part 3: Conv 레이어 채널 분해")
    print(f"    - Part 4: 종합 실험 분석")

    return {
        'baseline_accuracy': baseline_accuracy,
        'train_losses': train_losses,
        'total_params': total_params
    }


if __name__ == "__main__":
    results = main()
