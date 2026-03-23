"""
Part 2: Linear 레이어 SVD 분해

이 스크립트에서 다루는 내용:
1. SVD 분해 원리와 Linear 레이어 적용
2. classifier.0 (2048→512)에 다양한 rank 적용
3. SVD 초기화 vs 랜덤 초기화 비교
4. Fine-tuning 후 정확도 회복
5. Rank 선택 전략: 고정 비율 vs 에너지 기반

SVD 분해 상세:
===============
Linear(in_features, out_features):
  W ∈ ℝ^(out × in), 파라미터: out × in

SVD 분해 후:
  W ≈ (U_r @ Σ_r) @ V_r^T
  → Linear(in, rank) + Linear(rank, out)
  → 파라미터: rank × (in + out)

classifier.0 예시 (2048 → 512):
  원본:   2048 × 512 = 1,048,576
  rank=256: 256 × (2048 + 512) = 655,360  (62.5%)
  rank=128: 128 × (2048 + 512) = 327,680  (31.3%)
  rank=64:   64 × (2048 + 512) = 163,840  (15.6%)
  rank=32:   32 × (2048 + 512) =  81,920  ( 7.8%)
"""

import sys
import os
import copy
import json

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
    CNN, SVDLinear,
    select_rank_by_ratio, select_rank_by_energy,
    decompose_model_linear, get_reconstruction_error
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

# Low-Rank 전용
FINETUNE_EPOCHS = 5
FINETUNE_LR = 0.0001
LINEAR_RANKS = [256, 128, 64, 32]
ENERGY_THRESHOLD = 0.95

# 재현성
SEED = 42
torch.manual_seed(SEED)


# =============================================================================
# 2. 데이터 로드
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
        print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return epoch_losses


def fine_tune(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str
) -> list[float]:
    """분해 후 Fine-tuning (낮은 학습률)"""
    return train(model, train_loader, epochs, learning_rate, device)


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
    return accuracy


def count_parameters(model: nn.Module) -> int:
    """모델의 총 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# 4. SVD 분해 실험
# =============================================================================

def experiment_svd_ranks(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    ranks: list[int],
    device: str
) -> list[dict]:
    """
    다양한 rank로 classifier.0을 SVD 분해하고 fine-tuning 후 평가

    각 rank에 대해:
    1. 원본 모델 복사
    2. classifier.0을 SVDLinear로 교체
    3. 분해 직후 정확도 측정 (without fine-tuning)
    4. Fine-tuning 수행
    5. Fine-tuning 후 정확도 측정

    Returns:
        각 rank별 결과 리스트
    """
    baseline_params = count_parameters(model)
    baseline_acc = test(model, test_loader, device)
    print(f"\n  Baseline: {baseline_acc:.2f}% ({baseline_params:,} params)")

    results = []

    for rank in ranks:
        print(f"\n  --- Rank = {rank} ---")

        # SVD 분해
        decomposed_model = decompose_model_linear(
            model, {'classifier.0': rank}
        )
        decomposed_params = count_parameters(decomposed_model)
        compression_ratio = baseline_params / decomposed_params

        # 분해 직후 정확도
        acc_before_ft = test(decomposed_model, test_loader, device)
        print(f"  분해 직후: {acc_before_ft:.2f}%")
        print(f"  파라미터: {decomposed_params:,} ({decomposed_params/baseline_params*100:.1f}%)")
        print(f"  압축률: {compression_ratio:.2f}x")

        # Fine-tuning
        print(f"  Fine-tuning ({FINETUNE_EPOCHS} epochs, lr={FINETUNE_LR})...")
        ft_losses = fine_tune(
            decomposed_model, train_loader,
            FINETUNE_EPOCHS, FINETUNE_LR, device
        )

        # Fine-tuning 후 정확도
        acc_after_ft = test(decomposed_model, test_loader, device)
        print(f"  Fine-tuning 후: {acc_after_ft:.2f}%")
        print(f"  정확도 회복: {acc_after_ft - acc_before_ft:+.2f}%")

        # 복원 오차
        original_weight = model.classifier[0].weight.data.cpu()
        recon_error = get_reconstruction_error(original_weight, rank)

        results.append({
            'rank': rank,
            'params': decomposed_params,
            'param_ratio': decomposed_params / baseline_params,
            'compression_ratio': compression_ratio,
            'acc_before_ft': acc_before_ft,
            'acc_after_ft': acc_after_ft,
            'acc_recovery': acc_after_ft - acc_before_ft,
            'reconstruction_error': recon_error,
            'ft_losses': ft_losses
        })

    return results


def experiment_init_comparison(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    rank: int,
    device: str
) -> dict:
    """
    SVD 초기화 vs 랜덤 초기화 비교

    SVD 초기화: 원본 가중치의 SVD 분해로 초기화 (정보 보존)
    랜덤 초기화: 기본 nn.Linear 초기화 (정보 손실)

    Returns:
        두 초기화 전략의 비교 결과
    """
    print(f"\n  [SVD 초기화 vs 랜덤 초기화] rank={rank}")

    # 1) SVD 초기화
    print("\n  (a) SVD 초기화:")
    svd_model = decompose_model_linear(model, {'classifier.0': rank})
    svd_acc_before = test(svd_model, test_loader, device)
    print(f"  초기 정확도: {svd_acc_before:.2f}%")

    svd_losses = fine_tune(
        svd_model, train_loader, FINETUNE_EPOCHS, FINETUNE_LR, device
    )
    svd_acc_after = test(svd_model, test_loader, device)
    print(f"  Fine-tuning 후: {svd_acc_after:.2f}%")

    # 2) 랜덤 초기화
    print("\n  (b) 랜덤 초기화:")
    random_model = copy.deepcopy(model)

    # classifier.0을 SVDLinear로 교체하되, 랜덤 가중치 사용
    original_layer = random_model.classifier[0]
    has_bias = original_layer.bias is not None
    random_decomposed = SVDLinear(
        original_layer.in_features, original_layer.out_features,
        rank, bias=has_bias
    )
    # 기본 nn.Linear 초기화 유지 (랜덤)
    random_model.classifier[0] = random_decomposed

    random_acc_before = test(random_model, test_loader, device)
    print(f"  초기 정확도: {random_acc_before:.2f}%")

    random_losses = fine_tune(
        random_model, train_loader, FINETUNE_EPOCHS, FINETUNE_LR, device
    )
    random_acc_after = test(random_model, test_loader, device)
    print(f"  Fine-tuning 후: {random_acc_after:.2f}%")

    return {
        'rank': rank,
        'svd_acc_before': svd_acc_before,
        'svd_acc_after': svd_acc_after,
        'svd_losses': svd_losses,
        'random_acc_before': random_acc_before,
        'random_acc_after': random_acc_after,
        'random_losses': random_losses
    }


def experiment_rank_selection(model: nn.Module) -> dict:
    """
    Rank 선택 전략 비교: 고정 비율 vs 에너지 기반

    Returns:
        각 전략별 선택된 rank 및 분석 결과
    """
    W = model.classifier[0].weight.data.cpu()

    results = {
        'weight_shape': list(W.shape),
        'original_params': W.numel(),
        'strategies': {}
    }

    # 고정 비율 전략
    print("\n  (a) 고정 비율 전략:")
    print(f"  {'Ratio':<10} {'Rank':<10} {'Params':<12} {'압축률':<10} {'Error':<10}")
    print("  " + "-" * 52)

    for ratio in [0.75, 0.5, 0.25, 0.1]:
        rank = select_rank_by_ratio(W, ratio)
        params = rank * (W.shape[0] + W.shape[1])
        compression = W.numel() / params
        error = get_reconstruction_error(W, rank)
        print(f"  {ratio:<10.2f} {rank:<10} {params:<12,} {compression:<10.2f}x {error:<10.4f}")

        results['strategies'][f'ratio_{ratio}'] = {
            'rank': rank, 'params': params,
            'compression': compression, 'error': error
        }

    # 에너지 기반 전략
    print("\n  (b) 에너지 기반 전략:")
    print(f"  {'Threshold':<12} {'Rank':<10} {'Params':<12} {'압축률':<10} {'Error':<10}")
    print("  " + "-" * 54)

    for threshold in [0.90, 0.95, 0.99, 0.999]:
        rank = select_rank_by_energy(W, threshold)
        params = rank * (W.shape[0] + W.shape[1])
        compression = W.numel() / params
        error = get_reconstruction_error(W, rank)
        print(f"  {threshold:<12.3f} {rank:<10} {params:<12,} {compression:<10.2f}x {error:<10.4f}")

        results['strategies'][f'energy_{threshold}'] = {
            'rank': rank, 'params': params,
            'compression': compression, 'error': error
        }

    return results


# =============================================================================
# 5. 시각화
# =============================================================================

def plot_rank_tradeoff(
    results: list[dict],
    baseline_acc: float,
    save_path: str = 'results/linear_svd_tradeoff.png'
):
    """Rank vs 정확도/파라미터 trade-off 곡선"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ranks = [r['rank'] for r in results]
    accs_before = [r['acc_before_ft'] for r in results]
    accs_after = [r['acc_after_ft'] for r in results]
    param_ratios = [r['param_ratio'] * 100 for r in results]
    recon_errors = [r['reconstruction_error'] for r in results]

    # (1) Rank vs 정확도
    ax1 = axes[0]
    ax1.plot(ranks, accs_before, 'ro--', linewidth=2, markersize=8,
             label='Before Fine-tuning')
    ax1.plot(ranks, accs_after, 'bs-', linewidth=2, markersize=8,
             label='After Fine-tuning')
    ax1.axhline(y=baseline_acc, color='green', linestyle='--',
                alpha=0.7, label=f'Baseline ({baseline_acc:.1f}%)')
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Rank vs Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) 파라미터 비율 vs 정확도
    ax2 = axes[1]
    ax2.plot(param_ratios, accs_after, 'bs-', linewidth=2, markersize=8)
    for i, rank in enumerate(ranks):
        ax2.annotate(f'r={rank}', (param_ratios[i], accs_after[i]),
                     textcoords="offset points", xytext=(10, 5), fontsize=9)
    ax2.axhline(y=baseline_acc, color='green', linestyle='--',
                alpha=0.7, label=f'Baseline ({baseline_acc:.1f}%)')
    ax2.set_xlabel('Parameter Ratio (%)')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Parameters vs Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # (3) Rank vs 복원 오차
    ax3 = axes[2]
    ax3.plot(ranks, recon_errors, 'g^-', linewidth=2, markersize=8)
    ax3.set_xlabel('Rank')
    ax3.set_ylabel('Reconstruction Error')
    ax3.set_title('Rank vs Reconstruction Error')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Trade-off 곡선 저장: {save_path}")


def plot_init_comparison(
    init_results: dict,
    save_path: str = 'results/init_comparison.png'
):
    """SVD 초기화 vs 랜덤 초기화 비교 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    rank = init_results['rank']

    # (1) Fine-tuning loss 곡선
    ax1 = axes[0]
    epochs = range(1, FINETUNE_EPOCHS + 1)
    ax1.plot(epochs, init_results['svd_losses'], 'b-', linewidth=2,
             marker='o', label='SVD Init')
    ax1.plot(epochs, init_results['random_losses'], 'r-', linewidth=2,
             marker='s', label='Random Init')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title(f'Fine-tuning Loss (rank={rank})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) 정확도 비교 (before/after)
    ax2 = axes[1]
    x = np.arange(2)
    width = 0.3

    svd_accs = [init_results['svd_acc_before'], init_results['svd_acc_after']]
    random_accs = [init_results['random_acc_before'], init_results['random_acc_after']]

    bars1 = ax2.bar(x - width/2, svd_accs, width, label='SVD Init', color='steelblue')
    bars2 = ax2.bar(x + width/2, random_accs, width, label='Random Init', color='indianred')

    ax2.set_xticks(x)
    ax2.set_xticklabels(['Before FT', 'After FT'])
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title(f'Init Strategy Comparison (rank={rank})')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  초기화 비교 시각화 저장: {save_path}")


def plot_finetuning_curves(
    results: list[dict],
    save_path: str = 'results/linear_svd_finetuning.png'
):
    """각 rank별 Fine-tuning loss 곡선"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for r in results:
        epochs = range(1, len(r['ft_losses']) + 1)
        ax.plot(epochs, r['ft_losses'], linewidth=2, marker='o',
                label=f"rank={r['rank']} ({r['param_ratio']*100:.1f}%)")

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Fine-tuning Loss by Rank')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fine-tuning 곡선 저장: {save_path}")


# =============================================================================
# 6. 메인 실행
# =============================================================================

def main():
    print("=" * 60)
    print("Part 2: Linear 레이어 SVD 분해")
    print("=" * 60)

    # 데이터 로드
    print("\n[1] 데이터 로드")
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)

    # Baseline 모델 로드 또는 학습
    print("\n[2] Baseline 모델 준비")
    print("-" * 40)
    model = CNN(num_classes=NUM_CLASSES)
    model_path = 'data/trained_models/baseline_model.pth'

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(DEVICE)
        print(f"  저장된 모델 로드: {model_path}")
    else:
        print("  Baseline 모델 학습 중...")
        torch.manual_seed(SEED)
        train(model, train_loader, EPOCHS, LEARNING_RATE, DEVICE)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)

    baseline_acc = test(model, test_loader, DEVICE)
    baseline_params = count_parameters(model)
    print(f"  Baseline 정확도: {baseline_acc:.2f}%")
    print(f"  Baseline 파라미터: {baseline_params:,}")

    # SVD 분해 실험 (다양한 rank)
    print("\n[3] SVD 분해 실험 (classifier.0)")
    print("-" * 40)
    print(f"  대상: classifier.0 (Linear 2048→512)")
    print(f"  실험 ranks: {LINEAR_RANKS}")

    svd_results = experiment_svd_ranks(
        model, train_loader, test_loader, LINEAR_RANKS, DEVICE
    )

    # 결과 요약 테이블
    print(f"\n  {'Rank':<8} {'Before FT':<12} {'After FT':<12} {'Params':<12} {'압축률':<10} {'Error':<10}")
    print("  " + "-" * 64)
    for r in svd_results:
        print(f"  {r['rank']:<8} {r['acc_before_ft']:<12.2f} {r['acc_after_ft']:<12.2f} "
              f"{r['params']:<12,} {r['compression_ratio']:<10.2f}x {r['reconstruction_error']:<10.4f}")

    # 시각화
    plot_rank_tradeoff(svd_results, baseline_acc)
    plot_finetuning_curves(svd_results)

    # SVD vs 랜덤 초기화 비교
    print("\n[4] 초기화 전략 비교")
    print("-" * 40)

    init_results = experiment_init_comparison(
        model, train_loader, test_loader, rank=128, device=DEVICE
    )
    plot_init_comparison(init_results)

    print(f"\n  SVD 초기화 우위: {init_results['svd_acc_after'] - init_results['random_acc_after']:+.2f}%")

    # Rank 선택 전략
    print("\n[5] Rank 선택 전략 비교")
    print("-" * 40)
    model_cpu = model.cpu()
    rank_selection = experiment_rank_selection(model_cpu)

    # 결과 저장
    print("\n[6] 결과 저장")
    print("-" * 40)
    os.makedirs('results', exist_ok=True)

    save_data = {
        'baseline_accuracy': baseline_acc,
        'baseline_params': baseline_params,
        'svd_results': [
            {k: v for k, v in r.items() if k != 'ft_losses'}
            for r in svd_results
        ],
        'init_comparison': {
            k: v for k, v in init_results.items()
            if k not in ('svd_losses', 'random_losses')
        },
        'rank_selection': rank_selection
    }

    with open('results/linear_svd_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("  결과 저장: results/linear_svd_results.json")

    # 최종 요약
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"  Baseline: {baseline_acc:.2f}% ({baseline_params:,} params)")
    print()

    best = max(svd_results, key=lambda x: x['acc_after_ft'])
    most_compressed = min(svd_results, key=lambda x: x['params'])

    print(f"  최고 정확도: rank={best['rank']}, {best['acc_after_ft']:.2f}% "
          f"({best['compression_ratio']:.2f}x 압축)")
    print(f"  최대 압축: rank={most_compressed['rank']}, "
          f"{most_compressed['acc_after_ft']:.2f}% "
          f"({most_compressed['compression_ratio']:.2f}x 압축)")
    print()
    print(f"  초기화 비교 (rank=128):")
    print(f"    SVD 초기화: {init_results['svd_acc_after']:.2f}%")
    print(f"    랜덤 초기화: {init_results['random_acc_after']:.2f}%")
    print(f"    → SVD 초기화가 {init_results['svd_acc_after'] - init_results['random_acc_after']:+.2f}% 우수")
    print()
    print(f"  핵심 관찰:")
    print(f"    - SVD 분해는 원본 정보를 최대한 보존 (Eckart-Young 정리)")
    print(f"    - Fine-tuning으로 분해 후 정확도 회복 가능")
    print(f"    - SVD 초기화가 랜덤 초기화보다 훨씬 우수")
    print(f"\n  다음 단계: Part 3에서 Conv 레이어 분해")

    return {
        'baseline_accuracy': baseline_acc,
        'baseline_params': baseline_params,
        'svd_results': svd_results,
        'init_comparison': init_results,
        'rank_selection': rank_selection
    }


if __name__ == "__main__":
    results = main()
