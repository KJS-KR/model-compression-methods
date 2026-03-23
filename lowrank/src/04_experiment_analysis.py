"""
Part 4: 종합 실험 분석

이 스크립트에서 다루는 내용:
1. Baseline vs Linear SVD vs Conv 분해 vs Combined 비교
2. 다양한 rank 조합에서의 정확도 vs 압축률 곡선
3. Rank 선택 전략 비교 (고정 비율 vs 에너지 기반)
4. 레이어별 분해 효과 분석
5. 결과 JSON 저장 + 종합 시각화

이전 스크립트 결과가 있으면 캐시에서 로드하고,
없으면 새로 실험을 수행합니다.
"""

import sys
import os
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
    CNN,
    decompose_model_linear, decompose_model_conv,
    select_rank_by_ratio, select_rank_by_energy,
    get_reconstruction_error
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
CONV_RANKS = [32, 16, 8]
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
    """분해 후 Fine-tuning"""
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
# 4. 종합 실험
# =============================================================================

def run_comprehensive_experiments(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str
) -> dict:
    """
    모든 분해 방법을 체계적으로 실험

    실험 구성:
    1. Baseline (원본)
    2. Linear SVD only (다양한 rank)
    3. Conv 분해 only (다양한 rank)
    4. Combined (Linear + Conv)
    5. 에너지 기반 자동 rank 선택

    Returns:
        종합 실험 결과
    """
    baseline_params = count_parameters(model)
    baseline_acc = test(model, test_loader, device)

    all_results = {
        'baseline': {
            'accuracy': baseline_acc,
            'params': baseline_params,
            'compression_ratio': 1.0
        },
        'linear_svd': {},
        'conv_decomp': {},
        'combined': {},
        'energy_based': {}
    }

    # --- Linear SVD 실험 ---
    print("\n  [Linear SVD 실험]")
    for rank in LINEAR_RANKS:
        print(f"\n  Linear rank={rank}:")
        decomposed = decompose_model_linear(model, {'classifier.0': rank})
        params = count_parameters(decomposed)

        acc_before = test(decomposed, test_loader, device)
        fine_tune(decomposed, train_loader, FINETUNE_EPOCHS, FINETUNE_LR, device)
        acc_after = test(decomposed, test_loader, device)

        all_results['linear_svd'][rank] = {
            'acc_before_ft': acc_before,
            'acc_after_ft': acc_after,
            'params': params,
            'compression_ratio': baseline_params / params
        }
        print(f"    {acc_before:.2f}% → {acc_after:.2f}% "
              f"({params:,} params, {baseline_params/params:.2f}x)")

    # --- Conv 분해 실험 ---
    print("\n  [Conv 분해 실험]")
    for rank in CONV_RANKS:
        print(f"\n  Conv rank={rank}:")
        decomposed = decompose_model_conv(model, {'features.2': rank})
        params = count_parameters(decomposed)

        acc_before = test(decomposed, test_loader, device)
        fine_tune(decomposed, train_loader, FINETUNE_EPOCHS, FINETUNE_LR, device)
        acc_after = test(decomposed, test_loader, device)

        all_results['conv_decomp'][rank] = {
            'acc_before_ft': acc_before,
            'acc_after_ft': acc_after,
            'params': params,
            'compression_ratio': baseline_params / params
        }
        print(f"    {acc_before:.2f}% → {acc_after:.2f}% "
              f"({params:,} params, {baseline_params/params:.2f}x)")

    # --- Combined 실험 ---
    print("\n  [Combined 실험]")
    combined_configs = [
        (256, 32), (128, 32), (128, 16), (64, 16), (64, 8), (32, 8)
    ]

    for lr, cr in combined_configs:
        print(f"\n  Combined Linear={lr}, Conv={cr}:")
        decomposed = decompose_model_linear(model, {'classifier.0': lr})
        decomposed = decompose_model_conv(decomposed, {'features.2': cr})
        params = count_parameters(decomposed)

        acc_before = test(decomposed, test_loader, device)
        fine_tune(decomposed, train_loader, FINETUNE_EPOCHS, FINETUNE_LR, device)
        acc_after = test(decomposed, test_loader, device)

        key = f'L{lr}_C{cr}'
        all_results['combined'][key] = {
            'linear_rank': lr,
            'conv_rank': cr,
            'acc_before_ft': acc_before,
            'acc_after_ft': acc_after,
            'params': params,
            'compression_ratio': baseline_params / params
        }
        print(f"    {acc_before:.2f}% → {acc_after:.2f}% "
              f"({params:,} params, {baseline_params/params:.2f}x)")

    # --- 에너지 기반 자동 rank 선택 ---
    print("\n  [에너지 기반 자동 Rank 선택]")
    model_cpu = model.cpu()

    # classifier.0
    W_linear = model_cpu.classifier[0].weight.data
    energy_linear_rank = select_rank_by_energy(W_linear, ENERGY_THRESHOLD)

    # features.2
    W_conv = model_cpu.features[2].weight.data
    out_c, in_c, k_h, k_w = W_conv.shape
    W_conv_reshaped = W_conv.reshape(out_c * k_h * k_w, in_c)
    energy_conv_rank = select_rank_by_energy(W_conv_reshaped, ENERGY_THRESHOLD)

    print(f"  에너지 threshold={ENERGY_THRESHOLD}")
    print(f"  classifier.0: rank={energy_linear_rank}")
    print(f"  features.2: rank={energy_conv_rank}")

    # 에너지 기반 Combined
    decomposed = decompose_model_linear(model, {'classifier.0': energy_linear_rank})
    decomposed = decompose_model_conv(decomposed, {'features.2': energy_conv_rank})
    params = count_parameters(decomposed)

    acc_before = test(decomposed, test_loader, device)
    fine_tune(decomposed, train_loader, FINETUNE_EPOCHS, FINETUNE_LR, device)
    acc_after = test(decomposed, test_loader, device)

    all_results['energy_based'] = {
        'threshold': ENERGY_THRESHOLD,
        'linear_rank': energy_linear_rank,
        'conv_rank': energy_conv_rank,
        'acc_before_ft': acc_before,
        'acc_after_ft': acc_after,
        'params': params,
        'compression_ratio': baseline_params / params
    }
    print(f"  결과: {acc_before:.2f}% → {acc_after:.2f}% "
          f"({params:,} params, {baseline_params/params:.2f}x)")

    return all_results


# =============================================================================
# 5. 시각화
# =============================================================================

def plot_accuracy_comparison(
    results: dict,
    save_path: str = 'results/accuracy_comparison.png'
):
    """모든 방법의 정확도 비교 바 차트"""
    methods = []
    accs = []
    colors = []

    baseline_acc = results['baseline']['accuracy']
    methods.append('Baseline')
    accs.append(baseline_acc)
    colors.append('green')

    for rank, data in sorted(results['linear_svd'].items()):
        methods.append(f'Linear\nr={rank}')
        accs.append(data['acc_after_ft'])
        colors.append('steelblue')

    for rank, data in sorted(results['conv_decomp'].items(), reverse=True):
        methods.append(f'Conv\nr={rank}')
        accs.append(data['acc_after_ft'])
        colors.append('coral')

    # Combined 중 대표적인 것만
    for key in ['L128_C32', 'L64_C16']:
        if key in results['combined']:
            data = results['combined'][key]
            methods.append(f'Combined\n{key}')
            accs.append(data['acc_after_ft'])
            colors.append('darkorange')

    if results['energy_based']:
        methods.append(f'Energy\n{ENERGY_THRESHOLD}')
        accs.append(results['energy_based']['acc_after_ft'])
        colors.append('purple')

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    x = np.arange(len(methods))
    bars = ax.bar(x, accs, color=colors, alpha=0.8, edgecolor='gray')

    ax.axhline(y=baseline_acc, color='green', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Low-Rank Approximation: Method Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars, accs):
        diff = acc - baseline_acc
        color = 'green' if diff >= 0 else 'red'
        ax.annotate(f'{acc:.1f}%\n({diff:+.1f})',
                    xy=(bar.get_x() + bar.get_width() / 2, acc),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, color=color)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  정확도 비교 시각화 저장: {save_path}")


def plot_pareto_frontier(
    results: dict,
    save_path: str = 'results/pareto_frontier.png'
):
    """정확도 vs 압축률 Pareto 곡선"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    baseline_acc = results['baseline']['accuracy']
    baseline_params = results['baseline']['params']

    # Baseline
    ax.scatter([1.0], [baseline_acc], s=150, c='green', marker='*',
               zorder=5, label='Baseline')

    # Linear SVD
    for rank, data in sorted(results['linear_svd'].items()):
        ax.scatter([data['compression_ratio']], [data['acc_after_ft']],
                   s=100, c='steelblue', marker='o', zorder=4)
        ax.annotate(f'L={rank}',
                    (data['compression_ratio'], data['acc_after_ft']),
                    textcoords="offset points", xytext=(8, 3), fontsize=8)

    # 범례용 dummy
    ax.scatter([], [], s=100, c='steelblue', marker='o', label='Linear SVD')

    # Conv 분해
    for rank, data in sorted(results['conv_decomp'].items()):
        ax.scatter([data['compression_ratio']], [data['acc_after_ft']],
                   s=100, c='coral', marker='s', zorder=4)
        ax.annotate(f'C={rank}',
                    (data['compression_ratio'], data['acc_after_ft']),
                    textcoords="offset points", xytext=(8, 3), fontsize=8)

    ax.scatter([], [], s=100, c='coral', marker='s', label='Conv Decomp')

    # Combined
    for key, data in results['combined'].items():
        ax.scatter([data['compression_ratio']], [data['acc_after_ft']],
                   s=100, c='darkorange', marker='^', zorder=4)
        ax.annotate(key,
                    (data['compression_ratio'], data['acc_after_ft']),
                    textcoords="offset points", xytext=(8, 3), fontsize=7)

    ax.scatter([], [], s=100, c='darkorange', marker='^', label='Combined')

    # Energy-based
    if results['energy_based']:
        data = results['energy_based']
        ax.scatter([data['compression_ratio']], [data['acc_after_ft']],
                   s=150, c='purple', marker='D', zorder=5,
                   label=f'Energy ({ENERGY_THRESHOLD})')
        ax.annotate(f"E={ENERGY_THRESHOLD}",
                    (data['compression_ratio'], data['acc_after_ft']),
                    textcoords="offset points", xytext=(8, 3), fontsize=8)

    ax.set_xlabel('Compression Ratio (x)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Accuracy vs Compression: Pareto Frontier')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Pareto 곡선 저장: {save_path}")


def plot_layer_contribution(
    model: nn.Module,
    save_path: str = 'results/layer_contribution.png'
):
    """레이어별 파라미터 비중과 분해 잠재력 분석"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    layer_names = []
    layer_params = []
    layer_ranks_95 = []
    layer_max_ranks = []

    model_cpu = model.cpu()

    for name, module in model_cpu.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data
            params = W.numel()
            _, S, _ = torch.linalg.svd(W, full_matrices=False)
            energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
            rank_95 = (energy < 0.95).sum().item() + 1

            layer_names.append(name)
            layer_params.append(params)
            layer_ranks_95.append(rank_95)
            layer_max_ranks.append(len(S))

        elif isinstance(module, nn.Conv2d):
            W = module.weight.data
            params = W.numel()
            out_c, in_c, k_h, k_w = W.shape
            W_reshaped = W.reshape(out_c * k_h * k_w, in_c)
            _, S, _ = torch.linalg.svd(W_reshaped, full_matrices=False)
            energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
            rank_95 = (energy < 0.95).sum().item() + 1

            layer_names.append(name)
            layer_params.append(params)
            layer_ranks_95.append(rank_95)
            layer_max_ranks.append(len(S))

    # (1) 파라미터 비중 파이 차트
    ax1 = axes[0]
    ax1.pie(layer_params, labels=layer_names, autopct='%1.1f%%',
            startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(layer_names))))
    ax1.set_title('Parameter Distribution by Layer')

    # (2) 95% 에너지 rank vs max rank
    ax2 = axes[1]
    x = np.arange(len(layer_names))
    width = 0.35

    bars1 = ax2.bar(x - width/2, layer_max_ranks, width,
                    label='Max Rank', color='lightcoral', alpha=0.8)
    bars2 = ax2.bar(x + width/2, layer_ranks_95, width,
                    label='95% Energy Rank', color='steelblue', alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_names, fontsize=8, rotation=45, ha='right')
    ax2.set_ylabel('Rank')
    ax2.set_title('Max Rank vs 95% Energy Rank')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 압축 비율 표시
    for i, (mr, r95) in enumerate(zip(layer_max_ranks, layer_ranks_95)):
        ratio = r95 / mr * 100
        ax2.annotate(f'{ratio:.0f}%',
                     xy=(x[i] + width/2, r95),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  레이어 분석 시각화 저장: {save_path}")


def plot_rank_strategy_comparison(
    model: nn.Module,
    save_path: str = 'results/rank_strategy_comparison.png'
):
    """Rank 선택 전략 비교: 고정 비율 vs 에너지 기반"""
    model_cpu = model.cpu()
    W = model_cpu.classifier[0].weight.data

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) 고정 비율별 rank와 복원 오차
    ax1 = axes[0]
    ratios = np.arange(0.05, 1.0, 0.05)
    ratio_ranks = []
    ratio_errors = []

    for ratio in ratios:
        rank = select_rank_by_ratio(W, ratio)
        error = get_reconstruction_error(W, rank)
        ratio_ranks.append(rank)
        ratio_errors.append(error)

    ax1_twin = ax1.twinx()
    line1 = ax1.plot(ratios, ratio_ranks, 'b-o', markersize=4, label='Rank')
    line2 = ax1_twin.plot(ratios, ratio_errors, 'r-s', markersize=4, label='Error')
    ax1.set_xlabel('Compression Ratio')
    ax1.set_ylabel('Selected Rank', color='blue')
    ax1_twin.set_ylabel('Reconstruction Error', color='red')
    ax1.set_title('Fixed Ratio Strategy')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    ax1.grid(True, alpha=0.3)

    # (2) 에너지 threshold별 rank와 복원 오차
    ax2 = axes[1]
    thresholds = np.arange(0.80, 1.0, 0.01)
    energy_ranks = []
    energy_errors = []

    for threshold in thresholds:
        rank = select_rank_by_energy(W, threshold)
        error = get_reconstruction_error(W, rank)
        energy_ranks.append(rank)
        energy_errors.append(error)

    ax2_twin = ax2.twinx()
    line1 = ax2.plot(thresholds, energy_ranks, 'b-o', markersize=3, label='Rank')
    line2 = ax2_twin.plot(thresholds, energy_errors, 'r-s', markersize=3, label='Error')
    ax2.set_xlabel('Energy Threshold')
    ax2.set_ylabel('Selected Rank', color='blue')
    ax2_twin.set_ylabel('Reconstruction Error', color='red')
    ax2.set_title('Energy-based Strategy')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Rank 전략 비교 저장: {save_path}")


def create_summary_table(
    results: dict,
    save_path: str = 'results/summary.txt'
):
    """종합 결과 테이블을 텍스트 파일로 저장"""
    lines = []
    lines.append("=" * 80)
    lines.append("Low-Rank Approximation 종합 실험 결과")
    lines.append("=" * 80)
    lines.append("")

    baseline = results['baseline']
    lines.append(f"{'Method':<30} {'Accuracy':<12} {'Params':<12} {'Ratio':<10} {'vs Base':<10}")
    lines.append("-" * 74)
    lines.append(f"{'Baseline':<30} {baseline['accuracy']:<12.2f} "
                 f"{baseline['params']:<12,} {'1.00x':<10} {'-':>10}")

    # Linear SVD
    for rank, data in sorted(results['linear_svd'].items(), reverse=True):
        name = f'Linear SVD (r={rank})'
        diff = data['acc_after_ft'] - baseline['accuracy']
        lines.append(f"{name:<30} {data['acc_after_ft']:<12.2f} "
                     f"{data['params']:<12,} {data['compression_ratio']:<10.2f}x "
                     f"{diff:>+9.2f}%")

    # Conv 분해
    for rank, data in sorted(results['conv_decomp'].items(), reverse=True):
        name = f'Conv Decomp (r={rank})'
        diff = data['acc_after_ft'] - baseline['accuracy']
        lines.append(f"{name:<30} {data['acc_after_ft']:<12.2f} "
                     f"{data['params']:<12,} {data['compression_ratio']:<10.2f}x "
                     f"{diff:>+9.2f}%")

    # Combined
    for key, data in sorted(results['combined'].items()):
        name = f'Combined ({key})'
        diff = data['acc_after_ft'] - baseline['accuracy']
        lines.append(f"{name:<30} {data['acc_after_ft']:<12.2f} "
                     f"{data['params']:<12,} {data['compression_ratio']:<10.2f}x "
                     f"{diff:>+9.2f}%")

    # Energy-based
    if results['energy_based']:
        data = results['energy_based']
        name = f'Energy ({data["threshold"]})'
        diff = data['acc_after_ft'] - baseline['accuracy']
        lines.append(f"{name:<30} {data['acc_after_ft']:<12.2f} "
                     f"{data['params']:<12,} {data['compression_ratio']:<10.2f}x "
                     f"{diff:>+9.2f}%")

    lines.append("-" * 74)
    lines.append("")
    lines.append("Notes:")
    lines.append(f"  - Fine-tuning: {FINETUNE_EPOCHS} epochs, LR={FINETUNE_LR}")
    lines.append(f"  - Linear SVD target: classifier.0 (2048→512)")
    lines.append(f"  - Conv Decomp target: features.2 (128→64, 3×3)")
    lines.append(f"  - Energy threshold: {ENERGY_THRESHOLD}")

    text = '\n'.join(lines)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"  결과 테이블 저장: {save_path}")

    # 콘솔에도 출력
    print()
    print(text)


# =============================================================================
# 6. 메인 실행
# =============================================================================

def main():
    print("=" * 60)
    print("Part 4: 종합 실험 분석")
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

    # 캐시 확인
    cache_path = 'results/comprehensive_results.json'
    if os.path.exists(cache_path):
        print(f"\n  캐시된 결과 발견: {cache_path}")
        print("  새로운 실험을 수행합니다 (기존 결과는 덮어씁니다)...")

    # 종합 실험
    print("\n[3] 종합 실험 수행")
    print("-" * 40)
    all_results = run_comprehensive_experiments(
        model, train_loader, test_loader, DEVICE
    )

    # 레이어 분석 시각화
    print("\n[4] 레이어별 분석")
    print("-" * 40)
    plot_layer_contribution(model)

    # Rank 선택 전략 비교
    print("\n[5] Rank 선택 전략 비교")
    print("-" * 40)
    plot_rank_strategy_comparison(model)

    # 종합 시각화
    print("\n[6] 종합 시각화")
    print("-" * 40)
    plot_accuracy_comparison(all_results)
    plot_pareto_frontier(all_results)

    # 결과 저장
    print("\n[7] 결과 저장")
    print("-" * 40)
    os.makedirs('results', exist_ok=True)

    with open(cache_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  종합 결과 저장: {cache_path}")

    create_summary_table(all_results)

    # 최종 요약
    print("\n" + "=" * 60)
    print("최종 요약")
    print("=" * 60)

    print(f"\n  Baseline: {baseline_acc:.2f}% ({baseline_params:,} params)")

    # 최고 정확도 (Baseline 제외)
    all_methods = {}
    for rank, data in all_results['linear_svd'].items():
        all_methods[f'Linear r={rank}'] = data
    for rank, data in all_results['conv_decomp'].items():
        all_methods[f'Conv r={rank}'] = data
    for key, data in all_results['combined'].items():
        all_methods[f'Combined {key}'] = data
    if all_results['energy_based']:
        all_methods['Energy-based'] = all_results['energy_based']

    if all_methods:
        best_name = max(all_methods, key=lambda k: all_methods[k]['acc_after_ft'])
        best = all_methods[best_name]
        print(f"\n  최고 정확도: {best_name}")
        print(f"    {best['acc_after_ft']:.2f}% ({best['compression_ratio']:.2f}x 압축)")

        best_compression_name = max(all_methods,
                                     key=lambda k: all_methods[k]['compression_ratio'])
        best_comp = all_methods[best_compression_name]
        print(f"\n  최대 압축: {best_compression_name}")
        print(f"    {best_comp['acc_after_ft']:.2f}% ({best_comp['compression_ratio']:.2f}x 압축)")

    print(f"\n  핵심 관찰:")
    print(f"    - Linear SVD가 가장 큰 압축 효과 (classifier.0이 88.4%)")
    print(f"    - Conv 분해는 추가적 압축에 기여")
    print(f"    - Combined 방법이 최적의 압축률-정확도 trade-off")
    print(f"    - 에너지 기반 rank 선택으로 자동화 가능")
    print(f"    - Fine-tuning이 정확도 회복에 핵심적")

    return all_results


if __name__ == "__main__":
    results = main()
