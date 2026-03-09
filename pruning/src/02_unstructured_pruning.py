"""
Part 2: Unstructured Pruning (비정형 프루닝) 구현

이 스크립트에서 다루는 내용:
1. Unstructured Pruning의 핵심 개념
2. PyTorch의 torch.nn.utils.prune 모듈 사용법
3. L1 Unstructured Pruning (Magnitude-based)
4. Random Unstructured Pruning
5. Global Unstructured Pruning
6. Sparsity vs Accuracy 분석

Unstructured Pruning (비정형 프루닝):
=====================================

개별 가중치(weight) 단위로 pruning을 수행합니다.
제거된 가중치는 0으로 설정되며, mask를 통해 관리됩니다.

방법 1: L1 Unstructured Pruning
-------------------------------
- 각 레이어에서 절대값이 가장 작은 가중치를 제거
- 직관: |w|가 작으면 출력에 미치는 영향이 작다
- 가장 널리 사용되는 pruning 방법

방법 2: Random Unstructured Pruning
-----------------------------------
- 각 레이어에서 무작위로 가중치를 제거
- L1 pruning과의 비교를 위한 baseline
- Magnitude-based의 우수성을 검증

방법 3: Global Unstructured Pruning
-----------------------------------
- 모든 레이어를 통합해서 pruning (레이어 단위가 아님)
- 전체 모델에서 가장 중요하지 않은 가중치를 제거
- 레이어마다 다른 pruning 비율이 자동 적용

PyTorch Pruning 메커니즘:
========================
1. prune.l1_unstructured(module, 'weight', amount=0.3)
2. 내부적으로 weight_mask 생성 (0: pruned, 1: kept)
3. forward 시 weight = weight_orig * weight_mask
4. prune.remove()로 mask를 영구 적용
"""

import sys
import os
import copy

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
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

# Pruning 하이퍼파라미터
PRUNING_AMOUNT = 0.3  # 30% 가중치 제거

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
# 3. Pruning 유틸리티 함수
# =============================================================================

def get_sparsity(model: nn.Module) -> dict:
    """
    모델의 sparsity(희소성) 분석

    Sparsity = 0인 가중치 수 / 전체 가중치 수
    """
    total_params = 0
    zero_params = 0
    layer_sparsity = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            total = weight.numel()
            zeros = (weight == 0).sum().item()
            total_params += total
            zero_params += zeros

            if total > 0:
                layer_sparsity[name] = {
                    'total': total,
                    'zeros': zeros,
                    'sparsity': zeros / total * 100
                }

    overall_sparsity = zero_params / total_params * 100 if total_params > 0 else 0

    return {
        'overall_sparsity': overall_sparsity,
        'total_params': total_params,
        'zero_params': zero_params,
        'layer_sparsity': layer_sparsity
    }


def print_sparsity(model: nn.Module, title: str = ""):
    """Sparsity 정보 출력"""
    info = get_sparsity(model)

    if title:
        print(f"\n    [{title}]")
    print(f"    전체 Sparsity: {info['overall_sparsity']:.1f}%")
    print(f"    (0인 파라미터: {info['zero_params']:,} / {info['total_params']:,})")
    print(f"    {'Layer':<30} {'Sparsity':<10} {'0s / Total':<20}")
    print(f"    {'-' * 60}")

    for name, stats in info['layer_sparsity'].items():
        print(f"    {name:<30} {stats['sparsity']:>7.1f}%   {stats['zeros']:>8,} / {stats['total']:>8,}")


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
        print(f"    Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")


# =============================================================================
# 4. L1 Unstructured Pruning
# =============================================================================

def apply_l1_unstructured_pruning(model: nn.Module, amount: float) -> nn.Module:
    """
    L1 Unstructured Pruning 적용

    각 Conv2d, Linear 레이어에서 절대값이 가장 작은 가중치를 제거합니다.

    Args:
        model: pruning할 모델
        amount: 제거할 가중치 비율 (0~1)

    Returns:
        pruning된 모델
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
        elif isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)

    return model


def remove_pruning_reparametrization(model: nn.Module) -> nn.Module:
    """
    Pruning mask를 영구 적용

    prune.remove()를 호출하면:
    - weight = weight_orig * weight_mask
    - weight_orig, weight_mask 제거
    - weight만 남김 (0이 포함된 상태)
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass

    return model


# =============================================================================
# 5. Random Unstructured Pruning
# =============================================================================

def apply_random_unstructured_pruning(model: nn.Module, amount: float) -> nn.Module:
    """
    Random Unstructured Pruning 적용

    각 레이어에서 무작위로 가중치를 제거합니다.
    L1 pruning과의 비교를 위한 baseline으로 사용됩니다.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.random_unstructured(module, name='weight', amount=amount)
        elif isinstance(module, nn.Linear):
            prune.random_unstructured(module, name='weight', amount=amount)

    return model


# =============================================================================
# 6. Global Unstructured Pruning
# =============================================================================

def apply_global_unstructured_pruning(model: nn.Module, amount: float) -> nn.Module:
    """
    Global Unstructured Pruning 적용

    모든 레이어를 통합하여 전체 모델에서 가장 중요도가 낮은 가중치를 제거합니다.

    장점:
    - 레이어마다 적절한 pruning 비율이 자동 결정
    - 중요한 레이어는 적게, 덜 중요한 레이어는 많이 pruning
    - 일반적으로 per-layer pruning보다 성능이 좋음
    """
    parameters_to_prune = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )

    return model


# =============================================================================
# 7. Pruning 전후 가중치 분포 비교
# =============================================================================

def visualize_pruning_effect(
    original_model: nn.Module,
    pruned_model: nn.Module,
    save_path: str = 'pruning_effect.png'
):
    """Pruning 전후 가중치 분포 비교 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Weight Distribution: Before vs After L1 Pruning', fontsize=14)

    orig_conv_layers = []
    pruned_conv_layers = []

    for (name, module) in original_model.named_modules():
        if isinstance(module, nn.Conv2d):
            orig_conv_layers.append((name, module))

    for (name, module) in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            pruned_conv_layers.append((name, module))

    for idx in range(min(4, len(orig_conv_layers))):
        ax = axes[idx // 2][idx % 2]
        name = orig_conv_layers[idx][0]

        orig_weights = orig_conv_layers[idx][1].weight.data.cpu().numpy().flatten()
        pruned_weights = pruned_conv_layers[idx][1].weight.data.cpu().numpy().flatten()

        # 0이 아닌 가중치만 표시 (pruning 후)
        nonzero_pruned = pruned_weights[pruned_weights != 0]

        ax.hist(orig_weights, bins=100, alpha=0.5, label='Before', color='steelblue')
        ax.hist(nonzero_pruned, bins=100, alpha=0.5, label='After (nonzero)', color='coral')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.3)
        ax.set_title(f'{name}')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')
        ax.legend()

        sparsity = (pruned_weights == 0).sum() / pruned_weights.size * 100
        ax.text(0.95, 0.95, f'Sparsity: {sparsity:.1f}%',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Pruning 효과 시각화 저장: {save_path}")


# =============================================================================
# 8. 메인 실행
# =============================================================================

def main():
    print("=" * 60)
    print("Part 2: Unstructured Pruning (비정형 프루닝)")
    print("=" * 60)

    # 데이터 로드
    print("\n[1] 데이터 로드 중...")
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)

    # Baseline 모델 준비
    print("\n[2] Baseline 모델 준비...")
    model = CNN(num_classes=NUM_CLASSES).to(DEVICE)

    try:
        model.load_state_dict(torch.load('baseline_model.pth', map_location=DEVICE, weights_only=True))
        print("    저장된 Baseline 모델 로드 완료")
    except FileNotFoundError:
        print("    저장된 모델 없음. 학습 시작...")
        torch.manual_seed(SEED)
        model = CNN(num_classes=NUM_CLASSES).to(DEVICE)
        train_baseline(model, train_loader, EPOCHS, LEARNING_RATE, DEVICE)
        torch.save(model.state_dict(), 'baseline_model.pth')

    baseline_accuracy = test(model, test_loader, DEVICE)
    print(f"    Baseline Accuracy: {baseline_accuracy:.2f}%")
    print_sparsity(model, "Baseline Sparsity")

    # =============================================
    # 방법 1: L1 Unstructured Pruning
    # =============================================
    print("\n[3] L1 Unstructured Pruning")
    print("-" * 40)
    print(f"    Pruning 비율: {PRUNING_AMOUNT * 100:.0f}%")

    # 원본 모델 복사 (비교용)
    original_model = copy.deepcopy(model)

    l1_model = copy.deepcopy(model)
    l1_model = apply_l1_unstructured_pruning(l1_model, amount=PRUNING_AMOUNT)
    l1_model = remove_pruning_reparametrization(l1_model)

    l1_accuracy = test(l1_model, test_loader, DEVICE)
    print(f"    L1 Pruning Accuracy: {l1_accuracy:.2f}% (변화: {l1_accuracy - baseline_accuracy:+.2f}%)")
    print_sparsity(l1_model, "L1 Pruning 후 Sparsity")

    # Pruning 전후 비교 시각화
    visualize_pruning_effect(original_model, l1_model)

    # =============================================
    # 방법 2: Random Unstructured Pruning
    # =============================================
    print("\n[4] Random Unstructured Pruning")
    print("-" * 40)

    random_model = copy.deepcopy(model)
    random_model = apply_random_unstructured_pruning(random_model, amount=PRUNING_AMOUNT)
    random_model = remove_pruning_reparametrization(random_model)

    random_accuracy = test(random_model, test_loader, DEVICE)
    print(f"    Random Pruning Accuracy: {random_accuracy:.2f}% (변화: {random_accuracy - baseline_accuracy:+.2f}%)")
    print_sparsity(random_model, "Random Pruning 후 Sparsity")

    # =============================================
    # 방법 3: Global Unstructured Pruning
    # =============================================
    print("\n[5] Global Unstructured Pruning")
    print("-" * 40)

    global_model = copy.deepcopy(model)
    global_model = apply_global_unstructured_pruning(global_model, amount=PRUNING_AMOUNT)
    global_model = remove_pruning_reparametrization(global_model)

    global_accuracy = test(global_model, test_loader, DEVICE)
    print(f"    Global Pruning Accuracy: {global_accuracy:.2f}% (변화: {global_accuracy - baseline_accuracy:+.2f}%)")
    print_sparsity(global_model, "Global Pruning 후 Sparsity (레이어별 비율이 다름!)")

    # =============================================
    # Sparsity별 정확도 실험 (L1 기준)
    # =============================================
    print("\n[6] Sparsity별 정확도 변화 (L1 Unstructured)")
    print("-" * 40)

    sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    l1_sparsity_results = {}

    for amount in sparsity_levels:
        test_model = copy.deepcopy(model)
        test_model = apply_l1_unstructured_pruning(test_model, amount=amount)
        test_model = remove_pruning_reparametrization(test_model)
        acc = test(test_model, test_loader, DEVICE)
        l1_sparsity_results[amount * 100] = acc
        print(f"    Sparsity {amount * 100:5.1f}%: Accuracy {acc:.2f}%")

    # Sparsity vs Accuracy 시각화
    plt.figure(figsize=(10, 6))
    sparsities = list(l1_sparsity_results.keys())
    accuracies = list(l1_sparsity_results.values())

    plt.plot(sparsities, accuracies, 'bo-', linewidth=2, markersize=8, label='L1 Unstructured')
    plt.axhline(y=baseline_accuracy, color='r', linestyle='--', alpha=0.7, label=f'Baseline ({baseline_accuracy:.2f}%)')

    plt.xlabel('Sparsity (%)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('L1 Unstructured Pruning: Sparsity vs Accuracy', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sparsity_vs_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Sparsity vs Accuracy 그래프 저장: sparsity_vs_accuracy.png")

    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약: Unstructured Pruning (30% pruning)")
    print("=" * 60)
    print(f"{'방법':<35} {'정확도':<10} {'vs Baseline':<12}")
    print("-" * 60)
    print(f"{'Baseline (no pruning)':<35} {baseline_accuracy:>8.2f}%")
    print(f"{'L1 Unstructured':<35} {l1_accuracy:>8.2f}% {l1_accuracy - baseline_accuracy:>+10.2f}%")
    print(f"{'Random Unstructured':<35} {random_accuracy:>8.2f}% {random_accuracy - baseline_accuracy:>+10.2f}%")
    print(f"{'Global Unstructured (L1)':<35} {global_accuracy:>8.2f}% {global_accuracy - baseline_accuracy:>+10.2f}%")
    print("-" * 60)

    print("\n분석:")
    if l1_accuracy > random_accuracy:
        print("- L1 > Random: Magnitude 기반 pruning이 랜덤보다 효과적")
    else:
        print("- Random >= L1: 이 pruning 비율에서는 차이가 크지 않음")

    if global_accuracy > l1_accuracy:
        print("- Global > Per-layer: 전역적으로 중요도를 판단하는 것이 더 효과적")
    else:
        print("- Per-layer >= Global: 레이어별 pruning도 충분히 효과적")

    return {
        'baseline_accuracy': baseline_accuracy,
        'l1_accuracy': l1_accuracy,
        'random_accuracy': random_accuracy,
        'global_accuracy': global_accuracy,
        'sparsity_results': l1_sparsity_results
    }


if __name__ == "__main__":
    results = main()
