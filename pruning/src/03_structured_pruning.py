"""
Part 3: Structured Pruning (정형 프루닝) 구현

이 스크립트에서 다루는 내용:
1. Structured Pruning의 핵심 개념
2. Ln Structured Pruning (필터 단위 제거)
3. Random Structured Pruning
4. Fine-tuning (pruning 후 재학습)
5. Unstructured vs Structured 비교

Structured Pruning (정형 프루닝):
================================

Unstructured pruning은 개별 가중치를 0으로 만들지만,
실제로 행렬 크기는 변하지 않아 속도 향상이 제한적입니다.

Structured pruning은 필터/채널/레이어 단위로 제거하여
실제 연산량과 메모리를 줄입니다.

방법 1: Ln Structured Pruning
-----------------------------
- Conv2d 필터의 Ln norm 기준으로 중요도 판단
- L1 norm (Manhattan): Σ|w| - 가중치 절대값의 합
- L2 norm (Euclidean): √(Σw²) - 가중치 크기
- norm이 작은 필터 = 출력 feature map에 기여도 낮음

방법 2: Random Structured Pruning
----------------------------------
- 무작위로 필터를 제거
- Ln 방법과의 비교를 위한 baseline

Fine-tuning (재학습):
====================
- Pruning 후 정확도가 하락할 수 있음
- 소량의 추가 학습으로 정확도 회복 가능
- 핵심: pruning -> fine-tuning -> (반복 가능)
- 일반적으로 원래 학습률의 1/10로 fine-tuning

Structured vs Unstructured 비교:
================================
| 특성          | Unstructured    | Structured      |
|--------------|-----------------|-----------------|
| 단위          | 개별 가중치       | 필터/채널        |
| 압축률        | 높음             | 중간             |
| 속도 향상     | 제한적 (sparse)  | 실질적           |
| 정확도 유지   | 좋음             | 상대적으로 어려움  |
| HW 지원      | 특수 HW 필요     | 범용 HW 가능     |
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

# Structured Pruning 하이퍼파라미터
PRUNING_AMOUNT = 0.3  # 30% 필터 제거
FINETUNE_EPOCHS = 3   # Fine-tuning 에포크 수
FINETUNE_LR = 0.0001  # Fine-tuning 학습률 (원래의 1/10)

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
# 3. 유틸리티 함수
# =============================================================================

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


def get_sparsity(model: nn.Module) -> dict:
    """모델의 sparsity 분석"""
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
    print(f"    {'Layer':<30} {'Sparsity':<10} {'0s / Total':<20}")
    print(f"    {'-' * 60}")
    for name, stats in info['layer_sparsity'].items():
        print(f"    {name:<30} {stats['sparsity']:>7.1f}%   {stats['zeros']:>8,} / {stats['total']:>8,}")


# =============================================================================
# 4. Structured Pruning 적용
# =============================================================================

def apply_ln_structured_pruning(model: nn.Module, amount: float, n: int = 1) -> nn.Module:
    """
    Ln Structured Pruning 적용

    Conv2d 레이어의 필터(output channel)를 Ln norm 기준으로 pruning합니다.

    Args:
        model: pruning할 모델
        amount: 제거할 필터 비율 (0~1)
        n: norm 차수 (1=L1, 2=L2)

    동작:
    - dim=0: output channel(필터) 단위로 pruning
    - 각 필터의 Ln norm을 계산하여 가장 작은 것을 제거
    - 예: Conv2d(64, 32, 3, 3)에서 amount=0.3이면 9개 필터 제거
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=n, dim=0)

    return model


def apply_random_structured_pruning(model: nn.Module, amount: float) -> nn.Module:
    """Random Structured Pruning 적용"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.random_structured(module, name='weight', amount=amount, dim=0)

    return model


def remove_pruning_reparametrization(model: nn.Module) -> nn.Module:
    """Pruning mask를 영구 적용"""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass
    return model


# =============================================================================
# 5. Fine-tuning (Pruning 후 재학습)
# =============================================================================

def fine_tune(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str
) -> list[float]:
    """
    Pruning 후 Fine-tuning

    Pruning으로 인한 정확도 하락을 회복하기 위해
    소량의 추가 학습을 수행합니다.

    핵심:
    - 낮은 학습률 사용 (원래의 1/10)
    - 적은 에포크 (2~5 에포크)
    - Pruning mask를 유지한 채 fine-tuning해야 함 (Han et al., 2015)
      → forward 시 weight = weight_orig * weight_mask
      → pruned 가중치는 항상 0 유지
      → fine-tuning 완료 후 prune.remove()로 mask 영구 적용
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
        print(f"    Fine-tune Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return epoch_losses


# =============================================================================
# 6. 필터별 norm 시각화
# =============================================================================

def visualize_filter_norms(model: nn.Module, save_path: str = 'filter_norms.png'):
    """
    Conv2d 레이어의 필터별 L1 norm 시각화

    Pruning 전에 필터의 중요도 분포를 확인할 수 있습니다.
    norm이 작은 필터 = pruning 대상
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Filter L1 Norms by Layer (Candidates for Structured Pruning)', fontsize=14)

    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))

    for idx, (name, layer) in enumerate(conv_layers[:4]):
        ax = axes[idx // 2][idx % 2]

        # 각 필터의 L1 norm 계산
        # weight shape: (out_channels, in_channels, H, W)
        weight = layer.weight.data.cpu()
        filter_norms = weight.abs().sum(dim=[1, 2, 3]).numpy()

        # 정렬하여 시각화
        sorted_norms = np.sort(filter_norms)
        colors = ['coral' if n < np.percentile(filter_norms, 30) else 'steelblue' for n in sorted_norms]

        ax.bar(range(len(sorted_norms)), sorted_norms, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(f'{name} ({layer.weight.shape[0]} filters)')
        ax.set_xlabel('Filter Index (sorted by norm)')
        ax.set_ylabel('L1 Norm')

        # 30% 커트라인 표시
        cutoff = np.percentile(filter_norms, 30)
        ax.axhline(y=cutoff, color='red', linestyle='--', alpha=0.7, label=f'30% cutoff: {cutoff:.3f}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    필터 norm 시각화 저장: {save_path}")


# =============================================================================
# 7. 메인 실행
# =============================================================================

def main():
    print("=" * 60)
    print("Part 3: Structured Pruning (정형 프루닝)")
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

    # 필터 norm 시각화
    print("\n[3] 필터별 L1 Norm 시각화")
    print("-" * 40)
    visualize_filter_norms(model)

    # =============================================
    # 방법 1: L1 Structured Pruning
    # =============================================
    print(f"\n[4] L1 Structured Pruning ({PRUNING_AMOUNT * 100:.0f}% 필터 제거)")
    print("-" * 40)

    l1_model = copy.deepcopy(model)
    l1_model = apply_ln_structured_pruning(l1_model, amount=PRUNING_AMOUNT, n=1)
    l1_model = remove_pruning_reparametrization(l1_model)

    l1_accuracy = test(l1_model, test_loader, DEVICE)
    print(f"    L1 Structured Accuracy: {l1_accuracy:.2f}% (변화: {l1_accuracy - baseline_accuracy:+.2f}%)")
    print_sparsity(l1_model, "L1 Structured Pruning 후 Sparsity")

    # =============================================
    # 방법 2: L2 Structured Pruning
    # =============================================
    print(f"\n[5] L2 Structured Pruning ({PRUNING_AMOUNT * 100:.0f}% 필터 제거)")
    print("-" * 40)

    l2_model = copy.deepcopy(model)
    l2_model = apply_ln_structured_pruning(l2_model, amount=PRUNING_AMOUNT, n=2)
    l2_model = remove_pruning_reparametrization(l2_model)

    l2_accuracy = test(l2_model, test_loader, DEVICE)
    print(f"    L2 Structured Accuracy: {l2_accuracy:.2f}% (변화: {l2_accuracy - baseline_accuracy:+.2f}%)")

    # =============================================
    # 방법 3: Random Structured Pruning
    # =============================================
    print(f"\n[6] Random Structured Pruning ({PRUNING_AMOUNT * 100:.0f}% 필터 제거)")
    print("-" * 40)

    random_model = copy.deepcopy(model)
    random_model = apply_random_structured_pruning(random_model, amount=PRUNING_AMOUNT)
    random_model = remove_pruning_reparametrization(random_model)

    random_accuracy = test(random_model, test_loader, DEVICE)
    print(f"    Random Structured Accuracy: {random_accuracy:.2f}% (변화: {random_accuracy - baseline_accuracy:+.2f}%)")

    # =============================================
    # Fine-tuning 효과 검증
    # =============================================
    print(f"\n[7] Fine-tuning 효과 (L1 Structured + Fine-tune)")
    print("-" * 40)
    print(f"    Fine-tune Epochs: {FINETUNE_EPOCHS}")
    print(f"    Fine-tune LR: {FINETUNE_LR}")

    ft_model = copy.deepcopy(model)
    ft_model = apply_ln_structured_pruning(ft_model, amount=PRUNING_AMOUNT, n=1)
    # mask 유지한 채로 정확도 확인
    before_ft_accuracy = test(ft_model, test_loader, DEVICE)
    print(f"    Fine-tune 전: {before_ft_accuracy:.2f}%")

    # mask 유지한 채 fine-tuning → pruned 가중치는 0 유지
    fine_tune(ft_model, train_loader, FINETUNE_EPOCHS, FINETUNE_LR, DEVICE)
    # fine-tuning 완료 후 mask 영구 적용
    ft_model = remove_pruning_reparametrization(ft_model)
    after_ft_accuracy = test(ft_model, test_loader, DEVICE)
    print(f"    Fine-tune 후: {after_ft_accuracy:.2f}% (회복: {after_ft_accuracy - before_ft_accuracy:+.2f}%)")

    torch.save(ft_model.state_dict(), 'structured_pruned_finetuned.pth')
    print("    Fine-tuned 모델 저장: structured_pruned_finetuned.pth")

    # =============================================
    # Sparsity별 정확도 실험 (L1 Structured)
    # =============================================
    print("\n[8] Sparsity별 정확도 변화 (L1 Structured)")
    print("-" * 40)

    sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    structured_results = {}
    structured_ft_results = {}

    for amount in sparsity_levels:
        # Pruning only (mask 영구 적용 후 평가)
        test_model = copy.deepcopy(model)
        test_model = apply_ln_structured_pruning(test_model, amount=amount, n=1)
        test_model = remove_pruning_reparametrization(test_model)
        acc = test(test_model, test_loader, DEVICE)
        structured_results[amount * 100] = acc

        # Pruning + Fine-tuning (mask 유지한 채 fine-tuning → 이후 remove)
        ft_model = copy.deepcopy(model)
        ft_model = apply_ln_structured_pruning(ft_model, amount=amount, n=1)
        fine_tune(ft_model, train_loader, FINETUNE_EPOCHS, FINETUNE_LR, DEVICE)
        ft_model = remove_pruning_reparametrization(ft_model)
        ft_acc = test(ft_model, test_loader, DEVICE)
        structured_ft_results[amount * 100] = ft_acc

        print(f"    Sparsity {amount * 100:5.1f}%: Pruning only {acc:.2f}% | + Fine-tune {ft_acc:.2f}%")

    # Sparsity vs Accuracy 시각화 (Pruning only vs Fine-tuned)
    plt.figure(figsize=(10, 6))
    sparsities = list(structured_results.keys())
    acc_pruned = list(structured_results.values())
    acc_finetuned = list(structured_ft_results.values())

    plt.plot(sparsities, acc_pruned, 'bo-', linewidth=2, markersize=8, label='Pruning Only')
    plt.plot(sparsities, acc_finetuned, 'gs-', linewidth=2, markersize=8, label='Pruning + Fine-tune')
    plt.axhline(y=baseline_accuracy, color='r', linestyle='--', alpha=0.7, label=f'Baseline ({baseline_accuracy:.2f}%)')

    plt.xlabel('Sparsity (%)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Structured Pruning: Sparsity vs Accuracy (with Fine-tuning)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('structured_sparsity_vs_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Structured Sparsity vs Accuracy 저장: structured_sparsity_vs_accuracy.png")

    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약: Structured Pruning (30% 필터 제거)")
    print("=" * 60)
    print(f"{'방법':<40} {'정확도':<10} {'vs Baseline':<12}")
    print("-" * 65)
    print(f"{'Baseline (no pruning)':<40} {baseline_accuracy:>8.2f}%")
    print(f"{'L1 Structured':<40} {l1_accuracy:>8.2f}% {l1_accuracy - baseline_accuracy:>+10.2f}%")
    print(f"{'L2 Structured':<40} {l2_accuracy:>8.2f}% {l2_accuracy - baseline_accuracy:>+10.2f}%")
    print(f"{'Random Structured':<40} {random_accuracy:>8.2f}% {random_accuracy - baseline_accuracy:>+10.2f}%")
    print(f"{'L1 Structured + Fine-tune':<40} {after_ft_accuracy:>8.2f}% {after_ft_accuracy - baseline_accuracy:>+10.2f}%")
    print("-" * 65)

    print("\n분석:")
    print(f"- Fine-tuning으로 {after_ft_accuracy - l1_accuracy:+.2f}% 정확도 회복")
    if l1_accuracy > random_accuracy:
        print("- L1 norm 기반 pruning이 랜덤보다 효과적")
    print("- Structured pruning은 실제 연산량 감소에 효과적")
    print("  (0인 필터를 물리적으로 제거하면 실제 속도 향상)")

    return {
        'baseline_accuracy': baseline_accuracy,
        'l1_accuracy': l1_accuracy,
        'l2_accuracy': l2_accuracy,
        'random_accuracy': random_accuracy,
        'finetuned_accuracy': after_ft_accuracy,
        'structured_results': structured_results,
        'structured_ft_results': structured_ft_results
    }


if __name__ == "__main__":
    results = main()
