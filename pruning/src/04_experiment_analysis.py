"""
Part 4: 실험 결과 비교 및 시각화

이 스크립트에서 다루는 내용:
1. 모든 Pruning 방법 통합 실험
2. Unstructured vs Structured 비교
3. Sparsity 수준별 성능 비교
4. Fine-tuning 효과 분석
5. 학습 곡선 시각화
6. 최종 성능 비교 테이블

실험 구성:
==========
1. Baseline: Pruning 없이 학습된 모델
2. L1 Unstructured: 개별 가중치 magnitude 기반 제거
3. Global Unstructured: 전역 magnitude 기반 제거
4. L1 Structured: 필터 단위 L1 norm 기반 제거
5. L1 Structured + Fine-tune: Pruning 후 재학습
6. Iterative Pruning: 점진적 pruning + fine-tuning 반복
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
SEED = 42

# Fine-tuning
FINETUNE_EPOCHS = 3
FINETUNE_LR = 0.0001

# 결과 저장 디렉토리
os.makedirs('results', exist_ok=True)


# =============================================================================
# 2. 데이터 로드
# =============================================================================

def get_data_loaders(batch_size: int = 128):
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
# 3. 학습/평가/Pruning 함수
# =============================================================================

def train_baseline(model, train_loader, epochs, lr, device):
    """일반 CE 학습"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    losses = []
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
        losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    return losses


def test(model, test_loader, device):
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
    return 100 * correct / total


def fine_tune(model, train_loader, epochs, lr, device):
    """Pruning 후 Fine-tuning"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    losses = []
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
        losses.append(avg_loss)
        print(f"  Fine-tune Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    return losses


def apply_l1_unstructured(model, amount):
    """L1 Unstructured Pruning"""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model


def apply_global_unstructured(model, amount):
    """Global Unstructured Pruning"""
    params = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params.append((module, 'weight'))
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    return model


def apply_l1_structured(model, amount):
    """L1 Structured Pruning"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
    return model


def remove_pruning(model):
    """Pruning mask 영구 적용"""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass
    return model


def get_sparsity(model):
    """전체 sparsity 계산"""
    total = 0
    zeros = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            total += weight.numel()
            zeros += (weight == 0).sum().item()
    return zeros / total * 100 if total > 0 else 0


# =============================================================================
# 4. Iterative Pruning (점진적 프루닝)
# =============================================================================

def iterative_pruning(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    target_sparsity: float,
    num_iterations: int,
    finetune_epochs: int,
    finetune_lr: float,
    device: str
) -> tuple[nn.Module, list[dict]]:
    """
    Iterative Pruning: 점진적으로 pruning + fine-tuning 반복

    한 번에 많이 pruning하면 정확도가 크게 하락하므로,
    조금씩 pruning하고 fine-tuning하는 과정을 반복합니다.

    예: 50% sparsity 목표, 5회 반복
    → 매 회 ~13% pruning + fine-tuning
    → (1-0.87)^5 ≈ 0.50

    Args:
        target_sparsity: 최종 목표 sparsity (0~1)
        num_iterations: 반복 횟수
    """
    # 매 반복마다 pruning할 비율 계산
    # (1 - per_iter_amount)^num_iterations = (1 - target_sparsity)
    per_iter_amount = 1 - (1 - target_sparsity) ** (1 / num_iterations)

    history = []
    print(f"    목표 Sparsity: {target_sparsity * 100:.1f}%")
    print(f"    반복 횟수: {num_iterations}")
    print(f"    매 반복 pruning 비율: {per_iter_amount * 100:.1f}%")

    for i in range(num_iterations):
        print(f"\n    --- Iteration {i + 1}/{num_iterations} ---")

        # Pruning 적용 (mask 유지)
        model = apply_l1_unstructured(model, amount=per_iter_amount)

        # Pruning 후 정확도 (mask 유지 상태에서 평가)
        prune_acc = test(model, test_loader, device)
        sparsity = get_sparsity(model)
        print(f"    Pruning 후: Accuracy {prune_acc:.2f}%, Sparsity {sparsity:.1f}%")

        # Fine-tuning (mask 유지 → pruned 가중치 0 유지)
        fine_tune(model, train_loader, finetune_epochs, finetune_lr, device)
        ft_acc = test(model, test_loader, device)
        print(f"    Fine-tune 후: Accuracy {ft_acc:.2f}%")

        # Fine-tuning 완료 후 mask 영구 적용 (다음 iteration을 위해)
        model = remove_pruning(model)

        history.append({
            'iteration': i + 1,
            'sparsity': sparsity,
            'prune_accuracy': prune_acc,
            'finetune_accuracy': ft_acc
        })

    return model, history


# =============================================================================
# 5. 시각화 함수
# =============================================================================

def plot_method_comparison(results: dict, save_path: str = 'results/accuracy_comparison.png'):
    """방법별 정확도 비교 바 차트"""
    methods = list(results.keys())
    accuracies = list(results.values())

    baseline_acc = results.get('Baseline', 0)
    colors = ['#9E9E9E' if m == 'Baseline' else
              ('#4CAF50' if acc >= baseline_acc - 1 else '#FF9800' if acc >= baseline_acc - 3 else '#F44336')
              for m, acc in zip(methods, accuracies)]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.2)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{acc:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Pruning Methods - Test Accuracy Comparison', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(min(accuracies) - 5, max(accuracies) + 3)
    plt.axhline(y=baseline_acc, color='gray', linestyle=':', alpha=0.5)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"방법별 비교 차트 저장: {save_path}")


def plot_sparsity_comparison(
    unstructured_results: dict,
    structured_results: dict,
    structured_ft_results: dict,
    baseline_acc: float,
    save_path: str = 'results/sparsity_comparison.png'
):
    """Unstructured vs Structured Sparsity별 정확도 비교"""
    plt.figure(figsize=(12, 6))

    # Unstructured
    if unstructured_results:
        sp = list(unstructured_results.keys())
        acc = list(unstructured_results.values())
        plt.plot(sp, acc, 'bo-', linewidth=2, markersize=8, label='L1 Unstructured')

    # Structured (pruning only)
    if structured_results:
        sp = list(structured_results.keys())
        acc = list(structured_results.values())
        plt.plot(sp, acc, 'rs-', linewidth=2, markersize=8, label='L1 Structured')

    # Structured + Fine-tune
    if structured_ft_results:
        sp = list(structured_ft_results.keys())
        acc = list(structured_ft_results.values())
        plt.plot(sp, acc, 'g^-', linewidth=2, markersize=8, label='L1 Structured + Fine-tune')

    plt.axhline(y=baseline_acc, color='gray', linestyle='--', alpha=0.7, label=f'Baseline ({baseline_acc:.2f}%)')

    plt.xlabel('Sparsity (%)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Pruning Methods: Sparsity vs Accuracy Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sparsity 비교 차트 저장: {save_path}")


def plot_iterative_pruning(history: list[dict], save_path: str = 'results/iterative_pruning.png'):
    """Iterative Pruning 과정 시각화"""
    iterations = [h['iteration'] for h in history]
    sparsities = [h['sparsity'] for h in history]
    prune_accs = [h['prune_accuracy'] for h in history]
    ft_accs = [h['finetune_accuracy'] for h in history]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 정확도 (왼쪽 축)
    ax1.plot(iterations, prune_accs, 'ro--', linewidth=1.5, markersize=8, label='After Pruning')
    ax1.plot(iterations, ft_accs, 'go-', linewidth=2, markersize=8, label='After Fine-tuning')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.legend(loc='center left')

    # Sparsity (오른쪽 축)
    ax2 = ax1.twinx()
    ax2.bar(iterations, sparsities, alpha=0.2, color='blue', label='Sparsity')
    ax2.set_ylabel('Sparsity (%)', fontsize=12, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.legend(loc='center right')

    plt.title('Iterative Pruning: Accuracy & Sparsity per Iteration', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Iterative Pruning 시각화 저장: {save_path}")


def create_summary_table(results: dict, save_path: str = 'results/summary.txt'):
    """결과 요약 테이블 생성"""
    baseline_acc = results.get('Baseline', 0)

    lines = [
        "=" * 70,
        "Pruning 실험 결과 요약",
        "=" * 70,
        "",
        f"{'Method':<40} {'Accuracy':<12} {'vs Baseline':<12}",
        "-" * 70,
    ]

    for method, acc in results.items():
        diff = acc - baseline_acc if method != 'Baseline' else 0
        diff_str = f"{diff:+.2f}%" if diff != 0 else "-"
        lines.append(f"{method:<40} {acc:>10.2f}% {diff_str:>12}")

    lines.extend([
        "-" * 70,
        "",
        "분석:",
    ])

    # 최고 성능 방법 찾기 (Baseline 제외)
    non_baseline = {k: v for k, v in results.items() if k != 'Baseline'}
    if non_baseline:
        best_method = max(non_baseline, key=non_baseline.get)
        best_acc = non_baseline[best_method]
        lines.extend([
            f"- 최고 Pruning 성능: {best_method} ({best_acc:.2f}%)",
            f"- Baseline 대비 변화: {best_acc - baseline_acc:+.2f}%",
        ])

    content = "\n".join(lines)
    print(content)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n결과 요약 저장: {save_path}")


# =============================================================================
# 6. 메인 실험
# =============================================================================

def main():
    print("=" * 70)
    print("Part 4: Pruning 종합 실험 및 분석")
    print("=" * 70)

    # 데이터 로드
    print("\n[1] 데이터 로드...")
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)

    # 결과 저장용
    all_accuracies = {}

    # ===========================================
    # Baseline 모델 준비
    # ===========================================
    print("\n[2] Baseline 모델 준비...")
    model = CNN(num_classes=NUM_CLASSES).to(DEVICE)

    try:
        model.load_state_dict(torch.load('baseline_model.pth', map_location=DEVICE, weights_only=True))
        print("    저장된 Baseline 로드 완료")
    except FileNotFoundError:
        print("    Baseline 학습 중...")
        torch.manual_seed(SEED)
        train_baseline(model, train_loader, EPOCHS, LEARNING_RATE, DEVICE)
        torch.save(model.state_dict(), 'baseline_model.pth')

    baseline_acc = test(model, test_loader, DEVICE)
    all_accuracies['Baseline'] = baseline_acc
    print(f"    Baseline Accuracy: {baseline_acc:.2f}%")

    # ===========================================
    # L1 Unstructured (30%)
    # ===========================================
    print("\n[3] L1 Unstructured (30%)...")
    l1_model = copy.deepcopy(model)
    l1_model = apply_l1_unstructured(l1_model, amount=0.3)
    l1_model = remove_pruning(l1_model)
    l1_acc = test(l1_model, test_loader, DEVICE)
    all_accuracies['L1 Unstructured (30%)'] = l1_acc
    print(f"    Accuracy: {l1_acc:.2f}%")

    # ===========================================
    # Global Unstructured (30%)
    # ===========================================
    print("\n[4] Global Unstructured (30%)...")
    global_model = copy.deepcopy(model)
    global_model = apply_global_unstructured(global_model, amount=0.3)
    global_model = remove_pruning(global_model)
    global_acc = test(global_model, test_loader, DEVICE)
    all_accuracies['Global Unstructured (30%)'] = global_acc
    print(f"    Accuracy: {global_acc:.2f}%")

    # ===========================================
    # L1 Structured (30%)
    # ===========================================
    print("\n[5] L1 Structured (30%)...")
    struct_model = copy.deepcopy(model)
    struct_model = apply_l1_structured(struct_model, amount=0.3)
    struct_model = remove_pruning(struct_model)
    struct_acc = test(struct_model, test_loader, DEVICE)
    all_accuracies['L1 Structured (30%)'] = struct_acc
    print(f"    Accuracy: {struct_acc:.2f}%")

    # ===========================================
    # L1 Structured (30%) + Fine-tune
    # ===========================================
    print("\n[6] L1 Structured (30%) + Fine-tune...")
    struct_ft_model = copy.deepcopy(model)
    struct_ft_model = apply_l1_structured(struct_ft_model, amount=0.3)
    fine_tune(struct_ft_model, train_loader, FINETUNE_EPOCHS, FINETUNE_LR, DEVICE)
    struct_ft_model = remove_pruning(struct_ft_model)
    struct_ft_acc = test(struct_ft_model, test_loader, DEVICE)
    all_accuracies['L1 Structured + Fine-tune'] = struct_ft_acc
    print(f"    Accuracy: {struct_ft_acc:.2f}%")

    # ===========================================
    # Iterative Pruning (50% target, 5 iterations)
    # ===========================================
    print("\n[7] Iterative Pruning (50% target)...")
    iter_model = copy.deepcopy(model)
    iter_model, iter_history = iterative_pruning(
        iter_model, train_loader, test_loader,
        target_sparsity=0.5,
        num_iterations=5,
        finetune_epochs=FINETUNE_EPOCHS,
        finetune_lr=FINETUNE_LR,
        device=DEVICE
    )
    iter_acc = test(iter_model, test_loader, DEVICE)
    all_accuracies['Iterative (50% target)'] = iter_acc
    print(f"    Final Accuracy: {iter_acc:.2f}%")

    # ===========================================
    # Sparsity 수준별 비교 실험
    # ===========================================
    print("\n[8] Sparsity 수준별 비교 실험...")
    sparsity_levels = [10, 20, 30, 40, 50, 60, 70, 80]

    unstructured_results = {}
    structured_results = {}
    structured_ft_results = {}

    for sp in sparsity_levels:
        amount = sp / 100

        # L1 Unstructured
        m = copy.deepcopy(model)
        m = apply_l1_unstructured(m, amount=amount)
        m = remove_pruning(m)
        acc = test(m, test_loader, DEVICE)
        unstructured_results[sp] = acc

        # L1 Structured (pruning only)
        m = copy.deepcopy(model)
        m = apply_l1_structured(m, amount=amount)
        m = remove_pruning(m)
        acc = test(m, test_loader, DEVICE)
        structured_results[sp] = acc

        # L1 Structured + Fine-tune (mask 유지한 채 fine-tuning)
        ft_m = copy.deepcopy(model)
        ft_m = apply_l1_structured(ft_m, amount=amount)
        fine_tune(ft_m, train_loader, FINETUNE_EPOCHS, FINETUNE_LR, DEVICE)
        ft_m = remove_pruning(ft_m)
        ft_acc = test(ft_m, test_loader, DEVICE)
        structured_ft_results[sp] = ft_acc

        print(f"    Sparsity {sp}%: Unstruct {unstructured_results[sp]:.2f}% | "
              f"Struct {structured_results[sp]:.2f}% | Struct+FT {structured_ft_results[sp]:.2f}%")

    # ===========================================
    # 시각화
    # ===========================================
    print("\n[9] 결과 시각화...")
    plot_method_comparison(all_accuracies)
    plot_sparsity_comparison(unstructured_results, structured_results, structured_ft_results, baseline_acc)
    plot_iterative_pruning(iter_history)
    create_summary_table(all_accuracies)

    # ===========================================
    # 최종 요약
    # ===========================================
    print("\n" + "=" * 70)
    print("실험 완료!")
    print("=" * 70)
    print("\n생성된 파일:")
    print("- results/accuracy_comparison.png   : 방법별 정확도 비교")
    print("- results/sparsity_comparison.png    : Sparsity별 정확도 비교")
    print("- results/iterative_pruning.png      : Iterative Pruning 과정")
    print("- results/summary.txt               : 결과 요약")

    return {
        'accuracies': all_accuracies,
        'unstructured_results': unstructured_results,
        'structured_results': structured_results,
        'structured_ft_results': structured_ft_results,
        'iterative_history': iter_history
    }


if __name__ == "__main__":
    results = main()
