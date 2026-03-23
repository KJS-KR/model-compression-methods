"""
Part 2: Random Search NAS

이 스크립트에서 다루는 내용:
1. Random Search 전략의 원리
2. 탐색 공간에서 무작위 아키텍처 샘플링 및 평가
3. 최적 아키텍처 선정 및 전체 학습
4. Baseline과 비교 분석

Random Search 핵심 아이디어:
- 탐색 공간에서 N개의 아키텍처를 무작위로 샘플링
- 각 아키텍처를 축소 학습(reduced training)으로 빠르게 평가
- 가장 성능이 좋은 아키텍처를 최종 선택하여 전체 학습

Random Search의 장점:
- 구현이 매우 간단 (baseline 탐색 전략)
- 병렬화 용이 (각 아키텍처 독립 평가)
- 놀라울 정도로 경쟁력 있는 성능 (Li & Talwalkar, 2020)
- 다른 탐색 전략의 비교 기준으로 사용

성능 추정 - Reduced Training:
============================
- 전체 학습(10 epochs) 대신 축소 학습(5 epochs)으로 근사
- 가정: "5 epochs에서 좋은 아키텍처 ≈ 10 epochs에서도 좋은 아키텍처"
- 정확한 순위 보존은 보장되지 않지만, 실용적으로 효과적
- 탐색 비용을 ~50% 절감
"""

import sys
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'module'))
from models import (
    CNN, FlexibleCNN, SEARCH_SPACE,
    sample_architecture, architecture_to_string
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
LEARNING_RATE = 0.001
NUM_CLASSES = 10

# NAS 전용 하이퍼파라미터
SEARCH_EPOCHS = 5       # 탐색 시 축소 학습 에포크
FULL_EPOCHS = 10        # 최종 재학습 에포크
NUM_SAMPLES = 20        # Random Search 샘플 수

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
    device: str,
    verbose: bool = True
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
        if verbose:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

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

    return 100 * correct / total


def count_parameters(model: nn.Module) -> int:
    """모델의 총 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# 4. 아키텍처 평가 함수
# =============================================================================

def train_and_evaluate(
    arch: dict,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    seed: int = 42
) -> dict:
    """
    하나의 아키텍처를 학습하고 평가

    Args:
        arch: 아키텍처 구성 딕셔너리
        train_loader: 학습 데이터 로더
        test_loader: 테스트 데이터 로더
        epochs: 학습 에포크 수
        learning_rate: 학습률
        device: 학습 디바이스
        seed: 재현성을 위한 시드

    Returns:
        dict: {arch, accuracy, params, train_losses, train_time}
    """
    torch.manual_seed(seed)
    model = FlexibleCNN(arch, num_classes=NUM_CLASSES)
    params = count_parameters(model)

    start_time = time.time()
    train_losses = train(model, train_loader, epochs, learning_rate, device, verbose=False)
    train_time = time.time() - start_time

    accuracy = test(model, test_loader, device)

    return {
        'arch': arch,
        'accuracy': accuracy,
        'params': params,
        'train_losses': train_losses,
        'train_time': train_time
    }


# =============================================================================
# 5. Random Search
# =============================================================================

def random_search(
    num_samples: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    search_epochs: int,
    learning_rate: float,
    device: str,
    base_seed: int = 42
) -> list[dict]:
    """
    Random Search NAS

    탐색 공간에서 num_samples개의 아키텍처를 무작위 샘플링하고,
    축소 학습으로 빠르게 평가합니다.

    Args:
        num_samples: 샘플링할 아키텍처 수
        train_loader: 학습 데이터 로더
        test_loader: 테스트 데이터 로더
        search_epochs: 축소 학습 에포크 수
        learning_rate: 학습률
        device: 디바이스
        base_seed: 기본 시드

    Returns:
        list[dict]: 각 아키텍처의 평가 결과 리스트
    """
    results = []

    print(f"\nRandom Search: {num_samples}개 아키텍처 탐색 시작")
    print(f"  축소 학습: {search_epochs} epochs per architecture")
    print("-" * 60)

    for i in tqdm(range(num_samples), desc="Random Search"):
        # 아키텍처 샘플링
        arch = sample_architecture(seed=base_seed + i)
        arch_str = architecture_to_string(arch)

        # 학습 및 평가
        result = train_and_evaluate(
            arch, train_loader, test_loader,
            search_epochs, learning_rate, device, seed=base_seed
        )
        results.append(result)

        print(f"  [{i+1}/{num_samples}] {arch_str} | "
              f"Acc: {result['accuracy']:.2f}% | "
              f"Params: {result['params']:,} | "
              f"Time: {result['train_time']:.1f}s")

    # 결과를 정확도 기준 내림차순 정렬
    results.sort(key=lambda x: x['accuracy'], reverse=True)

    print("-" * 60)
    print(f"Best: {architecture_to_string(results[0]['arch'])} | "
          f"Acc: {results[0]['accuracy']:.2f}%")

    return results


# =============================================================================
# 6. 시각화
# =============================================================================

def visualize_random_search_results(results: list[dict], baseline_accuracy: float,
                                     baseline_params: int,
                                     save_path: str = 'results/random_search_results.png'):
    """
    Random Search 결과 시각화

    1. 파라미터 수 vs 정확도 scatter plot
    2. 탐색 진행에 따른 best-so-far 정확도 곡선
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    accuracies = [r['accuracy'] for r in results]
    params = [r['params'] for r in results]

    # --- Plot 1: 파라미터 수 vs 정확도 ---
    ax1 = axes[0]
    # 탐색 순서대로 (정렬 전) scatter
    search_order_results = sorted(results, key=lambda x: results.index(x))
    search_accs = [r['accuracy'] for r in search_order_results]
    search_params = [r['params'] for r in search_order_results]

    scatter = ax1.scatter(search_params, search_accs, c=range(len(results)),
                          cmap='viridis', s=80, edgecolors='black', linewidth=0.5,
                          zorder=3)
    plt.colorbar(scatter, ax=ax1, label='Search Order')

    # Baseline 표시
    ax1.scatter([baseline_params], [baseline_accuracy], c='red', s=200,
                marker='*', edgecolors='black', linewidth=1.5,
                zorder=4, label=f'Baseline ({baseline_accuracy:.1f}%)')

    # Best 표시
    best = results[0]
    ax1.scatter([best['params']], [best['accuracy']], c='gold', s=200,
                marker='D', edgecolors='black', linewidth=1.5,
                zorder=4, label=f'Best ({best["accuracy"]:.1f}%)')

    ax1.set_xlabel('Number of Parameters')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Random Search: Parameters vs Accuracy')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: 탐색 진행 곡선 (best-so-far) ---
    ax2 = axes[1]
    # 원래 탐색 순서에서 best-so-far 계산
    best_so_far = []
    current_best = 0
    original_accs = [r['accuracy'] for r in search_order_results]
    for acc in original_accs:
        current_best = max(current_best, acc)
        best_so_far.append(current_best)

    ax2.plot(range(1, len(best_so_far) + 1), best_so_far, 'b-o',
             markersize=5, linewidth=2, label='Best-so-far')
    ax2.plot(range(1, len(original_accs) + 1), original_accs, 'gray',
             alpha=0.5, linewidth=1, label='Individual')
    ax2.axhline(y=baseline_accuracy, color='red', linestyle='--',
                linewidth=2, label=f'Baseline ({baseline_accuracy:.1f}%)')

    ax2.set_xlabel('Number of Architectures Evaluated')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Random Search: Search Progress')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Random Search 결과 시각화 저장: {save_path}")


# =============================================================================
# 7. 메인 실행
# =============================================================================

def main():
    print("=" * 60)
    print("Part 2: NAS - Random Search")
    print("=" * 60)

    # 데이터 로드
    print("\n[1] CIFAR-10 데이터셋 로드 중...")
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    print(f"    - 학습 데이터: {len(train_loader.dataset):,} 샘플")
    print(f"    - 테스트 데이터: {len(test_loader.dataset):,} 샘플")

    # Baseline 로드 또는 학습
    print("\n[2] Baseline 모델 준비")
    print("-" * 40)

    baseline_path = 'data/trained_models/baseline_model.pth'
    torch.manual_seed(SEED)
    baseline_model = CNN(num_classes=NUM_CLASSES).to(DEVICE)
    baseline_params = count_parameters(baseline_model)

    if os.path.exists(baseline_path):
        baseline_model.load_state_dict(torch.load(baseline_path, map_location=DEVICE, weights_only=True))
        print(f"    Baseline 모델 로드: {baseline_path}")
    else:
        print("    Baseline 모델 학습 중...")
        train(baseline_model, train_loader, FULL_EPOCHS, LEARNING_RATE, DEVICE)
        os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
        torch.save(baseline_model.state_dict(), baseline_path)

    baseline_accuracy = test(baseline_model, test_loader, DEVICE)
    print(f"    Baseline 정확도: {baseline_accuracy:.2f}%")
    print(f"    Baseline 파라미터: {baseline_params:,}")

    # Random Search 실행
    print("\n[3] Random Search 실행")
    print("-" * 40)

    total_start = time.time()
    search_results = random_search(
        num_samples=NUM_SAMPLES,
        train_loader=train_loader,
        test_loader=test_loader,
        search_epochs=SEARCH_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        base_seed=SEED
    )
    search_time = time.time() - total_start

    # Best 아키텍처 선정
    best_result = search_results[0]
    best_arch = best_result['arch']
    print(f"\n    Best 아키텍처 (축소 학습): {architecture_to_string(best_arch)}")
    print(f"    축소 학습 정확도: {best_result['accuracy']:.2f}%")
    print(f"    파라미터 수: {best_result['params']:,}")
    print(f"    총 탐색 시간: {search_time:.1f}s")

    # Best 아키텍처 전체 학습
    print("\n[4] Best 아키텍처 전체 학습 (Full Training)")
    print("-" * 40)
    print(f"    아키텍처: {architecture_to_string(best_arch)}")
    print(f"    전체 학습: {FULL_EPOCHS} epochs")

    torch.manual_seed(SEED)
    best_model = FlexibleCNN(best_arch, num_classes=NUM_CLASSES)
    full_train_losses = train(best_model, train_loader, FULL_EPOCHS, LEARNING_RATE, DEVICE)
    best_full_accuracy = test(best_model, test_loader, DEVICE)
    print(f"    전체 학습 정확도: {best_full_accuracy:.2f}%")

    # 모델 저장
    torch.save(best_model.state_dict(), 'data/trained_models/random_search_best.pth')
    print("    Best 모델 저장: data/trained_models/random_search_best.pth")

    # 결과 저장
    import json
    search_summary = {
        'method': 'random_search',
        'num_samples': NUM_SAMPLES,
        'search_epochs': SEARCH_EPOCHS,
        'full_epochs': FULL_EPOCHS,
        'best_arch': best_arch,
        'best_search_accuracy': best_result['accuracy'],
        'best_full_accuracy': best_full_accuracy,
        'best_params': best_result['params'],
        'search_time': search_time,
        'baseline_accuracy': baseline_accuracy,
        'baseline_params': baseline_params,
        'all_results': [
            {
                'arch': r['arch'],
                'accuracy': r['accuracy'],
                'params': r['params'],
                'train_time': r['train_time']
            }
            for r in search_results
        ]
    }
    os.makedirs('results', exist_ok=True)
    with open('results/random_search_results.json', 'w') as f:
        json.dump(search_summary, f, indent=2)
    print("    결과 저장: results/random_search_results.json")

    # 시각화
    print("\n[5] 결과 시각화")
    print("-" * 40)
    visualize_random_search_results(search_results, baseline_accuracy, baseline_params)

    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약: Random Search vs Baseline")
    print("=" * 60)
    print(f"{'Method':<25} {'Accuracy':<15} {'Parameters':<15}")
    print("-" * 55)
    print(f"{'Baseline (Human)':<25} {baseline_accuracy:>10.2f}% {baseline_params:>12,}")
    print(f"{'Random Search (축소)':<25} {best_result['accuracy']:>10.2f}% {best_result['params']:>12,}")
    print(f"{'Random Search (전체)':<25} {best_full_accuracy:>10.2f}% {best_result['params']:>12,}")
    print("-" * 55)

    acc_diff = best_full_accuracy - baseline_accuracy
    param_ratio = best_result['params'] / baseline_params * 100
    print(f"\n    정확도 차이: {acc_diff:+.2f}%")
    print(f"    파라미터 비율: {param_ratio:.1f}% (Baseline 대비)")
    print(f"    탐색 비용: {NUM_SAMPLES}개 아키텍처 x {SEARCH_EPOCHS} epochs = 총 {search_time:.1f}s")

    return {
        'search_results': search_results,
        'best_arch': best_arch,
        'best_full_accuracy': best_full_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'search_time': search_time
    }


if __name__ == "__main__":
    results = main()
