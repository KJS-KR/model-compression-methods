"""
Part 3: Evolutionary Search NAS

이 스크립트에서 다루는 내용:
1. Evolutionary Search 전략의 원리
2. Tournament Selection + Mutation 기반 진화적 탐색
3. 세대별 진화 과정 추적
4. Random Search와 비교 분석

Evolutionary NAS 핵심 아이디어:
- 유전 알고리즘(Genetic Algorithm)을 아키텍처 탐색에 적용
- 좋은 아키텍처를 "부모"로 선택하여 "자식" 아키텍처를 생성
- 여러 세대에 걸쳐 아키텍처가 점진적으로 개선

진화 과정:
==========
1. 초기 인구(Population) 생성 및 평가
2. 각 세대(Generation)마다:
   a. Tournament Selection: k개 중 가장 좋은 아키텍처 선택
   b. Mutation: 선택된 아키텍처의 파라미터 하나를 랜덤 변경
   c. 자식 아키텍처 학습 및 평가
   d. 엘리트(상위 N개) 유지 + 새 자식으로 인구 교체

Evolutionary vs Random Search:
- Random: 탐색 이력을 활용하지 않음 (무작위)
- Evolutionary: 좋은 아키텍처 "근처"를 집중 탐색 (exploitation)
- 적은 평가 횟수로 더 좋은 아키텍처를 찾을 수 있음

참고: Real et al., 2019 - Regularized Evolution for Image Classifier Architecture Search
"""

import sys
import os
import time
import random
import copy

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
    sample_architecture, architecture_to_string, mutate_architecture
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
POPULATION_SIZE = 10    # 인구 크기
NUM_GENERATIONS = 5     # 진화 세대 수
TOURNAMENT_SIZE = 3     # 토너먼트 선택 크기
ELITE_SIZE = 2          # 엘리트 유지 수

# 재현성을 위한 시드 설정
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)


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


def train_and_evaluate(
    arch: dict,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    seed: int = 42
) -> dict:
    """하나의 아키텍처를 학습하고 평가"""
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
# 4. Evolutionary Search 핵심 함수
# =============================================================================

def tournament_selection(population: list[dict], fitnesses: list[float],
                          tournament_size: int = 3) -> dict:
    """
    토너먼트 선택 (Tournament Selection)

    인구에서 k개를 무작위로 선택한 후, 그 중 가장 적합도가 높은 개체를 선택

    Args:
        population: 아키텍처 리스트
        fitnesses: 각 아키텍처의 적합도 (정확도)
        tournament_size: 토너먼트 크기 (k)

    Returns:
        선택된 아키텍처
    """
    # 인구에서 k개 랜덤 선택
    indices = random.sample(range(len(population)), min(tournament_size, len(population)))

    # 선택된 후보 중 가장 적합도가 높은 개체 선택
    best_idx = max(indices, key=lambda i: fitnesses[i])

    return copy.deepcopy(population[best_idx])


def evolutionary_search(
    train_loader: DataLoader,
    test_loader: DataLoader,
    population_size: int,
    num_generations: int,
    tournament_size: int,
    elite_size: int,
    search_epochs: int,
    learning_rate: float,
    device: str,
    base_seed: int = 42
) -> dict:
    """
    Evolutionary Search NAS

    Args:
        train_loader: 학습 데이터 로더
        test_loader: 테스트 데이터 로더
        population_size: 인구 크기
        num_generations: 진화 세대 수
        tournament_size: 토너먼트 크기
        elite_size: 엘리트 유지 수
        search_epochs: 축소 학습 에포크
        learning_rate: 학습률
        device: 디바이스
        base_seed: 기본 시드

    Returns:
        dict: {best_arch, best_accuracy, generation_history, all_evaluated}
    """
    all_evaluated = []          # 평가된 모든 아키텍처
    generation_history = []     # 세대별 통계

    # --- 1. 초기 인구 생성 및 평가 ---
    print(f"\n[초기 인구 생성] {population_size}개 아키텍처")
    print("-" * 60)

    population = []
    fitnesses = []

    for i in tqdm(range(population_size), desc="Initial Population"):
        arch = sample_architecture(seed=base_seed + i)
        result = train_and_evaluate(
            arch, train_loader, test_loader,
            search_epochs, learning_rate, device, seed=base_seed
        )
        population.append(arch)
        fitnesses.append(result['accuracy'])
        all_evaluated.append(result)

        print(f"  [{i+1}/{population_size}] {architecture_to_string(arch)} | "
              f"Acc: {result['accuracy']:.2f}% | Params: {result['params']:,}")

    # 초기 세대 통계
    best_idx = np.argmax(fitnesses)
    gen_stats = {
        'generation': 0,
        'best_accuracy': fitnesses[best_idx],
        'mean_accuracy': np.mean(fitnesses),
        'best_arch': architecture_to_string(population[best_idx]),
        'num_evaluated': len(all_evaluated)
    }
    generation_history.append(gen_stats)
    print(f"\n  세대 0 | Best: {gen_stats['best_accuracy']:.2f}% | "
          f"Mean: {gen_stats['mean_accuracy']:.2f}%")

    # --- 2. 세대별 진화 ---
    for gen in range(1, num_generations + 1):
        print(f"\n[세대 {gen}/{num_generations}]")
        print("-" * 60)

        # 엘리트 선택: 상위 elite_size개 유지
        sorted_indices = sorted(range(len(fitnesses)),
                                 key=lambda i: fitnesses[i], reverse=True)
        elite_indices = sorted_indices[:elite_size]

        new_population = [copy.deepcopy(population[i]) for i in elite_indices]
        new_fitnesses = [fitnesses[i] for i in elite_indices]

        print(f"  엘리트 {elite_size}개 유지:")
        for idx in elite_indices:
            print(f"    - {architecture_to_string(population[idx])} | "
                  f"Acc: {fitnesses[idx]:.2f}%")

        # 나머지는 토너먼트 선택 + 돌연변이로 생성
        num_children = population_size - elite_size
        print(f"\n  자식 {num_children}개 생성 (Tournament k={tournament_size} + Mutation):")

        for i in tqdm(range(num_children), desc=f"Gen {gen}"):
            # 토너먼트 선택
            parent = tournament_selection(population, fitnesses, tournament_size)

            # 돌연변이
            child = mutate_architecture(parent)

            # 자식 평가
            result = train_and_evaluate(
                child, train_loader, test_loader,
                search_epochs, learning_rate, device, seed=base_seed
            )
            new_population.append(child)
            new_fitnesses.append(result['accuracy'])
            all_evaluated.append(result)

            print(f"    [{i+1}/{num_children}] "
                  f"Parent: {architecture_to_string(parent)} -> "
                  f"Child: {architecture_to_string(child)} | "
                  f"Acc: {result['accuracy']:.2f}%")

        # 인구 교체
        population = new_population
        fitnesses = new_fitnesses

        # 세대 통계
        best_idx = np.argmax(fitnesses)
        gen_stats = {
            'generation': gen,
            'best_accuracy': fitnesses[best_idx],
            'mean_accuracy': np.mean(fitnesses),
            'best_arch': architecture_to_string(population[best_idx]),
            'num_evaluated': len(all_evaluated)
        }
        generation_history.append(gen_stats)

        print(f"\n  세대 {gen} 결과 | Best: {gen_stats['best_accuracy']:.2f}% | "
              f"Mean: {gen_stats['mean_accuracy']:.2f}% | "
              f"총 평가: {gen_stats['num_evaluated']}개")

    # --- 3. 최종 결과 ---
    overall_best = max(all_evaluated, key=lambda x: x['accuracy'])

    return {
        'best_result': overall_best,
        'generation_history': generation_history,
        'all_evaluated': all_evaluated,
        'final_population': population,
        'final_fitnesses': fitnesses
    }


# =============================================================================
# 5. 시각화
# =============================================================================

def visualize_evolutionary_results(evo_result: dict, baseline_accuracy: float,
                                    baseline_params: int,
                                    save_path: str = 'results/evolutionary_search_results.png'):
    """
    Evolutionary Search 결과 시각화

    1. 세대별 최고/평균 정확도 곡선
    2. 파라미터 수 vs 정확도 scatter plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    gen_history = evo_result['generation_history']
    all_evaluated = evo_result['all_evaluated']

    # --- Plot 1: 세대별 정확도 추이 ---
    ax1 = axes[0]
    generations = [g['generation'] for g in gen_history]
    best_accs = [g['best_accuracy'] for g in gen_history]
    mean_accs = [g['mean_accuracy'] for g in gen_history]

    ax1.plot(generations, best_accs, 'b-o', markersize=8, linewidth=2, label='Best in Population')
    ax1.plot(generations, mean_accs, 'g-s', markersize=6, linewidth=1.5, label='Mean of Population')
    ax1.axhline(y=baseline_accuracy, color='red', linestyle='--',
                linewidth=2, label=f'Baseline ({baseline_accuracy:.1f}%)')

    ax1.fill_between(generations, mean_accs, best_accs, alpha=0.15, color='blue')

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Evolutionary Search: Generation Progress')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(generations)

    # --- Plot 2: 전체 평가된 아키텍처 scatter ---
    ax2 = axes[1]
    accs = [r['accuracy'] for r in all_evaluated]
    params = [r['params'] for r in all_evaluated]

    scatter = ax2.scatter(params, accs, c=range(len(all_evaluated)),
                          cmap='plasma', s=60, edgecolors='black', linewidth=0.5,
                          alpha=0.8, zorder=3)
    plt.colorbar(scatter, ax=ax2, label='Evaluation Order')

    # Baseline 표시
    ax2.scatter([baseline_params], [baseline_accuracy], c='red', s=200,
                marker='*', edgecolors='black', linewidth=1.5,
                zorder=4, label=f'Baseline ({baseline_accuracy:.1f}%)')

    # Best 표시
    best = evo_result['best_result']
    ax2.scatter([best['params']], [best['accuracy']], c='gold', s=200,
                marker='D', edgecolors='black', linewidth=1.5,
                zorder=4, label=f'Best ({best["accuracy"]:.1f}%)')

    ax2.set_xlabel('Number of Parameters')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Evolutionary Search: Parameters vs Accuracy')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Evolutionary Search 결과 시각화 저장: {save_path}")


# =============================================================================
# 6. 메인 실행
# =============================================================================

def main():
    print("=" * 60)
    print("Part 3: NAS - Evolutionary Search")
    print("=" * 60)

    # Evolutionary Search 설정 출력
    print(f"\n[설정]")
    print(f"    인구 크기: {POPULATION_SIZE}")
    print(f"    세대 수: {NUM_GENERATIONS}")
    print(f"    토너먼트 크기: {TOURNAMENT_SIZE}")
    print(f"    엘리트 유지: {ELITE_SIZE}")
    print(f"    축소 학습: {SEARCH_EPOCHS} epochs")
    num_children_per_gen = POPULATION_SIZE - ELITE_SIZE
    total_evaluations = POPULATION_SIZE + NUM_GENERATIONS * num_children_per_gen
    print(f"    예상 총 평가 횟수: {POPULATION_SIZE} + {NUM_GENERATIONS} x {num_children_per_gen} = {total_evaluations}")

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

    # Evolutionary Search 실행
    print("\n[3] Evolutionary Search 실행")
    print("-" * 40)

    total_start = time.time()
    evo_result = evolutionary_search(
        train_loader=train_loader,
        test_loader=test_loader,
        population_size=POPULATION_SIZE,
        num_generations=NUM_GENERATIONS,
        tournament_size=TOURNAMENT_SIZE,
        elite_size=ELITE_SIZE,
        search_epochs=SEARCH_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        base_seed=SEED
    )
    search_time = time.time() - total_start

    best_result = evo_result['best_result']
    best_arch = best_result['arch']
    print(f"\n    Best 아키텍처 (축소 학습): {architecture_to_string(best_arch)}")
    print(f"    축소 학습 정확도: {best_result['accuracy']:.2f}%")
    print(f"    파라미터 수: {best_result['params']:,}")
    print(f"    총 탐색 시간: {search_time:.1f}s")
    print(f"    총 평가 횟수: {len(evo_result['all_evaluated'])}개")

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
    torch.save(best_model.state_dict(), 'data/trained_models/evolutionary_search_best.pth')
    print("    Best 모델 저장: data/trained_models/evolutionary_search_best.pth")

    # 결과 저장
    import json
    search_summary = {
        'method': 'evolutionary_search',
        'population_size': POPULATION_SIZE,
        'num_generations': NUM_GENERATIONS,
        'tournament_size': TOURNAMENT_SIZE,
        'elite_size': ELITE_SIZE,
        'search_epochs': SEARCH_EPOCHS,
        'full_epochs': FULL_EPOCHS,
        'best_arch': best_arch,
        'best_search_accuracy': best_result['accuracy'],
        'best_full_accuracy': best_full_accuracy,
        'best_params': best_result['params'],
        'search_time': search_time,
        'total_evaluations': len(evo_result['all_evaluated']),
        'baseline_accuracy': baseline_accuracy,
        'baseline_params': baseline_params,
        'generation_history': evo_result['generation_history'],
        'all_results': [
            {
                'arch': r['arch'],
                'accuracy': r['accuracy'],
                'params': r['params'],
                'train_time': r['train_time']
            }
            for r in evo_result['all_evaluated']
        ]
    }
    os.makedirs('results', exist_ok=True)
    with open('results/evolutionary_search_results.json', 'w') as f:
        json.dump(search_summary, f, indent=2)
    print("    결과 저장: results/evolutionary_search_results.json")

    # 시각화
    print("\n[5] 결과 시각화")
    print("-" * 40)
    visualize_evolutionary_results(evo_result, baseline_accuracy, baseline_params)

    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약: Evolutionary Search vs Baseline")
    print("=" * 60)
    print(f"{'Method':<25} {'Accuracy':<15} {'Parameters':<15}")
    print("-" * 55)
    print(f"{'Baseline (Human)':<25} {baseline_accuracy:>10.2f}% {baseline_params:>12,}")
    print(f"{'Evolutionary (축소)':<25} {best_result['accuracy']:>10.2f}% {best_result['params']:>12,}")
    print(f"{'Evolutionary (전체)':<25} {best_full_accuracy:>10.2f}% {best_result['params']:>12,}")
    print("-" * 55)

    acc_diff = best_full_accuracy - baseline_accuracy
    param_ratio = best_result['params'] / baseline_params * 100
    print(f"\n    정확도 차이: {acc_diff:+.2f}%")
    print(f"    파라미터 비율: {param_ratio:.1f}% (Baseline 대비)")
    print(f"    탐색 비용: {len(evo_result['all_evaluated'])}개 아키텍처 x {SEARCH_EPOCHS} epochs")
    print(f"    탐색 시간: {search_time:.1f}s")

    # 세대별 진화 요약
    print(f"\n[세대별 진화 요약]")
    print(f"{'세대':<8} {'Best Acc':<12} {'Mean Acc':<12} {'총 평가':<10}")
    print("-" * 42)
    for g in evo_result['generation_history']:
        print(f"{g['generation']:<8} {g['best_accuracy']:>8.2f}% {g['mean_accuracy']:>9.2f}% {g['num_evaluated']:>8}")

    return {
        'evo_result': evo_result,
        'best_arch': best_arch,
        'best_full_accuracy': best_full_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'search_time': search_time
    }


if __name__ == "__main__":
    results = main()
