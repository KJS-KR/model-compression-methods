"""
Part 4: NAS 실험 종합 분석

이 스크립트에서 다루는 내용:
1. Baseline vs Random Search vs Evolutionary Search 종합 비교
2. 정확도 vs 파라미터 수 scatter plot (전체 탐색 결과)
3. 탐색 효율 비교 (평가 횟수 vs best-so-far 정확도)
4. 아키텍처 패턴 분석 (어떤 필터/커널이 효과적인가?)
5. 종합 시각화 및 결과 저장

이전 스크립트의 결과를 로드하여 분석합니다.
결과 파일이 없는 경우 자동으로 실험을 수행합니다.
"""

import sys
import os
import json
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
SEARCH_EPOCHS = 5
FULL_EPOCHS = 10
NUM_SAMPLES = 20
POPULATION_SIZE = 10
NUM_GENERATIONS = 5
TOURNAMENT_SIZE = 3
ELITE_SIZE = 2

SEED = 42
torch.manual_seed(SEED)


# =============================================================================
# 2. 데이터 로드 및 학습/평가 함수 (이전 스크립트와 동일)
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


def train(model, train_loader, epochs, learning_rate, device, verbose=True):
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


def test(model, test_loader, device):
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


def count_parameters(model):
    """모델의 총 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters())


def train_and_evaluate(arch, train_loader, test_loader, epochs, learning_rate,
                        device, seed=42):
    """하나의 아키텍처를 학습하고 평가"""
    torch.manual_seed(seed)
    model = FlexibleCNN(arch, num_classes=NUM_CLASSES)
    params = count_parameters(model)
    start_time = time.time()
    train_losses = train(model, train_loader, epochs, learning_rate, device, verbose=False)
    train_time = time.time() - start_time
    accuracy = test(model, test_loader, device)
    return {
        'arch': arch, 'accuracy': accuracy, 'params': params,
        'train_losses': train_losses, 'train_time': train_time
    }


# =============================================================================
# 3. 결과 로드 또는 실험 실행
# =============================================================================

def load_or_run_experiments(train_loader, test_loader):
    """
    이전 스크립트의 결과를 로드하거나, 없으면 실험을 수행

    Returns:
        dict: {baseline, random_search, evolutionary_search} 결과
    """
    results = {}

    # --- Baseline ---
    print("\n[Baseline]")
    baseline_path = 'data/trained_models/baseline_model.pth'
    torch.manual_seed(SEED)
    baseline_model = CNN(num_classes=NUM_CLASSES).to(DEVICE)
    baseline_params = count_parameters(baseline_model)

    if os.path.exists(baseline_path):
        baseline_model.load_state_dict(torch.load(baseline_path, map_location=DEVICE, weights_only=True))
        print(f"  Baseline 모델 로드: {baseline_path}")
    else:
        print("  Baseline 모델 학습 중...")
        train(baseline_model, train_loader, FULL_EPOCHS, LEARNING_RATE, DEVICE)
        os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
        torch.save(baseline_model.state_dict(), baseline_path)

    baseline_accuracy = test(baseline_model, test_loader, DEVICE)
    print(f"  Baseline 정확도: {baseline_accuracy:.2f}%, 파라미터: {baseline_params:,}")
    results['baseline'] = {
        'accuracy': baseline_accuracy,
        'params': baseline_params
    }

    # --- Random Search ---
    print("\n[Random Search]")
    random_json = 'results/random_search_results.json'

    if os.path.exists(random_json):
        with open(random_json, 'r') as f:
            random_data = json.load(f)
        print(f"  결과 로드: {random_json}")
        results['random_search'] = random_data
    else:
        print("  Random Search 결과 없음. 실험 수행 중...")
        import random as rand_module
        rand_module.seed(SEED)

        random_results = []
        for i in tqdm(range(NUM_SAMPLES), desc="Random Search"):
            arch = sample_architecture(seed=SEED + i)
            result = train_and_evaluate(
                arch, train_loader, test_loader,
                SEARCH_EPOCHS, LEARNING_RATE, DEVICE, seed=SEED
            )
            random_results.append(result)

        random_results.sort(key=lambda x: x['accuracy'], reverse=True)
        best = random_results[0]

        # Best 전체 학습
        torch.manual_seed(SEED)
        best_model = FlexibleCNN(best['arch'], num_classes=NUM_CLASSES)
        train(best_model, train_loader, FULL_EPOCHS, LEARNING_RATE, DEVICE, verbose=False)
        best_full_accuracy = test(best_model, test_loader, DEVICE)

        results['random_search'] = {
            'best_arch': best['arch'],
            'best_search_accuracy': best['accuracy'],
            'best_full_accuracy': best_full_accuracy,
            'best_params': best['params'],
            'num_samples': NUM_SAMPLES,
            'all_results': [
                {'arch': r['arch'], 'accuracy': r['accuracy'],
                 'params': r['params'], 'train_time': r['train_time']}
                for r in random_results
            ]
        }

        os.makedirs('results', exist_ok=True)
        with open(random_json, 'w') as f:
            json.dump(results['random_search'], f, indent=2)
        print(f"  결과 저장: {random_json}")

    # --- Evolutionary Search ---
    print("\n[Evolutionary Search]")
    evo_json = 'results/evolutionary_search_results.json'

    if os.path.exists(evo_json):
        with open(evo_json, 'r') as f:
            evo_data = json.load(f)
        print(f"  결과 로드: {evo_json}")
        results['evolutionary_search'] = evo_data
    else:
        print("  Evolutionary Search 결과 없음. 실험 수행 중...")
        import random as rand_module
        rand_module.seed(SEED)
        import copy

        all_evaluated = []
        generation_history = []

        # 초기 인구
        population = []
        fitnesses = []
        for i in tqdm(range(POPULATION_SIZE), desc="Initial Pop"):
            arch = sample_architecture(seed=SEED + i)
            result = train_and_evaluate(
                arch, train_loader, test_loader,
                SEARCH_EPOCHS, LEARNING_RATE, DEVICE, seed=SEED
            )
            population.append(arch)
            fitnesses.append(result['accuracy'])
            all_evaluated.append(result)

        best_idx = np.argmax(fitnesses)
        generation_history.append({
            'generation': 0,
            'best_accuracy': fitnesses[best_idx],
            'mean_accuracy': float(np.mean(fitnesses)),
            'num_evaluated': len(all_evaluated)
        })

        # 세대별 진화
        for gen in range(1, NUM_GENERATIONS + 1):
            sorted_indices = sorted(range(len(fitnesses)),
                                     key=lambda i: fitnesses[i], reverse=True)
            elite_indices = sorted_indices[:ELITE_SIZE]

            new_population = [copy.deepcopy(population[i]) for i in elite_indices]
            new_fitnesses = [fitnesses[i] for i in elite_indices]

            num_children = POPULATION_SIZE - ELITE_SIZE
            for _ in tqdm(range(num_children), desc=f"Gen {gen}"):
                indices = rand_module.sample(range(len(population)),
                                             min(TOURNAMENT_SIZE, len(population)))
                parent_idx = max(indices, key=lambda i: fitnesses[i])
                parent = copy.deepcopy(population[parent_idx])
                child = mutate_architecture(parent)

                result = train_and_evaluate(
                    child, train_loader, test_loader,
                    SEARCH_EPOCHS, LEARNING_RATE, DEVICE, seed=SEED
                )
                new_population.append(child)
                new_fitnesses.append(result['accuracy'])
                all_evaluated.append(result)

            population = new_population
            fitnesses = new_fitnesses

            best_idx = np.argmax(fitnesses)
            generation_history.append({
                'generation': gen,
                'best_accuracy': fitnesses[best_idx],
                'mean_accuracy': float(np.mean(fitnesses)),
                'num_evaluated': len(all_evaluated)
            })

        overall_best = max(all_evaluated, key=lambda x: x['accuracy'])

        # Best 전체 학습
        torch.manual_seed(SEED)
        best_model = FlexibleCNN(overall_best['arch'], num_classes=NUM_CLASSES)
        train(best_model, train_loader, FULL_EPOCHS, LEARNING_RATE, DEVICE, verbose=False)
        best_full_accuracy = test(best_model, test_loader, DEVICE)

        results['evolutionary_search'] = {
            'best_arch': overall_best['arch'],
            'best_search_accuracy': overall_best['accuracy'],
            'best_full_accuracy': best_full_accuracy,
            'best_params': overall_best['params'],
            'total_evaluations': len(all_evaluated),
            'generation_history': generation_history,
            'all_results': [
                {'arch': r['arch'], 'accuracy': r['accuracy'],
                 'params': r['params'], 'train_time': r['train_time']}
                for r in all_evaluated
            ]
        }

        os.makedirs('results', exist_ok=True)
        with open(evo_json, 'w') as f:
            json.dump(results['evolutionary_search'], f, indent=2)
        print(f"  결과 저장: {evo_json}")

    return results


# =============================================================================
# 4. 종합 비교 시각화
# =============================================================================

def visualize_comprehensive_comparison(results: dict,
                                        save_path: str = 'results/comprehensive_comparison.png'):
    """
    Baseline vs Random Search vs Evolutionary Search 종합 비교

    4개의 subplot:
    1. 방법별 정확도 비교 (bar chart)
    2. 전체 탐색 결과 scatter (params vs accuracy)
    3. 탐색 효율 비교 (evaluation count vs best-so-far)
    4. 방법별 파라미터 수 비교
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    baseline = results['baseline']
    random_search = results['random_search']
    evo_search = results['evolutionary_search']

    # --- Plot 1: 방법별 정확도 비교 ---
    ax1 = axes[0][0]
    methods = ['Baseline\n(Human)', 'Random\nSearch', 'Evolutionary\nSearch']
    accuracies = [
        baseline['accuracy'],
        random_search['best_full_accuracy'],
        evo_search['best_full_accuracy']
    ]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    bars = ax1.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=0.5)

    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Method Comparison: Accuracy')
    ax1.set_ylim(min(accuracies) - 5, max(accuracies) + 3)
    ax1.grid(True, alpha=0.3, axis='y')

    # --- Plot 2: 전체 탐색 결과 scatter ---
    ax2 = axes[0][1]

    # Random Search 결과
    rand_accs = [r['accuracy'] for r in random_search['all_results']]
    rand_params = [r['params'] for r in random_search['all_results']]
    ax2.scatter(rand_params, rand_accs, c='#3498db', s=50, alpha=0.7,
                edgecolors='black', linewidth=0.3, label='Random Search', zorder=3)

    # Evolutionary Search 결과
    evo_accs = [r['accuracy'] for r in evo_search['all_results']]
    evo_params = [r['params'] for r in evo_search['all_results']]
    ax2.scatter(evo_params, evo_accs, c='#2ecc71', s=50, alpha=0.7,
                edgecolors='black', linewidth=0.3, marker='s',
                label='Evolutionary Search', zorder=3)

    # Baseline
    ax2.scatter([baseline['params']], [baseline['accuracy']], c='red', s=200,
                marker='*', edgecolors='black', linewidth=1.5,
                zorder=4, label=f'Baseline')

    ax2.set_xlabel('Number of Parameters')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('All Evaluated Architectures')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: 탐색 효율 비교 ---
    ax3 = axes[1][0]

    # Random Search: best-so-far
    rand_best_so_far = []
    current_best = 0
    for r in random_search['all_results']:
        current_best = max(current_best, r['accuracy'])
        rand_best_so_far.append(current_best)

    ax3.plot(range(1, len(rand_best_so_far) + 1), rand_best_so_far,
             'b-o', markersize=4, linewidth=2, label='Random Search')

    # Evolutionary Search: best-so-far
    evo_best_so_far = []
    current_best = 0
    for r in evo_search['all_results']:
        current_best = max(current_best, r['accuracy'])
        evo_best_so_far.append(current_best)

    ax3.plot(range(1, len(evo_best_so_far) + 1), evo_best_so_far,
             'g-s', markersize=4, linewidth=2, label='Evolutionary Search')

    ax3.axhline(y=baseline['accuracy'], color='red', linestyle='--',
                linewidth=2, label=f'Baseline ({baseline["accuracy"]:.1f}%)')

    ax3.set_xlabel('Number of Architectures Evaluated')
    ax3.set_ylabel('Best-so-far Accuracy (%)')
    ax3.set_title('Search Efficiency Comparison')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: 방법별 파라미터 수 비교 ---
    ax4 = axes[1][1]
    params_list = [
        baseline['params'],
        random_search['best_params'],
        evo_search['best_params']
    ]
    bars = ax4.bar(methods, params_list, color=colors, edgecolor='black', linewidth=0.5)

    for bar, params in zip(bars, params_list):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(params_list) * 0.01,
                 f'{params:,}', ha='center', va='bottom', fontsize=9)

    ax4.set_ylabel('Number of Parameters')
    ax4.set_title('Method Comparison: Model Size')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"종합 비교 시각화 저장: {save_path}")


# =============================================================================
# 5. 아키텍처 패턴 분석
# =============================================================================

def analyze_architecture_patterns(results: dict,
                                   save_path: str = 'results/architecture_patterns.png'):
    """
    평가된 아키텍처들에서 패턴 분석

    - 상위/하위 아키텍처에서 많이 사용된 필터 수, 커널 크기
    - 어떤 구성이 높은 정확도와 상관관계가 있는지
    """
    # 모든 평가 결과 합치기
    all_results = []
    for r in results['random_search']['all_results']:
        all_results.append(r)
    for r in results['evolutionary_search']['all_results']:
        all_results.append(r)

    # 정확도 기준 정렬
    all_results.sort(key=lambda x: x['accuracy'], reverse=True)

    # 상위 25%와 하위 25% 구분
    n = len(all_results)
    top_quarter = all_results[:n // 4]
    bottom_quarter = all_results[-(n // 4):]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: 레이어별 필터 수 분포 (Top vs Bottom) ---
    ax1 = axes[0]
    num_conv = SEARCH_SPACE['num_conv_layers']
    x_pos = np.arange(num_conv)
    width = 0.35

    top_mean_filters = [np.mean([r['arch']['filters'][i] for r in top_quarter])
                         for i in range(num_conv)]
    bottom_mean_filters = [np.mean([r['arch']['filters'][i] for r in bottom_quarter])
                            for i in range(num_conv)]

    ax1.bar(x_pos - width / 2, top_mean_filters, width, color='#2ecc71',
            edgecolor='black', linewidth=0.5, label=f'Top 25% (n={len(top_quarter)})')
    ax1.bar(x_pos + width / 2, bottom_mean_filters, width, color='#e74c3c',
            edgecolor='black', linewidth=0.5, label=f'Bottom 25% (n={len(bottom_quarter)})')

    ax1.set_xlabel('Conv Layer')
    ax1.set_ylabel('Mean Filter Count')
    ax1.set_title('Filter Count: Top vs Bottom Architectures')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Conv {i+1}' for i in range(num_conv)])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # --- Plot 2: 커널 크기 선호도 (Top vs Bottom) ---
    ax2 = axes[1]

    # 각 레이어에서 kernel=3, kernel=5 비율
    for group, label, color in [(top_quarter, 'Top 25%', '#2ecc71'),
                                  (bottom_quarter, 'Bottom 25%', '#e74c3c')]:
        k3_ratios = []
        for i in range(num_conv):
            k3_count = sum(1 for r in group if r['arch']['kernel_sizes'][i] == 3)
            k3_ratios.append(k3_count / len(group) * 100)

        ax2.plot(range(num_conv), k3_ratios, 'o-', color=color,
                 markersize=8, linewidth=2, label=f'{label}')

    ax2.set_xlabel('Conv Layer')
    ax2.set_ylabel('Kernel Size = 3 Ratio (%)')
    ax2.set_title('Kernel Size Preference: Top vs Bottom')
    ax2.set_xticks(range(num_conv))
    ax2.set_xticklabels([f'Conv {i+1}' for i in range(num_conv)])
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: FC Hidden Units 분포 ---
    ax3 = axes[2]
    fc_options = SEARCH_SPACE['fc_hidden']

    top_fc_counts = [sum(1 for r in top_quarter if r['arch']['fc_hidden'] == fc)
                      for fc in fc_options]
    bottom_fc_counts = [sum(1 for r in bottom_quarter if r['arch']['fc_hidden'] == fc)
                         for fc in fc_options]

    x_pos = np.arange(len(fc_options))
    ax3.bar(x_pos - width / 2, top_fc_counts, width, color='#2ecc71',
            edgecolor='black', linewidth=0.5, label=f'Top 25%')
    ax3.bar(x_pos + width / 2, bottom_fc_counts, width, color='#e74c3c',
            edgecolor='black', linewidth=0.5, label=f'Bottom 25%')

    ax3.set_xlabel('FC Hidden Units')
    ax3.set_ylabel('Count')
    ax3.set_title('FC Hidden Units Distribution')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([str(fc) for fc in fc_options])
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"아키텍처 패턴 분석 시각화 저장: {save_path}")


# =============================================================================
# 6. 메인 실행
# =============================================================================

def main():
    print("=" * 60)
    print("Part 4: NAS - 종합 분석")
    print("=" * 60)

    # 데이터 로드
    print("\n[1] CIFAR-10 데이터셋 로드 중...")
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    print(f"    - 학습 데이터: {len(train_loader.dataset):,} 샘플")
    print(f"    - 테스트 데이터: {len(test_loader.dataset):,} 샘플")

    # 결과 로드 또는 실험 수행
    print("\n[2] 실험 결과 로드")
    print("-" * 40)
    results = load_or_run_experiments(train_loader, test_loader)

    # 종합 비교 시각화
    print("\n[3] 종합 비교 시각화")
    print("-" * 40)
    visualize_comprehensive_comparison(results)

    # 아키텍처 패턴 분석
    print("\n[4] 아키텍처 패턴 분석")
    print("-" * 40)
    analyze_architecture_patterns(results)

    # 종합 결과 테이블
    print("\n" + "=" * 70)
    print("종합 결과 비교")
    print("=" * 70)

    baseline = results['baseline']
    random_search = results['random_search']
    evo_search = results['evolutionary_search']

    print(f"\n{'Method':<25} {'Accuracy':<12} {'Params':<15} {'탐색 비용':<15}")
    print("-" * 67)
    print(f"{'Baseline (Human)':<25} {baseline['accuracy']:>8.2f}% {baseline['params']:>12,} {'N/A':>12}")

    rand_evals = len(random_search['all_results'])
    print(f"{'Random Search':<25} {random_search['best_full_accuracy']:>8.2f}% "
          f"{random_search['best_params']:>12,} {str(rand_evals) + '개':>12}")

    evo_evals = len(evo_search['all_results'])
    print(f"{'Evolutionary Search':<25} {evo_search['best_full_accuracy']:>8.2f}% "
          f"{evo_search['best_params']:>12,} {str(evo_evals) + '개':>12}")
    print("-" * 67)

    # Best 아키텍처 상세
    print(f"\n[Best 아키텍처 상세]")
    print(f"  Random Search:       {architecture_to_string(random_search['best_arch'])}")
    print(f"  Evolutionary Search: {architecture_to_string(evo_search['best_arch'])}")

    # 탐색 효율 분석
    print(f"\n[탐색 효율 분석]")
    rand_acc = random_search['best_full_accuracy']
    evo_acc = evo_search['best_full_accuracy']
    print(f"  Random Search:       {rand_evals}개 평가 → {rand_acc:.2f}%")
    print(f"  Evolutionary Search: {evo_evals}개 평가 → {evo_acc:.2f}%")

    if evo_evals > 0 and rand_evals > 0:
        rand_efficiency = rand_acc / rand_evals
        evo_efficiency = evo_acc / evo_evals
        print(f"\n  평가당 정확도:")
        print(f"    Random:       {rand_efficiency:.3f}%/eval")
        print(f"    Evolutionary: {evo_efficiency:.3f}%/eval")

    # 결론
    print(f"\n[결론]")
    print(f"  1. NAS는 사람이 설계한 Baseline과 비교하여")
    if max(rand_acc, evo_acc) > baseline['accuracy']:
        print(f"     자동으로 더 좋은 아키텍처를 찾을 수 있었습니다.")
    else:
        print(f"     비슷한 수준의 아키텍처를 자동으로 찾을 수 있었습니다.")
    print(f"  2. Random Search는 간단하지만 놀라울 정도로 효과적입니다.")
    print(f"  3. Evolutionary Search는 좋은 아키텍처 '근처'를 집중 탐색하여")
    print(f"     탐색 효율을 높일 수 있습니다.")
    print(f"  4. 탐색 공간 설계가 NAS 성능에 매우 중요합니다.")

    # 전체 결과 JSON 저장
    summary = {
        'baseline': baseline,
        'random_search': {
            'best_arch': random_search['best_arch'],
            'best_full_accuracy': random_search['best_full_accuracy'],
            'best_params': random_search['best_params'],
            'num_evaluations': rand_evals
        },
        'evolutionary_search': {
            'best_arch': evo_search['best_arch'],
            'best_full_accuracy': evo_search['best_full_accuracy'],
            'best_params': evo_search['best_params'],
            'num_evaluations': evo_evals
        }
    }
    with open('results/experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  종합 결과 저장: results/experiment_summary.json")

    return results


if __name__ == "__main__":
    results = main()
