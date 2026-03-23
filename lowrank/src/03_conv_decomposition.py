"""
Part 3: Conv 레이어 채널 분해

이 스크립트에서 다루는 내용:
1. 채널 분해(Channel Decomposition) 원리
2. Conv2d 레이어에 다양한 rank 적용
3. Fine-tuning 후 정확도 회복
4. Combined: Linear SVD + Conv 분해 동시 적용
5. 분해 전/후 Feature Map 비교

채널 분해 원리:
================
Conv2d(in_c, out_c, k×k):
  W ∈ ℝ^(out_c × in_c × k × k), 파라미터: out_c × in_c × k × k

채널 분해:
  Conv2d(in_c, rank, 1×1) → Conv2d(rank, out_c, k×k)
  파라미터: (in_c × rank) + (rank × out_c × k × k)

features.2 예시 (Conv2d(128, 64, 3×3)):
  원본:  128 × 64 × 3 × 3 = 73,728
  rank=32: (128 × 32) + (32 × 64 × 3 × 3) = 4,096 + 18,432 = 22,528 (30.6%)
  rank=16: (128 × 16) + (16 × 64 × 3 × 3) = 2,048 +  9,216 = 11,264 (15.3%)
  rank=8:  (128 ×  8) + ( 8 × 64 × 3 × 3) = 1,024 +  4,608 =  5,632 ( 7.6%)
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
    CNN,
    decompose_model_linear, decompose_model_conv,
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
CONV_RANKS = [32, 16, 8]

# Combined 설정
COMBINED_LINEAR_RANK = 128
COMBINED_CONV_RANK = 32

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
# 4. Conv 분해 실험
# =============================================================================

def experiment_conv_ranks(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    ranks: list[int],
    device: str
) -> list[dict]:
    """
    다양한 rank로 Conv 레이어를 채널 분해하고 fine-tuning 후 평가

    분해 대상: features.2 (Conv2d(128, 64, 3×3)) - Conv 중 가장 큰 레이어

    Returns:
        각 rank별 결과 리스트
    """
    baseline_params = count_parameters(model)
    baseline_acc = test(model, test_loader, device)
    print(f"\n  Baseline: {baseline_acc:.2f}% ({baseline_params:,} params)")

    results = []

    for rank in ranks:
        print(f"\n  --- Conv Rank = {rank} ---")

        # 채널 분해
        decomposed_model = decompose_model_conv(
            model, {'features.2': rank}
        )
        decomposed_params = count_parameters(decomposed_model)
        compression_ratio = baseline_params / decomposed_params

        # 분해 직후 정확도
        acc_before_ft = test(decomposed_model, test_loader, device)
        print(f"  분해 직후: {acc_before_ft:.2f}%")
        print(f"  파라미터: {decomposed_params:,} ({decomposed_params/baseline_params*100:.1f}%)")
        print(f"  압축률: {compression_ratio:.2f}x")

        # Fine-tuning
        print(f"  Fine-tuning ({FINETUNE_EPOCHS} epochs)...")
        ft_losses = fine_tune(
            decomposed_model, train_loader,
            FINETUNE_EPOCHS, FINETUNE_LR, device
        )

        # Fine-tuning 후 정확도
        acc_after_ft = test(decomposed_model, test_loader, device)
        print(f"  Fine-tuning 후: {acc_after_ft:.2f}%")

        # 복원 오차
        original_weight = model.features[2].weight.data.cpu()
        out_c, in_c, k_h, k_w = original_weight.shape
        W_reshaped = original_weight.reshape(out_c * k_h * k_w, in_c)
        recon_error = get_reconstruction_error(W_reshaped, rank)

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


def experiment_multi_conv(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    rank: int,
    device: str
) -> dict:
    """
    여러 Conv 레이어를 동시에 분해

    features.2 (128→64, 3×3)와 features.5 (64→64, 3×3)를 함께 분해

    Returns:
        다중 Conv 분해 결과
    """
    baseline_params = count_parameters(model)

    print(f"\n  여러 Conv 레이어 동시 분해 (rank={rank})")

    # 단일 Conv 분해 (features.2만)
    single_model = decompose_model_conv(model, {'features.2': rank})
    single_params = count_parameters(single_model)
    single_acc_before = test(single_model, test_loader, device)
    fine_tune(single_model, train_loader, FINETUNE_EPOCHS, FINETUNE_LR, device)
    single_acc_after = test(single_model, test_loader, device)

    # 다중 Conv 분해 (features.2 + features.5)
    multi_model = decompose_model_conv(
        model, {'features.2': rank, 'features.5': rank}
    )
    multi_params = count_parameters(multi_model)
    multi_acc_before = test(multi_model, test_loader, device)
    fine_tune(multi_model, train_loader, FINETUNE_EPOCHS, FINETUNE_LR, device)
    multi_acc_after = test(multi_model, test_loader, device)

    print(f"  단일(features.2): {single_acc_after:.2f}% ({single_params:,} params)")
    print(f"  다중(features.2+5): {multi_acc_after:.2f}% ({multi_params:,} params)")

    return {
        'rank': rank,
        'single': {
            'params': single_params,
            'acc_before_ft': single_acc_before,
            'acc_after_ft': single_acc_after,
            'compression_ratio': baseline_params / single_params
        },
        'multi': {
            'params': multi_params,
            'acc_before_ft': multi_acc_before,
            'acc_after_ft': multi_acc_after,
            'compression_ratio': baseline_params / multi_params
        }
    }


def experiment_combined(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    linear_rank: int,
    conv_rank: int,
    device: str
) -> dict:
    """
    Linear SVD + Conv 분해를 동시 적용

    classifier.0 → SVDLinear(rank=linear_rank)
    features.2 → ChannelDecomposedConv2d(rank=conv_rank)

    Returns:
        Combined 분해 결과
    """
    baseline_params = count_parameters(model)

    print(f"\n  Combined 분해: Linear(rank={linear_rank}) + Conv(rank={conv_rank})")

    # Linear + Conv 분해
    combined_model = decompose_model_linear(model, {'classifier.0': linear_rank})
    combined_model = decompose_model_conv(combined_model, {'features.2': conv_rank})

    combined_params = count_parameters(combined_model)
    compression_ratio = baseline_params / combined_params

    # 분해 직후
    acc_before_ft = test(combined_model, test_loader, device)
    print(f"  분해 직후: {acc_before_ft:.2f}%")
    print(f"  파라미터: {combined_params:,} ({combined_params/baseline_params*100:.1f}%)")
    print(f"  압축률: {compression_ratio:.2f}x")

    # Fine-tuning
    print(f"  Fine-tuning ({FINETUNE_EPOCHS} epochs)...")
    ft_losses = fine_tune(
        combined_model, train_loader,
        FINETUNE_EPOCHS, FINETUNE_LR, device
    )

    acc_after_ft = test(combined_model, test_loader, device)
    print(f"  Fine-tuning 후: {acc_after_ft:.2f}%")

    return {
        'linear_rank': linear_rank,
        'conv_rank': conv_rank,
        'params': combined_params,
        'param_ratio': combined_params / baseline_params,
        'compression_ratio': compression_ratio,
        'acc_before_ft': acc_before_ft,
        'acc_after_ft': acc_after_ft,
        'ft_losses': ft_losses
    }


# =============================================================================
# 5. 시각화
# =============================================================================

def plot_conv_tradeoff(
    results: list[dict],
    baseline_acc: float,
    save_path: str = 'results/conv_decomposition_tradeoff.png'
):
    """Conv 분해 Rank vs 정확도/파라미터 trade-off"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ranks = [r['rank'] for r in results]
    accs_before = [r['acc_before_ft'] for r in results]
    accs_after = [r['acc_after_ft'] for r in results]
    param_ratios = [r['param_ratio'] * 100 for r in results]

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
    ax1.set_title('Conv Decomposition: Rank vs Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) Fine-tuning loss 곡선
    ax2 = axes[1]
    for r in results:
        epochs = range(1, len(r['ft_losses']) + 1)
        ax2.plot(epochs, r['ft_losses'], linewidth=2, marker='o',
                 label=f"rank={r['rank']}")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Fine-tuning Loss by Conv Rank')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Conv trade-off 곡선 저장: {save_path}")


def plot_feature_maps(
    original_model: nn.Module,
    decomposed_model: nn.Module,
    test_loader: DataLoader,
    device: str,
    save_path: str = 'results/feature_map_comparison.png'
):
    """분해 전/후 Feature Map 비교"""
    # 테스트 이미지 가져오기
    images, labels = next(iter(test_loader))
    image = images[0:1].to(device)

    # Feature map 추출 함수
    def get_feature_maps(model, x):
        model.to(device)
        model.eval()
        maps = []
        with torch.no_grad():
            for i, layer in enumerate(model.features):
                x = layer(x)
                if isinstance(layer, (nn.Conv2d, nn.ReLU)):
                    maps.append((f'features.{i}', x.cpu()))
                # ChannelDecomposedConv2d도 처리
                if hasattr(layer, 'conv_pointwise'):
                    maps.append((f'features.{i} (decomposed)', x.cpu()))
        return maps

    original_maps = get_feature_maps(original_model, image)
    decomposed_maps = get_feature_maps(decomposed_model, image)

    # 처음 4개 레이어의 feature map 비교
    n_layers = min(4, len(original_maps), len(decomposed_maps))
    n_channels = 4  # 각 레이어에서 보여줄 채널 수

    fig, axes = plt.subplots(n_layers, n_channels * 2 + 1, figsize=(20, 3 * n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)

    for layer_idx in range(n_layers):
        orig_name, orig_map = original_maps[layer_idx]
        _, decomp_map = decomposed_maps[layer_idx]

        # 채널 수 맞추기 (다를 수 있음)
        n_ch = min(n_channels, orig_map.shape[1], decomp_map.shape[1])

        for ch in range(n_ch):
            # 원본
            ax = axes[layer_idx][ch]
            ax.imshow(orig_map[0, ch].numpy(), cmap='viridis')
            if layer_idx == 0:
                ax.set_title(f'Original\nCh {ch}', fontsize=8)
            ax.axis('off')

        # 구분선
        ax_mid = axes[layer_idx][n_channels]
        ax_mid.text(0.5, 0.5, orig_name, ha='center', va='center',
                    fontsize=7, rotation=90, transform=ax_mid.transAxes)
        ax_mid.axis('off')

        for ch in range(n_ch):
            # 분해 후
            ax = axes[layer_idx][n_channels + 1 + ch]
            ax.imshow(decomp_map[0, ch].numpy(), cmap='viridis')
            if layer_idx == 0:
                ax.set_title(f'Decomposed\nCh {ch}', fontsize=8)
            ax.axis('off')

    fig.suptitle('Feature Map Comparison: Original vs Decomposed', fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Feature map 비교 저장: {save_path}")


def plot_combined_summary(
    baseline_acc: float,
    baseline_params: int,
    conv_results: list[dict],
    combined_result: dict,
    save_path: str = 'results/combined_summary.png'
):
    """Combined 분해 종합 비교 바 차트"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 방법별 정확도
    methods = ['Baseline']
    accs = [baseline_acc]
    params = [baseline_params]

    for r in conv_results:
        methods.append(f'Conv (r={r["rank"]})')
        accs.append(r['acc_after_ft'])
        params.append(r['params'])

    methods.append(f'Combined\n(L={combined_result["linear_rank"]},'
                   f'C={combined_result["conv_rank"]})')
    accs.append(combined_result['acc_after_ft'])
    params.append(combined_result['params'])

    x = np.arange(len(methods))
    colors = ['green'] + ['steelblue'] * len(conv_results) + ['darkorange']

    # (1) 정확도 비교
    ax1 = axes[0]
    bars = ax1.bar(x, accs, color=colors, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=8)
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Accuracy Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, accs):
        ax1.annotate(f'{acc:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, acc),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)

    # (2) 파라미터 수 비교
    ax2 = axes[1]
    bars = ax2.bar(x, [p / 1000 for p in params], color=colors, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=8)
    ax2.set_ylabel('Parameters (K)')
    ax2.set_title('Parameter Count Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, p in zip(bars, params):
        ax2.annotate(f'{p/1000:.0f}K',
                     xy=(bar.get_x() + bar.get_width() / 2, p / 1000),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  종합 비교 시각화 저장: {save_path}")


# =============================================================================
# 6. 메인 실행
# =============================================================================

def main():
    print("=" * 60)
    print("Part 3: Conv 레이어 채널 분해")
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

    # Conv 분해 대상 분석
    print("\n[3] Conv 레이어 분석")
    print("-" * 40)
    print(f"  {'Layer':<20} {'Shape':<25} {'Params':<12} {'비율':<8}")
    print("  " + "-" * 65)

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            W = module.weight.data
            num_params = W.numel() + (module.bias.numel() if module.bias is not None else 0)
            ratio = num_params / baseline_params * 100
            print(f"  {name:<20} {str(list(W.shape)):<25} {num_params:>10,} {ratio:>6.1f}%")

    print(f"\n  → features.2 (Conv2d(128, 64, 3, 3))가 Conv 중 가장 큰 레이어")

    # Conv 분해 실험 (다양한 rank)
    print("\n[4] Conv 분해 실험 (features.2)")
    print("-" * 40)
    print(f"  대상: features.2 (Conv2d 128→64, 3×3)")
    print(f"  실험 ranks: {CONV_RANKS}")

    conv_results = experiment_conv_ranks(
        model, train_loader, test_loader, CONV_RANKS, DEVICE
    )

    # 결과 요약 테이블
    print(f"\n  {'Rank':<8} {'Before FT':<12} {'After FT':<12} {'Params':<12} {'압축률':<10}")
    print("  " + "-" * 54)
    for r in conv_results:
        print(f"  {r['rank']:<8} {r['acc_before_ft']:<12.2f} {r['acc_after_ft']:<12.2f} "
              f"{r['params']:<12,} {r['compression_ratio']:<10.2f}x")

    # 시각화
    plot_conv_tradeoff(conv_results, baseline_acc)

    # 여러 Conv 레이어 동시 분해
    print("\n[5] 다중 Conv 레이어 분해")
    print("-" * 40)

    multi_conv_result = experiment_multi_conv(
        model, train_loader, test_loader, rank=32, device=DEVICE
    )

    # Feature Map 비교
    print("\n[6] Feature Map 비교")
    print("-" * 40)

    decomposed_for_viz = decompose_model_conv(model, {'features.2': 32})
    plot_feature_maps(model, decomposed_for_viz, test_loader, DEVICE)

    # Combined: Linear SVD + Conv 분해
    print("\n[7] Combined 분해 (Linear SVD + Conv)")
    print("-" * 40)

    combined_result = experiment_combined(
        model, train_loader, test_loader,
        linear_rank=COMBINED_LINEAR_RANK,
        conv_rank=COMBINED_CONV_RANK,
        device=DEVICE
    )

    plot_combined_summary(
        baseline_acc, baseline_params,
        conv_results, combined_result
    )

    # 결과 저장
    print("\n[8] 결과 저장")
    print("-" * 40)
    os.makedirs('results', exist_ok=True)

    save_data = {
        'baseline_accuracy': baseline_acc,
        'baseline_params': baseline_params,
        'conv_results': [
            {k: v for k, v in r.items() if k != 'ft_losses'}
            for r in conv_results
        ],
        'multi_conv_result': multi_conv_result,
        'combined_result': {
            k: v for k, v in combined_result.items() if k != 'ft_losses'
        }
    }

    with open('results/conv_decomposition_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("  결과 저장: results/conv_decomposition_results.json")

    # 최종 요약
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"  Baseline: {baseline_acc:.2f}% ({baseline_params:,} params)")
    print()

    print("  Conv 분해 (features.2):")
    for r in conv_results:
        print(f"    rank={r['rank']}: {r['acc_after_ft']:.2f}% "
              f"({r['compression_ratio']:.2f}x 압축)")

    print()
    print(f"  Combined (Linear r={COMBINED_LINEAR_RANK} + Conv r={COMBINED_CONV_RANK}):")
    print(f"    정확도: {combined_result['acc_after_ft']:.2f}%")
    print(f"    파라미터: {combined_result['params']:,} "
          f"({combined_result['param_ratio']*100:.1f}%)")
    print(f"    압축률: {combined_result['compression_ratio']:.2f}x")

    print(f"\n  핵심 관찰:")
    print(f"    - Conv 분해는 Linear SVD보다 압축 효과가 작음")
    print(f"      (Conv가 전체 파라미터의 ~11%만 차지)")
    print(f"    - Linear SVD + Conv 분해를 결합하면 더 큰 압축 가능")
    print(f"    - Fine-tuning으로 분해 후 정확도 회복 가능")
    print(f"\n  다음 단계: Part 4에서 종합 분석")

    return {
        'baseline_accuracy': baseline_acc,
        'baseline_params': baseline_params,
        'conv_results': conv_results,
        'multi_conv_result': multi_conv_result,
        'combined_result': combined_result
    }


if __name__ == "__main__":
    results = main()
