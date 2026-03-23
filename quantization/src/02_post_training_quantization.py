"""
Part 2: Post-Training Quantization (PTQ)

이 스크립트에서 다루는 내용:
1. Dynamic Quantization (동적 양자화)
2. Static Quantization (정적 양자화)
3. PTQ 결과 비교 (정확도, 모델 크기, 추론 시간)
4. 양자화된 가중치 분포 시각화

Dynamic Quantization:
====================
- 가중치: 사전에 INT8로 변환 (저장 시)
- 활성화: 추론 시 실시간으로 양자화 범위 결정
- 장점: calibration 불필요, 적용 간단
- 단점: 활성화 양자화 오버헤드, Conv2d에는 지원 제한적
- 적합: RNN, Transformer 등 Linear 레이어 위주 모델

Static Quantization:
===================
- 가중치 + 활성화 모두 사전에 INT8로 변환
- calibration 데이터로 활성화 범위(scale, zero_point) 미리 측정
- 모델 구조 수정 필요: QuantStub(입력) / DeQuantStub(출력)
- fuse_modules로 Conv+ReLU 등을 하나의 연산으로 융합

Pipeline:
  1) 모델에 QuantStub/DeQuantStub 추가 (QuantizableCNN)
  2) fuse_model()로 레이어 융합
  3) qconfig 설정 (fbgemm 백엔드)
  4) prepare()로 observer 삽입
  5) calibration 데이터 통과 (활성화 범위 측정)
  6) convert()로 실제 양자화 적용
"""

import sys
import os
import copy
import time
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as quant
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'module'))
from models import CNN, QuantizableCNN


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

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 10
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

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str
) -> list[float]:
    """Cross-Entropy 학습"""
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


def test_model(model: nn.Module, test_loader: DataLoader, device: str) -> float:
    """모델 평가 (정확도 반환)"""
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


def get_model_size_mb(model: nn.Module) -> float:
    """모델의 저장 크기 측정 (MB)"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
    tmp_path = tmp.name
    tmp.close()
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return size_mb


def measure_inference_time(model: nn.Module, test_loader: DataLoader, device: str, num_batches: int = 50) -> float:
    """추론 시간 측정 (초)"""
    model.to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= 5:
                break
            inputs = inputs.to(device)
            _ = model(inputs)

    # 측정
    start_time = time.time()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
    elapsed = time.time() - start_time

    return elapsed


# =============================================================================
# 4. Dynamic Quantization
# =============================================================================

def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """
    Dynamic Quantization 적용

    torch.ao.quantization.quantize_dynamic()로 간단하게 적용 가능.
    주로 Linear 레이어를 INT8로 변환합니다.

    주의: Conv2d는 dynamic quantization에서 지원이 제한적이므로
    Linear 레이어만 양자화합니다.
    """
    # CPU로 이동 (양자화된 모델은 CPU에서만 동작)
    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()

    quantized_model = quant.quantize_dynamic(
        model_cpu,
        {nn.Linear},  # Linear 레이어만 양자화
        dtype=torch.qint8
    )

    return quantized_model


# =============================================================================
# 5. Static Quantization
# =============================================================================

def apply_static_quantization(
    fp32_state_dict: dict,
    train_loader: DataLoader,
    num_calibration_batches: int = 100
) -> nn.Module:
    """
    Static Quantization 적용

    Pipeline:
    1. QuantizableCNN 생성 + FP32 가중치 로드
    2. eval() → fuse_model() (Conv+ReLU, Linear+ReLU 융합)
    3. qconfig 설정 (fbgemm 백엔드)
    4. prepare() → observer 삽입
    5. calibration 데이터 통과
    6. convert() → 실제 양자화

    Args:
        fp32_state_dict: FP32 모델의 state_dict
        train_loader: calibration용 데이터 로더
        num_calibration_batches: calibration에 사용할 배치 수
    """
    # 1. QuantizableCNN 생성 + 가중치 로드
    model = QuantizableCNN(num_classes=NUM_CLASSES)

    # CNN의 가중치를 QuantizableCNN에 매핑
    # (QuantizableCNN은 Dropout이 없으므로 classifier 인덱스가 다름)
    quant_state_dict = {}
    for key, value in fp32_state_dict.items():
        if key.startswith('features.'):
            quant_state_dict[key] = value
        elif key == 'classifier.0.weight':
            quant_state_dict['classifier.0.weight'] = value
        elif key == 'classifier.0.bias':
            quant_state_dict['classifier.0.bias'] = value
        elif key == 'classifier.3.weight':
            # CNN의 classifier.3 → QuantizableCNN의 classifier.2
            quant_state_dict['classifier.2.weight'] = value
        elif key == 'classifier.3.bias':
            quant_state_dict['classifier.2.bias'] = value

    model.load_state_dict(quant_state_dict)

    # 2. eval 모드 → 레이어 융합
    model.eval()
    model.fuse_model()

    # 3. qconfig 설정
    # fbgemm: x86 CPU 최적화 백엔드
    model.qconfig = quant.get_default_qconfig('fbgemm')

    # 4. prepare: observer 삽입 (활성화 범위 측정 준비)
    quant.prepare(model, inplace=True)

    # 5. Calibration: 학습 데이터를 통과시켜 활성화 범위 측정
    print(f"    Calibration 진행 중 ({num_calibration_batches} batches)...")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(train_loader):
            if i >= num_calibration_batches:
                break
            model(inputs)  # CPU에서 실행

    # 6. Convert: 실제 양자화 적용
    quant.convert(model, inplace=True)

    return model


# =============================================================================
# 6. 시각화
# =============================================================================

def visualize_quantized_weights(
    fp32_model: nn.Module,
    quantized_model: nn.Module,
    save_path: str = 'results/quantized_weight_comparison.png'
):
    """FP32 vs 양자화 모델의 가중치 분포 비교"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Weight Distribution: FP32 vs Quantized (Static PTQ)', fontsize=14)

    # FP32 모델의 Conv2d 레이어
    fp32_layers = []
    for name, module in fp32_model.named_modules():
        if isinstance(module, nn.Conv2d):
            fp32_layers.append((name, module.weight.data.cpu().numpy().flatten()))

    for idx, (name, weights) in enumerate(fp32_layers[:4]):
        ax = axes[idx // 2][idx % 2]

        ax.hist(weights, bins=100, alpha=0.6, color='steelblue', label='FP32', edgecolor='black', linewidth=0.3)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f'{name}')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')
        ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    가중치 비교 시각화 저장: {save_path}")


def visualize_ptq_results(results: dict, save_path: str = 'results/ptq_comparison.png'):
    """PTQ 결과 비교 바 차트"""
    methods = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in methods]
    sizes = [results[m]['size_mb'] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 정확도 비교
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    bars1 = ax1.bar(methods, accuracies, color=colors[:len(methods)], edgecolor='black', linewidth=0.5)
    ax1.set_title('Test Accuracy Comparison', fontsize=13)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(min(accuracies) - 5, max(accuracies) + 3)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=11)

    # 모델 크기 비교
    bars2 = ax2.bar(methods, sizes, color=colors[:len(methods)], edgecolor='black', linewidth=0.5)
    ax2.set_title('Model Size Comparison', fontsize=13)
    ax2.set_ylabel('Size (MB)')
    for bar, size in zip(bars2, sizes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{size:.2f} MB', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    PTQ 결과 비교 시각화 저장: {save_path}")


# =============================================================================
# 7. 메인 실행
# =============================================================================

def main():
    print("=" * 60)
    print("Part 2: Post-Training Quantization (PTQ)")
    print("=" * 60)

    # 데이터 로드
    print("\n[1] CIFAR-10 데이터셋 로드 중...")
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)

    # Baseline 모델 학습 (또는 로드)
    print("\n[2] Baseline 모델 준비")
    print("-" * 40)

    baseline_path = 'data/trained_models/baseline_model.pth'
    torch.manual_seed(SEED)
    model = CNN(num_classes=NUM_CLASSES)

    if os.path.exists(baseline_path):
        model.load_state_dict(torch.load(baseline_path, map_location='cpu', weights_only=True))
        print("    저장된 Baseline 모델 로드 완료")
    else:
        print("    Baseline 모델이 없으므로 새로 학습합니다...")
        model.to(DEVICE)
        train_model(model, train_loader, EPOCHS, LEARNING_RATE, DEVICE)
        model.cpu()
        os.makedirs('data/trained_models', exist_ok=True)
        torch.save(model.state_dict(), baseline_path)

    model.to(DEVICE)
    baseline_acc = test_model(model, test_loader, DEVICE)
    print(f"    Baseline 정확도: {baseline_acc:.2f}%")

    baseline_size = get_model_size_mb(model)
    print(f"    Baseline 크기: {baseline_size:.2f} MB")

    fp32_state_dict = copy.deepcopy(model.cpu().state_dict())

    # =========================================================================
    # Dynamic Quantization
    # =========================================================================
    print("\n[3] Dynamic Quantization")
    print("-" * 40)
    print("    Linear 레이어를 INT8로 동적 양자화합니다.")
    print("    Conv2d는 dynamic quantization에서 지원 제한적 → Linear만 적용")

    dynamic_model = apply_dynamic_quantization(model)

    # Dynamic quantization 모델은 CPU에서만 동작
    dynamic_acc = test_model(dynamic_model, test_loader, 'cpu')
    print(f"    Dynamic PTQ 정확도: {dynamic_acc:.2f}%")

    dynamic_size = get_model_size_mb(dynamic_model)
    print(f"    Dynamic PTQ 크기: {dynamic_size:.2f} MB")
    print(f"    크기 감소율: {(1 - dynamic_size / baseline_size) * 100:.1f}%")

    # =========================================================================
    # Static Quantization
    # =========================================================================
    print("\n[4] Static Quantization")
    print("-" * 40)
    print("    QuantizableCNN + fuse_model + calibration → 전체 양자화")

    static_model = apply_static_quantization(
        fp32_state_dict, train_loader, num_calibration_batches=100
    )

    static_acc = test_model(static_model, test_loader, 'cpu')
    print(f"    Static PTQ 정확도: {static_acc:.2f}%")

    static_size = get_model_size_mb(static_model)
    print(f"    Static PTQ 크기: {static_size:.2f} MB")
    print(f"    크기 감소율: {(1 - static_size / baseline_size) * 100:.1f}%")

    # =========================================================================
    # 추론 시간 비교
    # =========================================================================
    print("\n[5] 추론 시간 비교 (CPU)")
    print("-" * 40)

    # 양자화 모델은 CPU에서만 동작하므로 모두 CPU에서 비교
    model_cpu = copy.deepcopy(model).cpu()

    baseline_time = measure_inference_time(model_cpu, test_loader, 'cpu')
    dynamic_time = measure_inference_time(dynamic_model, test_loader, 'cpu')
    static_time = measure_inference_time(static_model, test_loader, 'cpu')

    print(f"    Baseline (FP32): {baseline_time:.3f}초")
    print(f"    Dynamic PTQ:     {dynamic_time:.3f}초 ({dynamic_time/baseline_time*100:.1f}%)")
    print(f"    Static PTQ:      {static_time:.3f}초 ({static_time/baseline_time*100:.1f}%)")

    # =========================================================================
    # 시각화
    # =========================================================================
    print("\n[6] 결과 시각화")
    print("-" * 40)

    visualize_quantized_weights(model, static_model)

    results = {
        'Baseline\n(FP32)': {'accuracy': baseline_acc, 'size_mb': baseline_size},
        'Dynamic\nPTQ': {'accuracy': dynamic_acc, 'size_mb': dynamic_size},
        'Static\nPTQ': {'accuracy': static_acc, 'size_mb': static_size},
    }
    visualize_ptq_results(results)

    # =========================================================================
    # 결과 요약
    # =========================================================================
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"    {'Method':<25} {'Accuracy':>10} {'Size (MB)':>10} {'Speed':>10}")
    print("    " + "-" * 55)
    print(f"    {'Baseline (FP32)':<25} {baseline_acc:>9.2f}% {baseline_size:>9.2f} {baseline_time:>9.3f}s")
    print(f"    {'Dynamic PTQ':<25} {dynamic_acc:>9.2f}% {dynamic_size:>9.2f} {dynamic_time:>9.3f}s")
    print(f"    {'Static PTQ':<25} {static_acc:>9.2f}% {static_size:>9.2f} {static_time:>9.3f}s")

    print(f"\n    Dynamic PTQ: Linear만 양자화 → 크기 감소 제한적, 적용 간단")
    print(f"    Static PTQ: 전체 양자화 → 크기 대폭 감소, calibration 필요")

    return {
        'baseline': {'accuracy': baseline_acc, 'size_mb': baseline_size, 'time': baseline_time},
        'dynamic': {'accuracy': dynamic_acc, 'size_mb': dynamic_size, 'time': dynamic_time},
        'static': {'accuracy': static_acc, 'size_mb': static_size, 'time': static_time},
    }


if __name__ == "__main__":
    results = main()
