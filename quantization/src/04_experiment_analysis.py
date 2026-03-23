"""
Part 4: 종합 실험 및 분석

이 스크립트에서 다루는 내용:
1. 모든 양자화 방법 종합 비교 (Baseline, Dynamic PTQ, Static PTQ, QAT)
2. Calibration 데이터 크기 실험 (Static PTQ)
3. QAT 에포크 수 실험
4. 종합 결과 시각화

종합 비교:
=========
- Dynamic PTQ: 가장 간단. Linear만 양자화. calibration 불필요
- Static PTQ: Conv+Linear 모두 양자화. calibration 필요
- QAT: 학습 중 양자화 시뮬레이션. 가장 높은 정확도, 학습 비용 추가
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
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()
print(f"Using device: {DEVICE}")

BATCH_SIZE = 128
EPOCHS = 10
QAT_EPOCHS = 5
LEARNING_RATE = 0.001
QAT_LEARNING_RATE = 0.0001
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
# 3. 유틸리티 함수
# =============================================================================

def train_model(model, train_loader, epochs, learning_rate, device):
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
        print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return epoch_losses


def train_qat(model, train_loader, epochs, learning_rate):
    """QAT 학습 (CPU)"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    epoch_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"  QAT Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return epoch_losses


def test_model(model, test_loader, device):
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


def get_model_size_mb(model):
    """모델 크기 측정 (MB)"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
    tmp_path = tmp.name
    tmp.close()
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return size_mb


def measure_inference_time(model, test_loader, device, num_batches=50):
    """추론 시간 측정"""
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= 5:
                break
            _ = model(inputs.to(device))

    start = time.time()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            _ = model(inputs.to(device))
    return time.time() - start


def load_cnn_weights_to_quantizable(fp32_state_dict, num_classes=10):
    """CNN → QuantizableCNN 가중치 매핑"""
    model = QuantizableCNN(num_classes=num_classes)
    quant_state_dict = {}
    for key, value in fp32_state_dict.items():
        if key.startswith('features.'):
            quant_state_dict[key] = value
        elif key == 'classifier.0.weight':
            quant_state_dict['classifier.0.weight'] = value
        elif key == 'classifier.0.bias':
            quant_state_dict['classifier.0.bias'] = value
        elif key == 'classifier.3.weight':
            quant_state_dict['classifier.2.weight'] = value
        elif key == 'classifier.3.bias':
            quant_state_dict['classifier.2.bias'] = value
    model.load_state_dict(quant_state_dict)
    return model


# =============================================================================
# 4. 양자화 방법 적용 함수들
# =============================================================================

def apply_dynamic_ptq(model):
    """Dynamic Quantization"""
    m = copy.deepcopy(model).cpu()
    m.eval()
    return quant.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)


def apply_static_ptq(fp32_state_dict, train_loader, num_calibration_batches=100):
    """Static Quantization"""
    model = load_cnn_weights_to_quantizable(fp32_state_dict, NUM_CLASSES)
    model.eval()
    model.fuse_model()
    model.qconfig = quant.get_default_qconfig('fbgemm')
    quant.prepare(model, inplace=True)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(train_loader):
            if i >= num_calibration_batches:
                break
            model(inputs)

    quant.convert(model, inplace=True)
    return model


def apply_qat(fp32_state_dict, train_loader, qat_epochs=5, qat_lr=0.0001):
    """Quantization-Aware Training"""
    model = load_cnn_weights_to_quantizable(fp32_state_dict, NUM_CLASSES)
    model.train()
    model.fuse_model()
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')
    quant.prepare_qat(model, inplace=True)

    train_qat(model, train_loader, qat_epochs, qat_lr)

    model.eval()
    quantized = quant.convert(model)
    return quantized


# =============================================================================
# 5. 메인 실행
# =============================================================================

def main():
    print("=" * 60)
    print("Part 4: Quantization - 종합 실험 및 분석")
    print("=" * 60)

    # 데이터 로드
    print("\n[1] CIFAR-10 데이터셋 로드 중...")
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)

    # Baseline 모델 준비
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
    baseline_size = get_model_size_mb(model)
    print(f"    Baseline 정확도: {baseline_acc:.2f}%, 크기: {baseline_size:.2f} MB")

    fp32_state_dict = copy.deepcopy(model.cpu().state_dict())

    # =========================================================================
    # 모든 방법 적용
    # =========================================================================
    print("\n[3] 모든 양자화 방법 적용")
    print("-" * 40)

    # Dynamic PTQ
    print("\n  >> Dynamic PTQ")
    dynamic_model = apply_dynamic_ptq(model)
    dynamic_acc = test_model(dynamic_model, test_loader, 'cpu')
    dynamic_size = get_model_size_mb(dynamic_model)
    print(f"     정확도: {dynamic_acc:.2f}%, 크기: {dynamic_size:.2f} MB")

    # Static PTQ
    print("\n  >> Static PTQ (calibration: 100 batches)")
    static_model = apply_static_ptq(fp32_state_dict, train_loader)
    static_acc = test_model(static_model, test_loader, 'cpu')
    static_size = get_model_size_mb(static_model)
    print(f"     정확도: {static_acc:.2f}%, 크기: {static_size:.2f} MB")

    # QAT
    print("\n  >> QAT (5 epochs)")
    qat_model = apply_qat(fp32_state_dict, train_loader, QAT_EPOCHS, QAT_LEARNING_RATE)
    qat_acc = test_model(qat_model, test_loader, 'cpu')
    qat_size = get_model_size_mb(qat_model)
    print(f"     정확도: {qat_acc:.2f}%, 크기: {qat_size:.2f} MB")

    # =========================================================================
    # 추론 시간 비교
    # =========================================================================
    print("\n[4] 추론 시간 비교 (CPU, 50 batches)")
    print("-" * 40)

    model_cpu = copy.deepcopy(model).cpu()
    baseline_time = measure_inference_time(model_cpu, test_loader, 'cpu')
    dynamic_time = measure_inference_time(dynamic_model, test_loader, 'cpu')
    static_time = measure_inference_time(static_model, test_loader, 'cpu')
    qat_time = measure_inference_time(qat_model, test_loader, 'cpu')

    print(f"    Baseline: {baseline_time:.3f}s")
    print(f"    Dynamic:  {dynamic_time:.3f}s ({dynamic_time/baseline_time*100:.1f}%)")
    print(f"    Static:   {static_time:.3f}s ({static_time/baseline_time*100:.1f}%)")
    print(f"    QAT:      {qat_time:.3f}s ({qat_time/baseline_time*100:.1f}%)")

    # =========================================================================
    # Calibration 데이터 크기 실험
    # =========================================================================
    print("\n[5] Calibration 데이터 크기 실험 (Static PTQ)")
    print("-" * 40)

    calib_sizes = [10, 25, 50, 100, 200, 390]  # 390 ≈ 전체 학습 데이터
    calib_results = []

    for n_batches in calib_sizes:
        m = apply_static_ptq(fp32_state_dict, train_loader, num_calibration_batches=n_batches)
        acc = test_model(m, test_loader, 'cpu')
        n_samples = min(n_batches * BATCH_SIZE, 50000)
        calib_results.append((n_batches, n_samples, acc))
        print(f"    {n_batches:>4} batches ({n_samples:>6} samples) → {acc:.2f}%")

    # =========================================================================
    # QAT 에포크 실험
    # =========================================================================
    print("\n[6] QAT 에포크 수 실험")
    print("-" * 40)

    qat_epoch_list = [1, 2, 3, 5, 10]
    qat_epoch_results = []

    for ep in qat_epoch_list:
        print(f"\n    QAT {ep} epochs:")
        m = apply_qat(fp32_state_dict, train_loader, qat_epochs=ep, qat_lr=QAT_LEARNING_RATE)
        acc = test_model(m, test_loader, 'cpu')
        qat_epoch_results.append((ep, acc))
        print(f"    → 정확도: {acc:.2f}%")

    # =========================================================================
    # 종합 시각화
    # =========================================================================
    print("\n[7] 종합 결과 시각화")
    print("-" * 40)
    os.makedirs('results', exist_ok=True)

    # 7-1. 모든 방법 비교 (정확도 + 크기 + 속도)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    methods = ['Baseline\n(FP32)', 'Dynamic\nPTQ', 'Static\nPTQ', 'QAT']
    accs = [baseline_acc, dynamic_acc, static_acc, qat_acc]
    sizes = [baseline_size, dynamic_size, static_size, qat_size]
    times = [baseline_time, dynamic_time, static_time, qat_time]
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

    # 정확도
    bars = axes[0].bar(methods, accs, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_title('Test Accuracy', fontsize=13)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim(min(accs) - 5, max(accs) + 3)
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

    # 모델 크기
    bars = axes[1].bar(methods, sizes, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_title('Model Size', fontsize=13)
    axes[1].set_ylabel('Size (MB)')
    for bar, s in zip(bars, sizes):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{s:.2f}MB', ha='center', va='bottom', fontsize=10)

    # 추론 시간
    bars = axes[2].bar(methods, times, color=colors, edgecolor='black', linewidth=0.5)
    axes[2].set_title('Inference Time (CPU, 50 batches)', fontsize=13)
    axes[2].set_ylabel('Time (seconds)')
    for bar, t in zip(bars, times):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{t:.3f}s', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/comprehensive_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    종합 비교 차트 저장: results/comprehensive_comparison.png")

    # 7-2. Calibration 크기 실험
    fig, ax = plt.subplots(figsize=(8, 5))
    batches, samples, cal_accs = zip(*calib_results)
    ax.plot(samples, cal_accs, 'o-', color='#4CAF50', linewidth=2, markersize=8)
    ax.axhline(y=baseline_acc, color='blue', linestyle='--', alpha=0.7, label=f'Baseline ({baseline_acc:.1f}%)')
    ax.set_title('Static PTQ: Calibration Data Size vs Accuracy', fontsize=13)
    ax.set_xlabel('Number of Calibration Samples')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/calibration_experiment.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Calibration 실험 차트 저장: results/calibration_experiment.png")

    # 7-3. QAT 에포크 실험
    fig, ax = plt.subplots(figsize=(8, 5))
    ep_list, ep_accs = zip(*qat_epoch_results)
    ax.plot(ep_list, ep_accs, 'o-', color='#E91E63', linewidth=2, markersize=8)
    ax.axhline(y=baseline_acc, color='blue', linestyle='--', alpha=0.7, label=f'Baseline ({baseline_acc:.1f}%)')
    ax.axhline(y=static_acc, color='green', linestyle='--', alpha=0.7, label=f'Static PTQ ({static_acc:.1f}%)')
    ax.set_title('QAT: Number of Epochs vs Accuracy', fontsize=13)
    ax.set_xlabel('QAT Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/qat_epochs_experiment.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    QAT 에포크 실험 차트 저장: results/qat_epochs_experiment.png")

    # =========================================================================
    # 텍스트 결과 저장
    # =========================================================================
    summary_path = 'results/summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Quantization 종합 실험 결과\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"{'Method':<25} {'Accuracy':>10} {'Size (MB)':>10} {'Time (s)':>10}\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'Baseline (FP32)':<25} {baseline_acc:>9.2f}% {baseline_size:>9.2f} {baseline_time:>9.3f}\n")
        f.write(f"{'Dynamic PTQ':<25} {dynamic_acc:>9.2f}% {dynamic_size:>9.2f} {dynamic_time:>9.3f}\n")
        f.write(f"{'Static PTQ':<25} {static_acc:>9.2f}% {static_size:>9.2f} {static_time:>9.3f}\n")
        f.write(f"{'QAT':<25} {qat_acc:>9.2f}% {qat_size:>9.2f} {qat_time:>9.3f}\n")

        f.write(f"\nCalibration 데이터 크기 실험 (Static PTQ):\n")
        for n_b, n_s, acc in calib_results:
            f.write(f"  {n_b:>4} batches ({n_s:>6} samples) → {acc:.2f}%\n")

        f.write(f"\nQAT 에포크 실험:\n")
        for ep, acc in qat_epoch_results:
            f.write(f"  {ep:>2} epochs → {acc:.2f}%\n")

    print(f"\n    텍스트 결과 저장: {summary_path}")

    # =========================================================================
    # 최종 요약
    # =========================================================================
    print("\n" + "=" * 60)
    print("최종 결과 요약")
    print("=" * 60)
    print(f"\n    {'Method':<25} {'Accuracy':>10} {'Size (MB)':>10} {'Time (s)':>10}")
    print("    " + "-" * 55)
    print(f"    {'Baseline (FP32)':<25} {baseline_acc:>9.2f}% {baseline_size:>9.2f} {baseline_time:>9.3f}")
    print(f"    {'Dynamic PTQ':<25} {dynamic_acc:>9.2f}% {dynamic_size:>9.2f} {dynamic_time:>9.3f}")
    print(f"    {'Static PTQ':<25} {static_acc:>9.2f}% {static_size:>9.2f} {static_time:>9.3f}")
    print(f"    {'QAT':<25} {qat_acc:>9.2f}% {qat_size:>9.2f} {qat_time:>9.3f}")

    print(f"\n    핵심 결론:")
    print(f"    - Dynamic PTQ: 가장 간단, Linear만 양자화 → 크기 감소 제한적")
    print(f"    - Static PTQ: 전체 양자화, calibration 필요 → 크기 대폭 감소")
    print(f"    - QAT: 학습 중 양자화 시뮬레이션 → 정확도 손실 최소화")
    print(f"    - Calibration 데이터 크기: 일정 수준 이상이면 정확도 수렴")
    print(f"    - QAT 에포크: 적은 에포크로도 효과적, 과도한 학습은 불필요")


if __name__ == "__main__":
    main()
