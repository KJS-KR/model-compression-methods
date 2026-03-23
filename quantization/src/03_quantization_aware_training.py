"""
Part 3: Quantization-Aware Training (QAT)

이 스크립트에서 다루는 내용:
1. QAT 개념 (Fake Quantization, Straight-Through Estimator)
2. QAT 적용 및 학습
3. QAT vs Static PTQ 비교
4. QAT Fine-tuning 에포크별 효과

QAT 핵심 아이디어:
================
- 학습(training) 중에 양자화 효과를 시뮬레이션
- Forward pass: Fake Quantization (FQ) 노드 삽입
  x_fq = dequant(quant(x)) = round(x/s)*s  (s=scale)
  → 양자화로 인한 정보 손실을 학습 중 경험
- Backward pass: Straight-Through Estimator (STE)
  ∂L/∂x ≈ ∂L/∂x_fq  (round 함수의 gradient를 1로 근사)
  → gradient가 round를 무시하고 그대로 전파

왜 QAT가 PTQ보다 나은가?
======================
- PTQ: 학습 완료 후 양자화 → 모델이 양자화 오차에 적응하지 못함
- QAT: 학습 중 양자화 시뮬레이션 → 모델이 양자화 오차에 적응
- 특히 낮은 비트(INT4 등)에서 차이가 큼

Pipeline:
  1) FP32 모델 학습 (Baseline)
  2) QuantizableCNN에 가중치 로드
  3) fuse_model() (train 모드에서)
  4) QAT qconfig 설정
  5) prepare_qat() → Fake Quantization 노드 삽입
  6) QAT Fine-tuning (추가 학습)
  7) convert() → 실제 INT8 모델
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
QAT_LEARNING_RATE = 0.0001  # QAT fine-tuning은 낮은 LR 사용
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
        print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return epoch_losses


def train_qat(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float
) -> list[float]:
    """
    QAT 학습 (CPU에서 실행)

    QAT 모델은 Fake Quantization 노드가 삽입되어 있으므로
    CPU에서 학습해야 합니다.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    epoch_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # QAT는 CPU에서 실행
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

    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= 5:
                break
            inputs = inputs.to(device)
            _ = model(inputs)

    start_time = time.time()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
    return time.time() - start_time


# =============================================================================
# 4. 가중치 매핑: CNN → QuantizableCNN
# =============================================================================

def load_cnn_weights_to_quantizable(fp32_state_dict: dict, num_classes: int = 10) -> QuantizableCNN:
    """
    CNN의 state_dict를 QuantizableCNN에 로드

    CNN의 classifier: Linear(2048,512) + ReLU + Dropout(0.1) + Linear(512,10)
      → classifier.0, classifier.1, classifier.2(Dropout), classifier.3

    QuantizableCNN의 classifier: Linear(2048,512) + ReLU + Linear(512,10)
      → classifier.0, classifier.1, classifier.2
    """
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
# 5. QAT 적용
# =============================================================================

def apply_qat(
    fp32_state_dict: dict,
    train_loader: DataLoader,
    test_loader: DataLoader,
    qat_epochs: int = 5,
    qat_lr: float = 0.0001
) -> tuple[nn.Module, list[float]]:
    """
    Quantization-Aware Training 적용

    Pipeline:
    1. QuantizableCNN + FP32 가중치 로드
    2. train() 모드 → fuse_model()
    3. QAT qconfig 설정
    4. prepare_qat() → Fake Quantization 노드 삽입
    5. QAT Fine-tuning
    6. convert() → 실제 INT8 모델
    """
    # 1. 모델 생성 + 가중치 로드
    model = load_cnn_weights_to_quantizable(fp32_state_dict, NUM_CLASSES)

    # 2. train 모드에서 fuse_model
    model.train()
    model.fuse_model()

    # 3. QAT qconfig 설정
    # QAT용 qconfig는 Fake Quantization observer를 사용
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')

    # 4. prepare_qat: Fake Quantization 노드 삽입
    quant.prepare_qat(model, inplace=True)

    print(f"    QAT 학습 시작 ({qat_epochs} epochs, LR={qat_lr})...")

    # 5. QAT Fine-tuning (CPU에서)
    qat_losses = train_qat(model, train_loader, qat_epochs, qat_lr)

    # QAT 학습 중 정확도 확인 (아직 Fake Quantization 상태)
    fq_acc = test_model(model, test_loader, 'cpu')
    print(f"    Fake Quantization 상태 정확도: {fq_acc:.2f}%")

    # 6. convert: 실제 INT8 양자화
    model.eval()
    quantized_model = quant.convert(model)

    return quantized_model, qat_losses


# =============================================================================
# 6. Static PTQ (비교용)
# =============================================================================

def apply_static_ptq(
    fp32_state_dict: dict,
    train_loader: DataLoader,
    num_calibration_batches: int = 100
) -> nn.Module:
    """비교를 위한 Static PTQ"""
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


# =============================================================================
# 7. 시각화
# =============================================================================

def visualize_qat_training(qat_losses: list[float], save_path: str = 'results/qat_training_curve.png'):
    """QAT 학습 곡선"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(qat_losses) + 1), qat_losses, 'o-', color='#FF5722', linewidth=2, markersize=6)
    ax.set_title('QAT Fine-tuning Loss Curve', fontsize=13)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    QAT 학습 곡선 저장: {save_path}")


def visualize_qat_vs_ptq(results: dict, save_path: str = 'results/qat_vs_ptq.png'):
    """QAT vs Static PTQ 비교"""
    methods = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in methods]
    sizes = [results[m]['size_mb'] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    bars1 = ax1.bar(methods, accuracies, color=colors[:len(methods)], edgecolor='black', linewidth=0.5)
    ax1.set_title('Accuracy: Baseline vs PTQ vs QAT', fontsize=13)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(min(accuracies) - 5, max(accuracies) + 3)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=11)

    bars2 = ax2.bar(methods, sizes, color=colors[:len(methods)], edgecolor='black', linewidth=0.5)
    ax2.set_title('Model Size: Baseline vs PTQ vs QAT', fontsize=13)
    ax2.set_ylabel('Size (MB)')
    for bar, size in zip(bars2, sizes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{size:.2f} MB', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    QAT vs PTQ 비교 시각화 저장: {save_path}")


# =============================================================================
# 8. 메인 실행
# =============================================================================

def main():
    print("=" * 60)
    print("Part 3: Quantization-Aware Training (QAT)")
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
    print(f"    Baseline 정확도: {baseline_acc:.2f}%")
    baseline_size = get_model_size_mb(model)
    print(f"    Baseline 크기: {baseline_size:.2f} MB")

    fp32_state_dict = copy.deepcopy(model.cpu().state_dict())

    # =========================================================================
    # Static PTQ (비교 기준)
    # =========================================================================
    print("\n[3] Static PTQ (비교 기준)")
    print("-" * 40)

    static_model = apply_static_ptq(fp32_state_dict, train_loader)
    static_acc = test_model(static_model, test_loader, 'cpu')
    static_size = get_model_size_mb(static_model)
    print(f"    Static PTQ 정확도: {static_acc:.2f}%")
    print(f"    Static PTQ 크기: {static_size:.2f} MB")

    # =========================================================================
    # QAT
    # =========================================================================
    print("\n[4] Quantization-Aware Training (QAT)")
    print("-" * 40)
    print(f"    QAT는 Fake Quantization을 통해 학습 중 양자화 오차에 적응합니다.")
    print(f"    Fine-tuning: {QAT_EPOCHS} epochs, LR={QAT_LEARNING_RATE}")

    qat_model, qat_losses = apply_qat(
        fp32_state_dict, train_loader, test_loader,
        qat_epochs=QAT_EPOCHS, qat_lr=QAT_LEARNING_RATE
    )

    qat_acc = test_model(qat_model, test_loader, 'cpu')
    qat_size = get_model_size_mb(qat_model)
    print(f"    QAT 최종 정확도: {qat_acc:.2f}%")
    print(f"    QAT 크기: {qat_size:.2f} MB")

    # =========================================================================
    # QAT 에포크별 효과 실험
    # =========================================================================
    print("\n[5] QAT 에포크별 효과 실험")
    print("-" * 40)

    epoch_results = []
    for ep in [1, 3, 5, 10]:
        print(f"\n    QAT {ep} epochs:")
        qat_m, _ = apply_qat(
            fp32_state_dict, train_loader, test_loader,
            qat_epochs=ep, qat_lr=QAT_LEARNING_RATE
        )
        acc = test_model(qat_m, test_loader, 'cpu')
        epoch_results.append((ep, acc))
        print(f"    → 정확도: {acc:.2f}%")

    # =========================================================================
    # 추론 시간 비교
    # =========================================================================
    print("\n[6] 추론 시간 비교 (CPU)")
    print("-" * 40)

    model_cpu = copy.deepcopy(model).cpu()
    baseline_time = measure_inference_time(model_cpu, test_loader, 'cpu')
    static_time = measure_inference_time(static_model, test_loader, 'cpu')
    qat_time = measure_inference_time(qat_model, test_loader, 'cpu')

    print(f"    Baseline (FP32): {baseline_time:.3f}초")
    print(f"    Static PTQ:      {static_time:.3f}초")
    print(f"    QAT:             {qat_time:.3f}초")

    # =========================================================================
    # 시각화
    # =========================================================================
    print("\n[7] 결과 시각화")
    print("-" * 40)

    visualize_qat_training(qat_losses)

    results = {
        'Baseline\n(FP32)': {'accuracy': baseline_acc, 'size_mb': baseline_size},
        'Static\nPTQ': {'accuracy': static_acc, 'size_mb': static_size},
        'QAT': {'accuracy': qat_acc, 'size_mb': qat_size},
    }
    visualize_qat_vs_ptq(results)

    # 에포크별 결과 시각화
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs_list, accs_list = zip(*epoch_results)
    ax.plot(epochs_list, accs_list, 'o-', color='#4CAF50', linewidth=2, markersize=8)
    ax.axhline(y=baseline_acc, color='blue', linestyle='--', alpha=0.7, label=f'Baseline ({baseline_acc:.1f}%)')
    ax.axhline(y=static_acc, color='orange', linestyle='--', alpha=0.7, label=f'Static PTQ ({static_acc:.1f}%)')
    ax.set_title('QAT Accuracy vs Number of Fine-tuning Epochs', fontsize=13)
    ax.set_xlabel('QAT Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/qat_epoch_experiment.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    QAT 에포크 실험 시각화 저장: results/qat_epoch_experiment.png")

    # =========================================================================
    # 결과 요약
    # =========================================================================
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"    {'Method':<25} {'Accuracy':>10} {'Size (MB)':>10} {'Speed':>10}")
    print("    " + "-" * 55)
    print(f"    {'Baseline (FP32)':<25} {baseline_acc:>9.2f}% {baseline_size:>9.2f} {baseline_time:>9.3f}s")
    print(f"    {'Static PTQ':<25} {static_acc:>9.2f}% {static_size:>9.2f} {static_time:>9.3f}s")
    print(f"    {'QAT':<25} {qat_acc:>9.2f}% {qat_size:>9.2f} {qat_time:>9.3f}s")

    print(f"\n    QAT 에포크별 정확도:")
    for ep, acc in epoch_results:
        print(f"      {ep} epochs → {acc:.2f}%")

    print(f"\n    QAT는 학습 중 양자화를 시뮬레이션하여")
    print(f"    PTQ 대비 정확도 손실을 줄일 수 있습니다.")

    return {
        'baseline': {'accuracy': baseline_acc, 'size_mb': baseline_size},
        'static_ptq': {'accuracy': static_acc, 'size_mb': static_size},
        'qat': {'accuracy': qat_acc, 'size_mb': qat_size},
        'epoch_results': epoch_results,
    }


if __name__ == "__main__":
    results = main()
