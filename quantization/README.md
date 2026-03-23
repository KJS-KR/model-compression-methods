# Vision Quantization 튜토리얼

학습된 CNN 모델의 가중치와 활성화를 낮은 비트(INT8)로 변환하여 모델을 압축하는 Quantization 기법을 학습합니다.

## 환경 설정

```bash
pip install -r requirements.txt
```

**요구사항:**
- Python 3.10+
- PyTorch 2.0+
- GPU (RTX 4000 시리즈 권장, 학습 시)

## 튜토리얼 구조

```
quantization/
├── requirements.txt
├── module/
│   └── models/
│       ├── __init__.py
│       ├── base_model.py              # CNN (~1.18M params)
│       └── quantizable_model.py       # QuantizableCNN (QuantStub/DeQuantStub)
├── src/
│   ├── 01_concept_and_baseline.py     # 개념 + Baseline 학습
│   ├── 02_post_training_quantization.py  # Dynamic/Static PTQ
│   ├── 03_quantization_aware_training.py # QAT
│   └── 04_experiment_analysis.py      # 종합 실험 및 시각화
├── data/
└── results/
```

## 실행 순서

### Part 1: 기본 개념 및 Baseline
```bash
python src/01_concept_and_baseline.py
```
- CNN 모델 학습 (Baseline)
- 모델 크기 측정 (FP32)
- 가중치 분포 시각화
- 레이어별 파라미터 분석

### Part 2: Post-Training Quantization (PTQ)
```bash
python src/02_post_training_quantization.py
```
- Dynamic Quantization (Linear 레이어만)
- Static Quantization (전체 양자화 + calibration)
- PTQ 결과 비교 (정확도, 크기, 속도)

### Part 3: Quantization-Aware Training (QAT)
```bash
python src/03_quantization_aware_training.py
```
- QAT (Fake Quantization + Fine-tuning)
- QAT vs Static PTQ 비교
- QAT 에포크별 효과

### Part 4: 종합 실험 및 분석
```bash
python src/04_experiment_analysis.py
```
- 모든 방법 비교 (Baseline, Dynamic, Static, QAT)
- Calibration 데이터 크기 실험
- QAT 에포크 수 실험
- 종합 결과 시각화

## 핵심 개념

### Quantization 분류

```
Quantization
├── Dynamic PTQ (동적 양자화)
│   └── 가중치: 사전 INT8 / 활성화: 추론 시 동적
│
├── Static PTQ (정적 양자화)
│   └── 가중치 + 활성화 모두 사전 INT8 (calibration 필요)
│
└── QAT (Quantization-Aware Training)
    └── 학습 중 Fake Quantization으로 양자화 시뮬레이션
```

### 방법별 특징

| 방법 | 적용 난이도 | 모델 크기 | 속도 향상 | 정확도 유지 |
|------|-----------|----------|----------|-----------|
| Dynamic PTQ | 매우 쉬움 | 약간 감소 | 약간 | 좋음 |
| Static PTQ | 보통 | 대폭 감소 (~4x) | 좋음 | 보통 |
| QAT | 어려움 | 대폭 감소 (~4x) | 좋음 | 매우 좋음 |

### Quantization Pipeline

```
[Dynamic PTQ]
학습된 모델 → quantize_dynamic() → 양자화된 모델

[Static PTQ]
학습된 모델 → QuantizableCNN + fuse_model()
  → prepare() → calibration → convert() → 양자화된 모델

[QAT]
학습된 모델 → QuantizableCNN + fuse_model()
  → prepare_qat() → QAT 학습 → convert() → 양자화된 모델
```

## 예상 결과

| Method | Test Accuracy | Model Size | 비고 |
|--------|--------------|------------|------|
| Baseline (FP32) | ~74% | ~4.7 MB | 기준 |
| Dynamic PTQ | ~74% | ~4.5 MB | Linear만 양자화 |
| Static PTQ | ~72-74% | ~1.2 MB | 전체 양자화 |
| QAT | ~73-74% | ~1.2 MB | 학습 중 양자화 시뮬레이션 |

## 노트북 커널 문제 해결 (에이전트 수정 후 무한 로딩/Interrupting)

에이전트가 `.ipynb`를 수정한 뒤 실행·재시작 시 커널이 **Interrupting** 상태로 멈추거나 무한 로딩될 수 있습니다. 아래 순서대로 시도해 보세요.

### 1. 출력 비우기 후 재시작 (가장 먼저 시도)
- 메뉴: **Cell → Clear All Outputs**
- 그다음 **Kernel → Restart** (또는 재시작 버튼)
- 노트북을 **저장**한 뒤 다시 실행

### 2. 커널을 표준 Python으로 변경
- 상태바에서 커널 선택 클릭
- **"Python 3.10.x"** 같은 기본 Python 커널 선택

### 3. 스크립트로 실행 (가장 안정적)
```bash
cd quantization
python src/01_concept_and_baseline.py   # Part 1
python src/02_post_training_quantization.py  # Part 2
python src/03_quantization_aware_training.py # Part 3
python src/04_experiment_analysis.py    # Part 4
```

---

## 참고 논문

- [Jacob et al., 2018 - Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
- [Krishnamoorthi, 2018 - Quantizing deep convolutional networks for efficient inference](https://arxiv.org/abs/1806.08342)
