# Vision Pruning 튜토리얼

학습된 CNN 모델에서 불필요한 가중치/필터를 제거하여 모델을 압축하는 Pruning 기법을 학습합니다.

## 환경 설정

```bash
pip install -r requirements.txt
```

**요구사항:**
- Python 3.10+
- PyTorch 2.0+
- GPU (RTX 4000 시리즈 권장)

## 튜토리얼 구조

```
pruning/
├── requirements.txt
├── module/
│   └── models/
│       ├── __init__.py
│       └── base_model.py          # CNN (~1.18M params)
├── src/
│   ├── 01_concept_and_baseline.py    # 개념 + Baseline 학습
│   ├── 02_unstructured_pruning.py    # 비정형 프루닝
│   ├── 03_structured_pruning.py      # 정형 프루닝 + Fine-tuning
│   └── 04_experiment_analysis.py     # 종합 실험 및 시각화
├── data/
└── results/
```

## 실행 순서

### Part 1: 기본 개념 및 Baseline
```bash
python src/01_concept_and_baseline.py
```
- CNN 모델 학습 (Baseline)
- 가중치 분포 시각화
- 레이어별 파라미터 분석

### Part 2: Unstructured Pruning (비정형 프루닝)
```bash
python src/02_unstructured_pruning.py
```
- L1 Unstructured Pruning (magnitude 기반)
- Random Unstructured Pruning
- Global Unstructured Pruning
- Sparsity별 정확도 변화

### Part 3: Structured Pruning (정형 프루닝)
```bash
python src/03_structured_pruning.py
```
- L1/L2 Structured Pruning (필터 단위)
- Fine-tuning (pruning 후 재학습)
- Pruning + Fine-tuning 효과 비교

### Part 4: 종합 실험 및 분석
```bash
python src/04_experiment_analysis.py
```
- 모든 방법 비교
- Iterative Pruning (점진적 프루닝)
- 결과 시각화

## 핵심 개념

### Pruning 분류

```
Pruning
├── Unstructured (비정형)
│   ├── L1 Magnitude: |w|가 작은 개별 가중치 제거
│   ├── Random: 무작위 가중치 제거
│   └── Global: 전체 모델에서 중요도 기준 제거
│
└── Structured (정형)
    ├── L1/L2 Norm: norm이 작은 필터 전체 제거
    └── Random: 무작위 필터 제거
```

### 방법별 특징

| 방법 | 단위 | 압축률 | 속도 향상 | 정확도 유지 |
|------|------|--------|----------|-----------|
| L1 Unstructured | 개별 가중치 | 높음 | 제한적 | 좋음 |
| Global Unstructured | 개별 가중치 | 높음 | 제한적 | 매우 좋음 |
| L1 Structured | 필터/채널 | 중간 | 실질적 | 보통 |
| Structured + Fine-tune | 필터/채널 | 중간 | 실질적 | 좋음 |
| Iterative Pruning | 점진적 | 높음 | 가변적 | 매우 좋음 |

### Pruning Pipeline

```
학습된 모델 → Pruning 적용 → (Fine-tuning) → 압축된 모델
                  ↑                  |
                  └──── 반복 (Iterative) ────┘
```

## 예상 결과

| Method | Test Accuracy | Sparsity |
|--------|---------------|----------|
| Baseline (no pruning) | ~74% | 0% |
| L1 Unstructured (30%) | ~73% | 30% |
| Global Unstructured (30%) | ~74% | 30% |
| L1 Structured (30%) | ~70% | ~30% |
| L1 Structured + Fine-tune | ~73% | ~30% |
| Iterative Pruning (50%) | ~73% | 50% |

## 노트북 커널 문제 해결 (에이전트 수정 후 무한 로딩/Interrupting)

에이전트가 `.ipynb`를 수정한 뒤 실행·재시작 시 커널이 **Interrupting** 상태로 멈추거나 무한 로딩될 수 있습니다. 아래 순서대로 시도해 보세요.

### 1. 출력 비우기 후 재시작 (가장 먼저 시도)
- 메뉴: **Cell → Clear All Outputs**
- 그다음 **Kernel → Restart** (또는 재시작 버튼)
- 노트북을 **저장**한 뒤 다시 실행

### 2. 커널을 표준 Python으로 변경
- 상태바에서 **"Kernel distillation (Python 3.10.19)"** 클릭
- **"Python 3.10.x"** 같은 기본 Python 커널 선택
- 커스텀/가상환경 커널이 꼬였을 때 효과적

### 3. Cursor 완전히 재시작
- Cursor 종료 후 다시 실행
- 노트북만 닫았다 열기

### 4. 스크립트로 실행 (가장 안정적)
- 이 프로젝트는 각 파트에 대응하는 **`src/*.py`** 가 있습니다.
- 노트북 대신 터미널에서:
  ```bash
  cd pruning
  python src/01_concept_and_baseline.py   # Part 1
  python src/02_unstructured_pruning.py   # Part 2
  python src/03_structured_pruning.py     # Part 3
  python src/04_experiment_analysis.py    # Part 4
  ```
- 에이전트가 노트북과 동일 로직을 `src/`에도 반영해 두었을 수 있으므로, 수정 내용은 해당 `.py`에서 확인·실행하는 것을 권장합니다.

### 5. 노트북 파일이 손상된 경우
- Git으로 이전 버전으로 되돌리거나, `src/*.py`를 기준으로 새 노트북을 만드는 방법을 고려

---

## 참고 논문

- [Han et al., 2015 - Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626)
- [Li et al., 2017 - Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)
- [Frankle & Carlin, 2019 - The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
