# Low-Rank Approximation 튜토리얼

가중치 행렬을 SVD(Singular Value Decomposition)로 분해하여 모델을 압축하는 Low-Rank Approximation 기법을 학습합니다.

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
lowrank/
├── requirements.txt
├── module/
│   └── models/
│       ├── __init__.py
│       ├── base_model.py          # CNN (~1.18M params) - Baseline
│       └── lowrank_utils.py       # SVDLinear, ChannelDecomposedConv2d, 유틸리티
├── src/
│   ├── 01_concept_and_baseline.py     # Low-Rank 개념 + Baseline 학습
│   ├── 02_linear_svd.py               # Linear 레이어 SVD 분해
│   ├── 03_conv_decomposition.py       # Conv 레이어 채널 분해
│   └── 04_experiment_analysis.py      # 종합 실험 및 시각화
├── data/
└── results/
```

## 실행 순서

### Part 1: Low-Rank 개념 및 Baseline
```bash
python src/01_concept_and_baseline.py
```
- SVD 분해 원리: W = UΣV^T → rank-r truncation
- CNN Baseline 학습 (10 epochs)
- 레이어별 특이값(Singular Value) 분석
- 누적 에너지 곡선으로 적절한 rank 파악
- Rank vs 복원 오차 곡선

### Part 2: Linear 레이어 SVD 분해
```bash
python src/02_linear_svd.py
```
- classifier.0 (2048→512, 전체 파라미터의 88.4%)에 SVD 적용
- 다양한 rank (256, 128, 64, 32)로 실험
- SVD 초기화 vs 랜덤 초기화 비교 → SVD 우위 확인
- Fine-tuning 후 정확도 회복 (5 epochs, lr=0.0001)
- Rank 선택 전략: 고정 비율 vs 에너지 기반 (95%)

### Part 3: Conv 레이어 채널 분해
```bash
python src/03_conv_decomposition.py
```
- 채널 분해: Conv2d(in, out, k) → Conv2d(in, r, 1) + Conv2d(r, out, k)
- features.2 (Conv2d 128→64, 3×3)에 다양한 rank 적용
- Combined: Linear SVD + Conv 분해 동시 적용
- 분해 전/후 Feature Map 비교

### Part 4: 종합 실험 분석
```bash
python src/04_experiment_analysis.py
```
- Baseline vs Linear SVD vs Conv 분해 vs Combined 비교
- 정확도 vs 압축률 Pareto 곡선
- 레이어별 분해 효과 분석 (파라미터 비중, 에너지 rank)
- Rank 선택 전략 비교 (고정 비율 vs 에너지 기반)
- 종합 결과 JSON 및 텍스트 테이블 저장

## 핵심 개념

### SVD 분해
```
W ∈ ℝ^(m×n)에 대해:
  W = U @ Σ @ V^T

Rank-r 근사:
  W_r ≈ U_r @ Σ_r @ V_r^T
  → 파라미터: m×n → r×(m+n)
```

### Linear 레이어 분해
```
Linear(in, out) → Linear(in, rank) + Linear(rank, out)
  원본: in × out
  분해: rank × (in + out)
```

### Conv 레이어 채널 분해
```
Conv2d(in, out, k) → Conv2d(in, rank, 1) + Conv2d(rank, out, k)
  원본: out × in × k × k
  분해: (in × rank) + (rank × out × k × k)
```

### Rank 선택 전략
- **고정 비율**: r ≤ ratio × m×n / (m+n)
- **에너지 기반**: cumsum(σ²) / sum(σ²) ≥ threshold

## 예상 결과

| Method | Accuracy | Parameters | 압축률 |
|--------|----------|-----------|--------|
| Baseline (CNN) | ~74% | ~1.18M | 1.0x |
| Linear SVD (r=256) | ~73% | ~790K | 1.5x |
| Linear SVD (r=128) | ~71% | ~460K | 2.6x |
| Linear SVD (r=64) | ~67% | ~300K | 3.9x |
| Conv 분해 (r=32) | ~72% | ~1.05M | 1.1x |
| Combined (L=128, C=32) | ~69% | ~330K | 3.6x |

## 참고 논문
- Denton et al., 2014 - Exploiting Linear Structure Within Convolutional Networks
- Jaderberg et al., 2014 - Speeding up CNNs with Low Rank Expansions
- Zhang et al., 2015 - Accelerating Very Deep Convolutional Networks
