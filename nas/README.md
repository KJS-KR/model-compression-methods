# Vision NAS (Neural Architecture Search) 튜토리얼

사람이 설계하는 대신, 알고리즘이 최적의 CNN 아키텍처를 자동으로 탐색하는 NAS 기법을 학습합니다.

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
nas/
├── requirements.txt
├── module/
│   └── models/
│       ├── __init__.py
│       ├── base_model.py          # CNN (~1.18M params) - Baseline
│       └── search_space.py        # FlexibleCNN + SEARCH_SPACE + 유틸리티
├── src/
│   ├── 01_concept_and_baseline.py     # NAS 개념 + Baseline 학습
│   ├── 02_random_search.py            # Random Search NAS
│   ├── 03_evolutionary_search.py      # Evolutionary Search NAS
│   └── 04_experiment_analysis.py      # 종합 실험 및 시각화
├── data/
└── results/
```

## 실행 순서

### Part 1: NAS 개념 및 Baseline
```bash
python src/01_concept_and_baseline.py
```
- NAS 3요소: Search Space, Search Strategy, Performance Estimation
- 탐색 공간 분석 (16,384가지 아키텍처)
- CNN Baseline 학습 (10 epochs)
- 샘플 아키텍처 시각화 (파라미터 수 비교, 필터 히트맵)

### Part 2: Random Search
```bash
python src/02_random_search.py
```
- 20개 아키텍처 무작위 샘플링
- 축소 학습(5 epochs)으로 빠른 평가
- 최적 아키텍처 선정 → 전체 학습(10 epochs)
- Baseline과 비교 (정확도, 파라미터 수)

### Part 3: Evolutionary Search
```bash
python src/03_evolutionary_search.py
```
- 초기 인구 10개 생성 및 평가
- Tournament Selection (k=3) + Mutation
- 5세대 진화 (엘리트 2개 유지)
- 세대별 최고/평균 정확도 추적

### Part 4: 종합 분석
```bash
python src/04_experiment_analysis.py
```
- Baseline vs Random vs Evolutionary 종합 비교
- 탐색 효율 비교 (best-so-far 곡선)
- 아키텍처 패턴 분석 (Top 25% vs Bottom 25%)

## 핵심 개념

### NAS의 3가지 요소

| 요소 | 설명 | 이 튜토리얼 |
|------|------|-------------|
| Search Space | 탐색할 아키텍처 공간 | 필터 수, 커널 크기, FC hidden |
| Search Strategy | 탐색 전략 | Random Search, Evolutionary |
| Performance Estimation | 빠른 성능 평가 | Reduced Training (5 epochs) |

### 탐색 공간

```
Conv 필터 수: [16, 32, 64, 128]     → 4가지/레이어
Conv 커널 크기: [3, 5]               → 2가지/레이어
FC Hidden Units: [128, 256, 512, 1024] → 4가지
Conv 레이어: 4개 (고정)
MaxPool 위치: [1, 3] (고정, 0-indexed)

전체 = (4 × 2)^4 × 4 = 16,384 가지 아키텍처
```

### 탐색 전략 비교

```
Random Search:
  탐색 공간 → [랜덤 샘플링 N개] → 축소 학습 → 최적 선택 → 전체 학습

Evolutionary Search:
  초기 인구 → [Tournament Selection] → [Mutation] → 평가
       ↑                                              ↓
       └──────── 엘리트 유지 + 자식 교체 ←─────────────┘
                        (세대 반복)
```

## 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| BATCH_SIZE | 128 | 배치 크기 |
| LEARNING_RATE | 0.001 | 학습률 (Adam) |
| SEARCH_EPOCHS | 5 | 탐색 시 축소 학습 |
| FULL_EPOCHS | 10 | 최종 전체 학습 |
| NUM_SAMPLES | 20 | Random Search 샘플 수 |
| POPULATION_SIZE | 10 | Evolutionary 인구 크기 |
| NUM_GENERATIONS | 5 | 진화 세대 수 |
| TOURNAMENT_SIZE | 3 | 토너먼트 선택 크기 |
| ELITE_SIZE | 2 | 엘리트 유지 수 |

## 예상 실행 시간 (GPU 기준)

| 스크립트 | 시간 | 비고 |
|---------|------|------|
| 01_concept_and_baseline.py | ~2분 | Baseline 학습 |
| 02_random_search.py | ~17분 | 20개 × 5 epochs + 재학습 |
| 03_evolutionary_search.py | ~40분 | 10 + 5세대 × 8 children |
| 04_experiment_analysis.py | ~60분(전체) / ~5분(캐시) | 종합 분석 |

## 참고 논문

- Zoph & Le, 2017 - Neural Architecture Search with Reinforcement Learning
- Real et al., 2019 - Regularized Evolution for Image Classifier Architecture Search
- Li & Talwalkar, 2020 - Random Search and Reproducibility for NAS
