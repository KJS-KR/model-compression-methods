# Vision Knowledge Distillation 튜토리얼

Knowledge Distillation을 통해 큰 Teacher 모델의 지식을 작은 Student 모델로 전달하는 방법을 학습합니다.

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
06_distillation/
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── teacher.py          # DeepNN (~1.18M params)
│   └── student.py          # LightNN (~267K params)
├── 01_concept_and_baseline.py    # 개념 + Baseline 학습
├── 02_soft_target_kd.py          # Soft Target KD
├── 03_feature_distillation.py    # Cosine Loss + MSE Regressor
└── 04_experiment_analysis.py     # 종합 실험 및 시각화
```

## 실행 순서

### Part 1: 기본 개념 및 Baseline
```bash
python 01_concept_and_baseline.py
```
- Teacher/Student 모델 학습
- 성능 차이 확인 (목표: 이 격차를 KD로 줄이기)

### Part 2: Soft Target Knowledge Distillation
```bash
python 02_soft_target_kd.py
```
- Temperature 파라미터 효과 시각화
- Teacher의 soft targets로 Student 학습

### Part 3: Feature-based Distillation
```bash
python 03_feature_distillation.py
```
- Cosine Similarity Loss
- MSE Feature Map Loss (FitNets)

### Part 4: 종합 실험 및 분석
```bash
python 04_experiment_analysis.py
```
- 모든 방법 비교
- Temperature 튜닝 실험
- 결과 시각화

## 핵심 개념

### Knowledge Distillation (Hinton et al., 2015)

```
L = α * KD_loss + (1-α) * CE_loss
```

- **KD_loss**: Teacher의 soft targets와 Student의 soft predictions 간 KL divergence
- **CE_loss**: 실제 라벨과의 Cross Entropy
- **Temperature (T)**: softmax 분포를 부드럽게 만듦 (T>1)

### 방법별 특징

| 방법 | 사용 정보 | 장점 | 단점 |
|------|----------|------|------|
| Soft Target KD | 출력 logits | 간단, 효과적 | 출력층 정보만 사용 |
| Cosine Loss | Hidden representation | 방향만 맞춤 | 차원 맞춤 필요 |
| MSE Regressor | Feature map | 유연한 변환 학습 | 추가 파라미터 필요 |

## 예상 결과

| Method | Test Accuracy |
|--------|---------------|
| Teacher (DeepNN) | ~74% |
| Student Baseline | ~70% |
| Student + Soft Target KD | ~70.5% |
| Student + Cosine Loss | ~69.5% |
| Student + MSE Regressor | ~71% |

## 참고 논문

- [Hinton et al., 2015 - Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Romero et al., 2015 - FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550)
