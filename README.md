# 🚀 모델 압축 기법 모음

딥러닝의 핵심 **모델 압축 기법**을 실습 예제와 함께 정리한 저장소입니다.
성능을 크게 희생하지 않으면서 신경망을 압축하는 방법을 알아보세요! 💡

---

## 📚 구성 내용

| 기법 | 설명 | 예제 |
|------|------|------|
| 🧹 **프루닝 (Pruning)** | 중요도가 낮은 가중치나 뉴런을 제거하여 모델 크기를 줄이고 추론 속도를 향상 | [노트북 보기](./pruning/) |
| ⚖️ **양자화 (Quantization)** | 가중치를 32비트 부동소수점에서 8비트 또는 이진수로 변환하여 메모리 절약 | [노트북 보기](./quantization/) |
| 🧠 **지식 증류 (Knowledge Distillation)** | 큰 모델(교사)의 지식을 작은 모델(학생)에게 전달하여 학습 | [노트북 보기](./distillation/) |
| 🧬 **신경망 구조 탐색 (NAS)** | 제약 조건 하에서 최적의 아키텍처를 자동으로 탐색 | [노트북 보기](./nas/) |
| 🔻 **저랭크 분해 (Low-Rank Factorization)** | 큰 가중치 행렬을 더 작은 행렬들로 분해 | [노트북 보기](./lowrank/) |
| ⏩ **동적 추론 (Dynamic Inference)** | 입력에 따라 연산량을 동적으로 조절 | [노트북 보기](./dynamic_inference/) |

---

## 🛠️ 사용 방법

저장소를 클론한 후 Jupyter 노트북을 통해 각 기법을 탐색해 보세요:

```bash
git clone https://github.com/your-username/model-compression-lab.git
cd model-compression-lab
