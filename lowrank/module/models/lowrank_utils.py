"""
Low-Rank Approximation 유틸리티

가중치 행렬을 SVD(Singular Value Decomposition)를 이용하여
더 작은 행렬의 곱으로 분해하는 도구들을 제공합니다.

이 모듈은 다음을 정의합니다:
1. SVDLinear: Linear 레이어를 SVD로 분해한 모듈
2. ChannelDecomposedConv2d: Conv2d 레이어를 채널 분해한 모듈
3. Rank 선택 유틸리티: 고정 비율, 에너지 기반
4. 모델 분해 함수: 지정된 레이어를 자동으로 교체

SVD 분해 원리:
=============
행렬 W ∈ ℝ^(m×n) 에 대해:
  W = U @ Σ @ V^T    (full rank)
  W_r ≈ U_r @ Σ_r @ V_r^T  (rank-r 근사)

여기서:
- U_r: m×r (left singular vectors)
- Σ_r: r×r (singular values, 대각행렬)
- V_r: n×r (right singular vectors)

파라미터 절감:
- 원본: m × n
- 분해: r × (m + n)
- 조건: r < mn/(m+n) 일 때 압축 효과
"""

import copy

import torch
import torch.nn as nn


# =============================================================================
# 1. SVDLinear: Linear 레이어 분해
# =============================================================================

class SVDLinear(nn.Module):
    """
    SVD로 분해된 Linear 레이어

    원본: Linear(in_features, out_features)
    분해: Linear(in_features, rank) → Linear(rank, out_features)

    파라미터:
    - 원본: in_features × out_features
    - 분해: rank × (in_features + out_features)

    Args:
        in_features: 입력 차원
        out_features: 출력 차원
        rank: 분해 rank (작을수록 더 많은 압축)
        bias: bias 사용 여부
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int, bias: bool = True):
        super(SVDLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # 두 개의 연속된 Linear 레이어
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.linear1(x))

    @staticmethod
    def from_pretrained(original_layer: nn.Linear, rank: int) -> 'SVDLinear':
        """
        학습된 Linear 레이어를 SVD로 분해하여 초기화

        W = U @ S @ V^T 에서 rank-r truncation 후:
        - linear1.weight = V_r^T (rank × in)
        - linear2.weight = U_r @ diag(S_r) (out × rank)

        Args:
            original_layer: 원본 Linear 레이어
            rank: 분해 rank

        Returns:
            SVDLinear: SVD로 초기화된 분해 레이어
        """
        W = original_layer.weight.data  # (out_features, in_features)
        has_bias = original_layer.bias is not None

        # SVD 분해: W = U @ diag(S) @ V^T
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        # rank-r truncation
        U_r = U[:, :rank]           # (out, rank)
        S_r = S[:rank]              # (rank,)
        Vt_r = Vt[:rank, :]         # (rank, in)

        # 두 factor로 분리:
        # W ≈ (U_r @ diag(S_r)) @ Vt_r
        factor1 = Vt_r                          # (rank, in) → linear1.weight
        factor2 = U_r @ torch.diag(S_r)         # (out, rank) → linear2.weight

        # SVDLinear 생성 및 초기화
        decomposed = SVDLinear(
            W.shape[1], W.shape[0], rank, bias=has_bias
        )
        decomposed.linear1.weight.data = factor1
        decomposed.linear2.weight.data = factor2

        if has_bias:
            decomposed.linear2.bias.data = original_layer.bias.data.clone()

        return decomposed

    def original_param_count(self) -> int:
        """원본 레이어의 파라미터 수"""
        bias_count = self.out_features if self.linear2.bias is not None else 0
        return self.in_features * self.out_features + bias_count

    def decomposed_param_count(self) -> int:
        """분해된 레이어의 파라미터 수"""
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# 2. ChannelDecomposedConv2d: Conv 레이어 분해
# =============================================================================

class ChannelDecomposedConv2d(nn.Module):
    """
    채널 분해된 Conv2d 레이어

    원본: Conv2d(in_c, out_c, k×k)
    분해: Conv2d(in_c, rank, 1×1) → Conv2d(rank, out_c, k×k)

    원리:
    - 1×1 conv로 입력 채널을 rank 차원으로 축소 (pointwise)
    - k×k conv로 공간 필터링 수행 (spatial)
    - 가중치 텐서를 채널 축으로 SVD 분해한 것과 동등

    파라미터:
    - 원본: out_c × in_c × k × k
    - 분해: (in_c × rank) + (rank × out_c × k × k)

    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        kernel_size: 커널 크기
        rank: 분해 rank
        padding: 패딩
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, rank: int, padding: int = 0):
        super(ChannelDecomposedConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rank = rank

        # 1×1 pointwise: 채널 축소
        self.conv_pointwise = nn.Conv2d(
            in_channels, rank, kernel_size=1, bias=False
        )
        # k×k spatial: 공간 필터링
        self.conv_spatial = nn.Conv2d(
            rank, out_channels, kernel_size=kernel_size,
            padding=padding, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pointwise(x)
        x = self.conv_spatial(x)
        return x

    @staticmethod
    def from_pretrained(original_layer: nn.Conv2d, rank: int) -> 'ChannelDecomposedConv2d':
        """
        학습된 Conv2d 레이어를 채널 분해하여 초기화

        가중치 텐서 W: (out_c, in_c, k, k)를
        (out_c * k * k, in_c)로 reshape → SVD → truncate → split

        Args:
            original_layer: 원본 Conv2d 레이어
            rank: 분해 rank

        Returns:
            ChannelDecomposedConv2d: 분해된 레이어
        """
        W = original_layer.weight.data  # (out_c, in_c, k, k)
        out_c, in_c, k_h, k_w = W.shape
        has_bias = original_layer.bias is not None
        padding = original_layer.padding[0]

        # (out_c, in_c, k, k) → (out_c * k * k, in_c)
        W_reshaped = W.reshape(out_c * k_h * k_w, in_c)

        # SVD 분해
        U, S, Vt = torch.linalg.svd(W_reshaped, full_matrices=False)

        # rank-r truncation
        U_r = U[:, :rank]           # (out_c*k*k, rank)
        S_r = S[:rank]              # (rank,)
        Vt_r = Vt[:rank, :]         # (rank, in_c)

        # Factor 분리:
        # pointwise weights: Vt_r → (rank, in_c, 1, 1)
        factor_pointwise = Vt_r.reshape(rank, in_c, 1, 1)

        # spatial weights: U_r @ diag(S_r) → (out_c*k*k, rank) → (out_c, rank, k, k)
        factor_spatial = (U_r @ torch.diag(S_r)).reshape(out_c, rank, k_h, k_w)

        # ChannelDecomposedConv2d 생성 및 초기화
        decomposed = ChannelDecomposedConv2d(
            in_c, out_c, k_h, rank, padding=padding
        )
        decomposed.conv_pointwise.weight.data = factor_pointwise
        decomposed.conv_spatial.weight.data = factor_spatial

        if has_bias:
            decomposed.conv_spatial.bias.data = original_layer.bias.data.clone()

        return decomposed

    def original_param_count(self) -> int:
        """원본 레이어의 파라미터 수"""
        bias_count = self.out_channels if self.conv_spatial.bias is not None else 0
        return (self.out_channels * self.in_channels *
                self.kernel_size * self.kernel_size + bias_count)

    def decomposed_param_count(self) -> int:
        """분해된 레이어의 파라미터 수"""
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# 3. Rank 선택 유틸리티
# =============================================================================

def select_rank_by_ratio(weight: torch.Tensor, ratio: float = 0.5) -> int:
    """
    고정 압축 비율로 rank 선택

    r(m + n) / (m * n) ≤ ratio 를 만족하는 최대 rank

    Args:
        weight: 2D 가중치 텐서 (m × n)
        ratio: 목표 압축 비율 (0.5 = 50% 파라미터 유지)

    Returns:
        선택된 rank
    """
    m, n = weight.shape
    max_rank = int(ratio * m * n / (m + n))
    max_rank = max(1, min(max_rank, min(m, n)))
    return max_rank


def select_rank_by_energy(weight: torch.Tensor, threshold: float = 0.95) -> int:
    """
    에너지 기반 rank 선택

    특이값의 누적 에너지가 threshold 이상이 되는 최소 rank 선택
    Energy = cumsum(σ²) / sum(σ²)

    Args:
        weight: 2D 가중치 텐서
        threshold: 에너지 보존 임계값 (0.95 = 95% 에너지 보존)

    Returns:
        선택된 rank
    """
    _, S, _ = torch.linalg.svd(weight, full_matrices=False)

    # 누적 에너지 계산
    energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)

    # threshold 이상인 최소 rank
    rank = (energy < threshold).sum().item() + 1
    rank = min(rank, len(S))

    return rank


# =============================================================================
# 4. 모델 분해 함수
# =============================================================================

def decompose_model_linear(model: nn.Module, rank_dict: dict) -> nn.Module:
    """
    모델의 지정된 Linear 레이어를 SVDLinear로 교체

    nn.Sequential 내부의 레이어를 인덱스 기반으로 교체합니다.

    Args:
        model: 원본 모델 (deepcopy 후 수정)
        rank_dict: {layer_path: rank} 매핑
            예: {'classifier.0': 128}

    Returns:
        분해된 모델
    """
    model = copy.deepcopy(model)

    for layer_path, rank in rank_dict.items():
        parts = layer_path.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        layer_name = parts[-1]
        original_layer = parent[int(layer_name)] if isinstance(parent, nn.Sequential) else getattr(parent, layer_name)

        if not isinstance(original_layer, nn.Linear):
            raise ValueError(f"{layer_path} is not a Linear layer")

        decomposed = SVDLinear.from_pretrained(original_layer, rank)

        if isinstance(parent, nn.Sequential):
            parent[int(layer_name)] = decomposed
        else:
            setattr(parent, layer_name, decomposed)

    return model


def decompose_model_conv(model: nn.Module, rank_dict: dict) -> nn.Module:
    """
    모델의 지정된 Conv2d 레이어를 ChannelDecomposedConv2d로 교체

    Args:
        model: 원본 모델 (deepcopy 후 수정)
        rank_dict: {layer_path: rank} 매핑
            예: {'features.2': 32}

    Returns:
        분해된 모델
    """
    model = copy.deepcopy(model)

    for layer_path, rank in rank_dict.items():
        parts = layer_path.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        layer_name = parts[-1]
        original_layer = parent[int(layer_name)] if isinstance(parent, nn.Sequential) else getattr(parent, layer_name)

        if not isinstance(original_layer, nn.Conv2d):
            raise ValueError(f"{layer_path} is not a Conv2d layer")

        decomposed = ChannelDecomposedConv2d.from_pretrained(original_layer, rank)

        if isinstance(parent, nn.Sequential):
            parent[int(layer_name)] = decomposed
        else:
            setattr(parent, layer_name, decomposed)

    return model


# =============================================================================
# 5. 복원 오차 계산
# =============================================================================

def get_reconstruction_error(original_weight: torch.Tensor, rank: int) -> float:
    """
    rank-r SVD 근사의 상대적 복원 오차 계산

    Error = ||W - W_r||_F / ||W||_F

    Args:
        original_weight: 원본 가중치 (2D 텐서)
        rank: 분해 rank

    Returns:
        상대적 Frobenius norm 오차
    """
    U, S, Vt = torch.linalg.svd(original_weight, full_matrices=False)

    # rank-r 근사 복원
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]
    W_approx = U_r @ torch.diag(S_r) @ Vt_r

    # 상대적 오차
    error = torch.norm(original_weight - W_approx) / torch.norm(original_weight)
    return error.item()
