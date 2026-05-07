# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of IQN Cost Critic for distributional safety critic."""

from __future__ import annotations

import torch
import torch.nn as nn

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace


class IQNCostCritic(Critic):
    """IQN-based cost critic that models the full quantile function of cost returns.

    Unlike the Gaussian cost critic which outputs (mu, var), this critic takes
    (obs, act, tau) as input and outputs the tau-th quantile value of the cost-return
    distribution. The quantile fraction tau is embedded via cosine features and fused
    with (obs, act) features through Hadamard product.

    Args:
        obs_space: Observation space.
        act_space: Action space.
        hidden_sizes: Hidden layer sizes for the quantile MLP.
        activation: Activation function. Default ``'relu'``.
        weight_initialization_mode: Weight init mode. Default ``'kaiming_uniform'``.
        embedding_dim: Dimension of cosine embedding for tau. Default 64.
    """

    _embedding_dim: int
    _obs_act_fc: nn.Linear
    _quantile_mlp: nn.Sequential

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        embedding_dim: int = 64,
    ) -> None:
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes,
            activation,
            weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
        )
        self._embedding_dim = embedding_dim

        # 将 (obs, act) 投影到 embedding_dim 维度，用于 Hadamard 积
        self._obs_act_fc = nn.Linear(
            self._obs_dim + self._act_dim, embedding_dim
        )

        # 分位数 MLP: embedding_dim -> hidden -> 1
        layers: list[nn.Module] = []
        in_dim = embedding_dim
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self._quantile_mlp = nn.Sequential(*layers)

        self._init_weights(weight_initialization_mode)

    def _init_weights(self, mode: InitFunction) -> None:
        """Initialize weights, with small uniform init for the final layer."""
        for module in self._quantile_mlp:
            if isinstance(module, nn.Linear):
                if mode == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif mode == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        # 最后一层小范围均匀初始化，稳定训练初期
        last_linear = self._quantile_mlp[-1]
        nn.init.uniform_(last_linear.weight, -1e-3, 1e-3)
        if last_linear.bias is not None:
            nn.init.constant_(last_linear.bias, 0.0)

    def _compute_cosine_embedding(self, tau: torch.Tensor) -> torch.Tensor:
        """Compute cosine embedding for quantile fractions.

        Args:
            tau: Quantile fractions of shape ``[B, N]``, values in [0, 1].

        Returns:
            Cosine embedding of shape ``[B, N, embedding_dim]``.
        """
        # tau: [B, N] -> [B, N, 1]
        tau_expanded = tau.unsqueeze(-1)
        # i * pi for i = 1..embedding_dim, shape [1, 1, embedding_dim]
        i_pi = (
            torch.arange(1, self._embedding_dim + 1, device=tau.device, dtype=tau.dtype)
            * torch.pi
        )
        # cos(i * pi * tau), shape [B, N, embedding_dim]
        return torch.cos(tau_expanded * i_pi.unsqueeze(0).unsqueeze(0))

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: compute quantile values for given tau.

        Args:
            obs: Observation tensor of shape ``[B, obs_dim]``.
            act: Action tensor of shape ``[B, act_dim]``.
            tau: Quantile fractions of shape ``[B, N]``, values in [0, 1].

        Returns:
            Quantile values of shape ``[B, N, 1]``.
        """
        # 1. 将 (obs, act) 投影为特征向量
        x = torch.cat([obs, act], dim=-1)
        phi_x = self._obs_act_fc(x)
        phi_x = phi_x.unsqueeze(1)                       # [B, 1, embedding_dim]

        # 2. 计算 tau 的余弦嵌入
        cos_embed = self._compute_cosine_embedding(tau)   # [B, N, embedding_dim]

        # 3. Hadamard 积融合
        merged = phi_x * cos_embed                        # [B, N, embedding_dim]

        # 4. MLP 输出分位数值
        quantile_values = self._quantile_mlp(merged)      # [B, N, 1]

        return quantile_values
