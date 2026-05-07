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
"""Implementation of WCSAC-IQN: Worst-Case SAC with IQN distributional safety critic."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.algorithms.off_policy.wcsac import WCSAC
from omnisafe.models.critic.iqn_cost_critic import IQNCostCritic


@registry.register
class WCSAC_IQN(WCSAC):
    """WCSAC with IQN distributional safety critic.

    Replaces the Gaussian CVaR approximation of WCSAC with an Implicit Quantile
    Network (IQN) that learns the full quantile function of the cost-return
    distribution. CVaR is estimated by sampling quantile fractions from U(0, alpha)
    and averaging the corresponding quantile values.

    References:
        - WCSAC-IQN: Safety-constrained reinforcement learning with a distributional
          safety critic (Yang et al., 2023, Machine Learning)
    """

    def _init_model(self) -> None:
        """Initialize actor, reward critics (from SAC), and IQN cost critic."""
        # 复用 SAC 的 actor + 双 reward critic 初始化
        SAC._init_model(self)

        # 用 IQNCostCritic 替换默认的 cost_critic
        iqn_embedding_dim = self._cfgs.model_cfgs.critic.get('iqn_embedding_dim', 64)

        self._actor_critic.cost_critic = IQNCostCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            hidden_sizes=self._cfgs.model_cfgs.critic.hidden_sizes,
            activation=self._cfgs.model_cfgs.critic.activation,
            weight_initialization_mode=self._cfgs.model_cfgs.weight_initialization_mode,
            embedding_dim=iqn_embedding_dim,
        ).to(self._device)

        self._actor_critic.target_cost_critic = deepcopy(self._actor_critic.cost_critic)
        for param in self._actor_critic.target_cost_critic.parameters():
            param.requires_grad = False

        self._actor_critic.add_module('cost_critic', self._actor_critic.cost_critic)
        self._actor_critic.add_module('target_cost_critic', self._actor_critic.target_cost_critic)

        if self._cfgs.model_cfgs.critic.lr is not None:
            self._actor_critic.cost_critic_optimizer = optim.Adam(
                self._actor_critic.cost_critic.parameters(),
                lr=self._cfgs.model_cfgs.critic.lr,
            )
