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
"""Actor-critic container for the Gaussian WCSAC implementation."""

from copy import deepcopy

from torch import optim

from omnisafe.models.actor_critic.actor_q_critic import ActorQCritic
from omnisafe.models.base import Critic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig


class WCSACActorQCritic(ActorQCritic):
    """WCSAC model with independent cost-mean and cost-variance critics."""

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
    ) -> None:
        """Initialize actor, reward critics, and the two cost critics."""
        super().__init__(obs_space, act_space, model_cfgs, epochs)

        def build_cost_critic() -> Critic:
            return CriticBuilder(
                obs_space=obs_space,
                act_space=act_space,
                hidden_sizes=model_cfgs.critic.hidden_sizes,
                activation=model_cfgs.critic.activation,
                weight_initialization_mode=model_cfgs.weight_initialization_mode,
                num_critics=1,
                use_obs_encoder=False,
            ).build_critic('q')

        self.cost_critic = build_cost_critic()
        self.cost_var_critic = build_cost_critic()
        self.target_cost_critic = deepcopy(self.cost_critic)
        self.target_cost_var_critic = deepcopy(self.cost_var_critic)

        for critic in (self.target_cost_critic, self.target_cost_var_critic):
            for param in critic.parameters():
                param.requires_grad = False

        if model_cfgs.critic.lr is not None:
            self.cost_critic_optimizer = optim.Adam(
                self.cost_critic.parameters(),
                lr=model_cfgs.critic.lr,
            )
            self.cost_var_critic_optimizer = optim.Adam(
                self.cost_var_critic.parameters(),
                lr=model_cfgs.critic.lr,
            )

    def polyak_update(self, tau: float) -> None:
        """Update all WCSAC target networks."""
        super().polyak_update(tau)
        for source, target in (
            (self.cost_critic, self.target_cost_critic),
            (self.cost_var_critic, self.target_cost_var_critic),
        ):
            for param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
