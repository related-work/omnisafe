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
"""Implementation of the Lagrange version of the PPO algorithm."""

import csv
import os

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.common.lagrange import Lagrange
from omnisafe.utils import distributed


@registry.register
class PPOLag(PPO):
    """The Lagrange version of the PPO algorithm.

    A simple combination of the Lagrange method and the Proximal Policy Optimization algorithm.
    """

    def _init(self) -> None:
        """Initialize the PPOLag specific model.

        The PPOLag algorithm uses a Lagrange multiplier to balance the cost and reward.
        """
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)
        self._rollout_csv_path: str = ''
        self._rollout_csv_initialized: bool = False

    def _init_log(self) -> None:
        """Log the PPOLag specific information.

        +----------------------------+--------------------------+
        | Things to log              | Description              |
        +============================+==========================+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier. |
        +----------------------------+--------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

        rank = distributed.get_rank()
        filename = 'rollout_buffer.csv'
        if distributed.world_size() > 1:
            filename = f'rollout_buffer_rank{rank}.csv'
        self._rollout_csv_path = os.path.join(self._logger.log_dir, filename)

    def _update(self) -> None:
        r"""Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the Lagrange multiplier parameter by calling the
        :meth:`update_lagrange_multiplier` method.

        .. note::
            The :meth:`_loss_pi` is defined in the :class:`PolicyGradient` algorithm. When a
            lagrange multiplier is used, the :meth:`_loss_pi` method will return the loss of the
            policy as:

            .. math::

                L_{\pi} = -\underset{s_t \sim \rho_{\theta}}{\mathbb{E}} \left[
                    \frac{\pi_{\theta} (a_t|s_t)}{\pi_{\theta}^{old}(a_t|s_t)}
                    [ A^{R}_{\pi_{\theta}} (s_t, a_t) - \lambda A^{C}_{\pi_{\theta}} (s_t, a_t) ]
                \right]

            where :math:`\lambda` is the Lagrange multiplier parameter.
        """
        self._save_rollout_to_csv()

        # note that logger already uses MPI statistics across all processes..
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost for updating lagrange multiplier is nan'
        # first update Lagrange multiplier parameter
        self._lagrange.update_lagrange_multiplier(Jc)
        # then update the policy and value function
        super()._update()

        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})

    def _save_rollout_to_csv(self) -> None:
        """Save the current rollout buffer to a CSV file before buffer reset."""
        os.makedirs(self._logger.log_dir, exist_ok=True)

        fieldnames = ['rollout_id', 'returns_c', 'timestep', 'step', 'env', 'reward', 'cost']
        rollout_id = self._logger.current_epoch
        num_envs = len(self._buf.buffers)
        rows = []

        for env_idx, buffer in enumerate(self._buf.buffers):
            valid_size = buffer.ptr
            rewards = buffer.data['reward'][:valid_size].detach().cpu()
            costs = buffer.data['cost'][:valid_size].detach().cpu()
            returns_c = buffer.data['target_value_c'][:valid_size].detach().cpu()

            for step_idx in range(valid_size):
                timestep = rollout_id * self._steps_per_epoch * num_envs + step_idx * num_envs + env_idx
                rows.append(
                    {
                        'rollout_id': rollout_id,
                        'returns_c': float(returns_c[step_idx].item()),
                        'timestep': timestep,
                        'step': step_idx,
                        'env': env_idx,
                        'reward': float(rewards[step_idx].item()),
                        'cost': float(costs[step_idx].item()),
                    },
                )

        with open(self._rollout_csv_path, mode='a', encoding='utf-8', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not self._rollout_csv_initialized:
                writer.writeheader()
                self._rollout_csv_initialized = True
            writer.writerows(rows)

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        r"""Compute surrogate loss.

        PPOLag uses the following surrogate loss:

        .. math::

            L = \frac{1}{1 + \lambda} [
                A^{R}_{\pi_{\theta}} (s, a)
                - \lambda A^C_{\pi_{\theta}} (s, a)
            ]

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function combined with reward and cost.
        """
        penalty = self._lagrange.lagrangian_multiplier.item()
        return (adv_r - penalty * adv_c) / (1 + penalty)
