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

        if self._cfgs.model_cfgs.critic.lr is not None:
            self._actor_critic.cost_critic_optimizer = optim.Adam(
                self._actor_critic.cost_critic.parameters(),
                lr=self._cfgs.model_cfgs.critic.lr,
            )

    # ==================== Quantile Huber Loss ====================

    def _quantile_huber_loss(
        self,
        td_error: torch.Tensor,
        tau: torch.Tensor,
        kappa: float = 1.0,
    ) -> torch.Tensor:
        """Compute quantile Huber loss.

        rho_tau(delta) = |tau - I(delta < 0)| * Huber(delta) / kappa

        Args:
            td_error: TD error of shape ``[B, N, N']`` where delta_{i,j} = target_j - current_i.
            tau: Quantile fractions for current network, shape ``[B, N]``.
            kappa: Huber loss threshold.

        Returns:
            Scalar loss.
        """
        # tau: [B, N] -> [B, N, 1] for broadcasting with td_error [B, N, N']
        tau_expanded = tau.unsqueeze(-1)

        # Huber 损失
        abs_error = td_error.abs()
        huber = torch.where(
            abs_error <= kappa,
            0.5 * td_error.pow(2),
            kappa * (abs_error - 0.5 * kappa),
        )

        # 分位数权重: |tau - I(delta < 0)|
        quantile_weight = (tau_expanded - (td_error.detach() < 0).float()).abs()

        return (quantile_weight * huber).mean() / kappa

    # ==================== Cost Critic 更新 ====================

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update IQN cost critic using quantile regression.

        Samples N quantile fractions tau ~ U(0,1), computes target quantile
        values via the distributional Bellman operator, and updates the critic
        with the quantile Huber loss.
        """
        batch_size = obs.shape[0]
        n_quantiles = self._cfgs.algo_cfgs.get('iqn_n_quantiles', 32)
        iqn_kappa = self._cfgs.algo_cfgs.get('iqn_kappa', 1.0)
        device = obs.device

        if cost.dim() == 1:
            cost = cost.unsqueeze(-1)
        if done.dim() == 1:
            done = done.unsqueeze(-1)

        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=False)

            # 采样 target network 的 tau: [B, N]
            next_tau = torch.rand(batch_size, n_quantiles, device=device)

            next_quantiles = self._actor_critic.target_cost_critic(
                next_obs, next_action, next_tau
            )  # [B, N, 1]

            # 分布 Bellman operator: T^pi Z(s,a) = c + gamma * (1-d) * Z(s',a')
            target_quantiles = (
                cost.unsqueeze(1)
                + self._cfgs.algo_cfgs.gamma * (1 - done.unsqueeze(1)) * next_quantiles
            )  # [B, N, 1]

        # 采样当前网络的 tau: [B, N]
        tau = torch.rand(batch_size, n_quantiles, device=device)

        current_quantiles = self._actor_critic.cost_critic(
            obs, action, tau
        )  # [B, N, 1]

        if not torch.isfinite(current_quantiles).all():
            raise RuntimeError('cost_critic outputs NaN/Inf')

        # 计算阻尼项 (基于 real actions 的 sampled CVaR 与 cost_limit 的差值)
        damp_scale = float(self._cfgs.algo_cfgs.get('damp_scale', 10.0))
        cost_limit = float(self._cfgs.lagrange_cfgs.cost_limit)
        cvar_samples = self._cfgs.algo_cfgs.get('cvar_quantile_samples', 32)
        alpha = float(self._cfgs.lagrange_cfgs.cvar_alpha)
        with torch.no_grad():
            tau_cvar_real = torch.rand(batch_size, cvar_samples, device=device) * alpha
            cvar_real = self._actor_critic.cost_critic(obs, action, tau_cvar_real).mean(dim=1)
        self._damp = damp_scale * (cost_limit - cvar_real.mean()).item()

        # 构建 TD error: delta_{i,j} = target_j - current_i
        # target: [B, N', 1] -> [B, 1, N']; current: [B, N, 1]
        td_error = (
            target_quantiles.squeeze(-1).unsqueeze(1)
            - current_quantiles.squeeze(-1).unsqueeze(-1)
        )  # [B, N, N']

        loss = self._quantile_huber_loss(td_error, tau, kappa=iqn_kappa)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.cost_critic_optimizer.step()

        # IQN 专有统计
        quantiles_sq = current_quantiles.squeeze(-1)  # [B, N]
        iqn_quantile_mean = quantiles_sq.mean().item()
        iqn_quantile_span = (quantiles_sq.max(dim=1).values - quantiles_sq.min(dim=1).values).mean().item()
        iqn_quantile_std = quantiles_sq.std(dim=1).mean().item()

        self._logger.store(
            {
                'Loss/Loss_cost_critic': loss.item(),
                'Value/cost_critic': current_quantiles.mean().item(),
                'Value/damp': self._damp,
                'Value/iqn_quantile_mean': iqn_quantile_mean,
                'Value/iqn_quantile_span': iqn_quantile_span,
                'Value/iqn_quantile_std': iqn_quantile_std,
            },
        )

    # ==================== CVaR 采样计算 ====================

    def _compute_cvar(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate CVaR_alpha by sampling tau ~ U(0, alpha).

        CVaR_alpha = (1/alpha) * integral_{0}^{alpha} Z(tau) d tau
                   ≈ (1/K) * sum_{k=1}^{K} Z(tau_k),  tau_k ~ U(0, alpha)

        Args:
            obs: Observation tensor of shape ``[B, obs_dim]``.
            act: Action tensor of shape ``[B, act_dim]``.

        Returns:
            CVaR estimate of shape ``[B, 1]``.
        """
        batch_size = obs.shape[0]
        cvar_samples = self._cfgs.algo_cfgs.get('cvar_quantile_samples', 32)
        alpha = float(self._cfgs.lagrange_cfgs.cvar_alpha)
        device = obs.device

        tau_cvar = torch.rand(batch_size, cvar_samples, device=device) * alpha

        quantiles = self._actor_critic.cost_critic(obs, act, tau_cvar)  # [B, K, 1]
        cvar = quantiles.mean(dim=1)  # [B, 1]

        return cvar

    # ==================== Actor Loss ====================

    def _loss_pi(self, obs: torch.Tensor) -> torch.Tensor:
        r"""Compute actor loss for WCSAC-IQN.

        L = E[ alpha * log pi(a|s) - min(Qr1, Qr2)(s,a) + lambda * CVaR(s,a) ]

        where CVaR is estimated by sampling quantile fractions from U(0, alpha).
        """
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)

        q1_value_r, q2_value_r = self._actor_critic.reward_critic(obs, action)
        loss_entropy = self._alpha * log_prob
        loss_reward = -torch.min(q1_value_r, q2_value_r)

        loss_cost = torch.zeros_like(loss_reward)

        if self._cfgs.algo_cfgs.use_cost:
            cvar = self._compute_cvar(obs, action)

            if torch.isfinite(cvar).all():
                loss_cost = (self._lagrange_multiplier - self._damp) * cvar.squeeze(-1)

        total_loss = (loss_entropy + loss_reward + loss_cost).mean()

        return total_loss

    # ==================== Lagrange 乘子更新 ====================

    def _update_lagrange_multiplier_wccvar(self, obs: torch.Tensor) -> None:
        """Update Lagrange multiplier based on CVaR constraint violation.

        Uses sampled CVaR from IQN critic (instead of closed-form Gaussian CVaR).
        """
        cost_limit = float(self._cfgs.lagrange_cfgs.cost_limit)
        lambda_lr = float(self._cfgs.lagrange_cfgs.lambda_lr)

        with torch.no_grad():
            action = self._actor_critic.actor.predict(obs, deterministic=False)
            cvar = self._compute_cvar(obs, action)

            if not torch.isfinite(cvar).all():
                self._logger.store(
                    {
                        'Value/lagrange_multiplier': self._lagrange_multiplier.item(),
                        'Value/wccvar': 0.0,
                    },
                )
                return

            if cvar.dim() == 2:
                cvar = cvar.squeeze(-1)

            J = cvar.mean()
            gap = J - cost_limit

            self._lagrange_multiplier = torch.clamp(
                self._lagrange_multiplier + lambda_lr * gap,
                min=0.0,
                max=self._LAMBDA_MAX,
            )

            self._logger.store(
                {
                    'Value/lagrange_multiplier': self._lagrange_multiplier.item(),
                    'Value/wccvar': J.item(),
                },
            )

    def _log_lagrange_when_warmup(self, obs: torch.Tensor) -> None:
        """Log Lagrange-related metrics during warmup (without updating lambda)."""
        with torch.no_grad():
            action = self._actor_critic.actor.predict(obs, deterministic=False)

            cvar_val = 0.0
            cvar = self._compute_cvar(obs, action)
            if torch.isfinite(cvar).all():
                cvar_val = cvar.mean().item()

            self._logger.store(
                {
                    'Value/lagrange_multiplier': self._lagrange_multiplier.item(),
                    'Value/wccvar': cvar_val,
                },
            )

    # ==================== 日志 ====================

    def _init_log(self) -> None:
        """Register IQN-specific logging keys (no Gaussian cost_mean/cost_var)."""
        SAC._init_log(self)
        self._logger.register_key('Value/wccvar')
        self._logger.register_key('Value/lagrange_multiplier')
        self._logger.register_key('Value/damp')
        self._logger.register_key('Value/iqn_quantile_mean')
        self._logger.register_key('Value/iqn_quantile_span')
        self._logger.register_key('Value/iqn_quantile_std')

    def _log_when_not_update(self) -> None:
        """Log default values when not updating (no Gaussian keys)."""
        SAC._log_when_not_update(self)
        self._logger.store(
            {
                'Value/wccvar': 0.0,
                'Value/lagrange_multiplier': self._lagrange_multiplier.item(),
                'Value/damp': 0.0,
                'Value/iqn_quantile_mean': 0.0,
                'Value/iqn_quantile_span': 0.0,
                'Value/iqn_quantile_std': 0.0,
            },
        )
