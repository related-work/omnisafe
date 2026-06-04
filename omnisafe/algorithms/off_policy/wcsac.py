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
"""Implementation of the Worst-Case Soft Actor-Critic (WCSAC) algorithm."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.models.actor_critic.disconstraint_actor_q_critic import DisConstraintActorQCritic


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class WCSAC(SAC):
    """Worst-Case Soft Actor-Critic (WCSAC) algorithm.

    This algorithm extends SAC-Lag by modeling the cost-return distribution as a Gaussian
    and using Conditional Value-at-Risk (CVaR) as the risk measure for constraint satisfaction.

    The cost critic outputs (mu, var) where:
    - mu: mean of cost Q-value
    - var: variance of cost Q-value (ensuring non-negative via softplus)

    CVaR for Gaussian: CVaR_alpha = mu + sigma * phi(Phi^{-1}(alpha)) / (1 - alpha)

    References:
        - WCSAC: Worst-Case Soft Actor Critic for Safety-Constrained Reinforcement Learning
        - SAC: Soft Actor-Critic (Haarnoja et al., 2018)
    """

    # 类内常量
    _LAMBDA_MAX: float = 1000.0

    _lagrange_multiplier: torch.Tensor
    _pdf_cdf: float  # CVaR 系数: phi(Phi^{-1}(alpha)) / (1 - alpha)
    _damp: float  # 阻尼项, 基于 real actions 的 CVaR 与 cost_limit 的差值

    def _init_model(self) -> None:
        """Initialize the model.

        Uses DisConstraintActorQCritic with:
        - reward_critic: dual Q-networks (num_critics=2) for SAC
        - cost_critic: single Q-network with 2-dim output (mu, var_raw) for Gaussian
        """
        self._cfgs.model_cfgs.critic['num_critics'] = 2

        self._actor_critic = DisConstraintActorQCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)

    def _init(self) -> None:
        """Initialize algorithm-specific components."""
        super()._init()

        # 初始化 Lagrange 乘子
        self._lagrange_multiplier = torch.tensor(
            float(self._cfgs.lagrange_cfgs.lagrangian_multiplier_init),
            device=self._device,
        )

        # 计算 CVaR 系数: phi(Phi^{-1}(alpha)) / (1 - alpha)
        # 其中 alpha 是风险水平（如 0.5 表示关注上 50% 的尾部风险）
        alpha = float(self._cfgs.lagrange_cfgs.cvar_alpha)
        normal = Normal(
            loc=torch.tensor(0.0),
            scale=torch.tensor(1.0),
        )
        z_alpha = normal.icdf(torch.tensor(alpha))
        # phi(z_alpha) = pdf of standard normal at z_alpha
        phi_z_alpha = torch.exp(normal.log_prob(z_alpha))
        self._pdf_cdf = (phi_z_alpha / (1.0 - alpha)).item()

        # 阻尼项，基于 real actions 的 CVaR 与 cost_limit 的差值
        self._damp = 0.0

    def _init_log(self) -> None:
        """Register logging keys for WCSAC-specific metrics."""
        super()._init_log()

        self._logger.register_key('Value/wccvar')
        self._logger.register_key('Value/lagrange_multiplier')
        self._logger.register_key('Value/cost_mean')
        self._logger.register_key('Value/cost_var')
        self._logger.register_key('Value/damp')

    # ==================== Gaussian 分布工具函数 ====================

    def _get_cost_dist_params(
        self,
        cost_critic: nn.Module,
        obs: torch.Tensor,
        act: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract Gaussian distribution parameters from cost critic output.

        Args:
            cost_critic: The cost critic network (current or target).
            obs: Observation tensor of shape (B, obs_dim).
            act: Action tensor of shape (B, act_dim).

        Returns:
            mu: Mean of cost Q-value, shape (B, 1).
            var: Variance of cost Q-value, shape (B, 1), always positive.
        """
        q_out = cost_critic(obs, act)[0]  # [B, 2]
        mu = q_out[:, 0:1]  # [B, 1]
        var_raw = q_out[:, 1:2]  # [B, 1]
        var = F.softplus(var_raw) + 1e-8  # 保证正数
        return mu, var

    def _gaussian_cvar(
        self,
        mu: torch.Tensor,
        var: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CVaR of Gaussian distribution.

        For X ~ N(mu, sigma^2), CVaR_alpha(X) = mu + sigma * phi(z_alpha) / (1 - alpha)
        where z_alpha = Phi^{-1}(alpha) and phi is the standard normal PDF.

        Args:
            mu: Mean of shape (B, 1).
            var: Variance of shape (B, 1).

        Returns:
            CVaR of shape (B, 1).
        """
        std = torch.sqrt(var)
        return mu + std * self._pdf_cdf

    # ==================== 主更新循环 ====================

    def _update(self) -> None:
        """Update actor, critic, and Lagrange multiplier."""
        super()._update()

        # 更新 Lagrange 乘子（仅在 warmup 后）
        if self._cfgs.algo_cfgs.use_cost:
            data = self._buf.sample_batch()
            obs = data['obs']

            if self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
                self._update_lagrange_multiplier_wccvar(obs)
            else:
                self._log_lagrange_when_warmup(obs)

    # ==================== Cost Critic 更新 ====================

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update cost critic using Gaussian distribution.

        Updates both mean (qc) and variance (qc_var) of the cost Q-value distribution.

        The variance target is derived from the second moment:
        Var(Q) = E[c² + 2γc·Q' + γ²Q'²] - E[Q]²
               = c² + 2γc·E[Q'] + γ²(Var(Q') + E[Q']²) - E[Q]²

        Loss for variance uses Wasserstein-style distance.
        """
        if cost.dim() == 1:
            cost = cost.unsqueeze(-1)
        if done.dim() == 1:
            done = done.unsqueeze(-1)

        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=False)

            # 从 target cost critic 获取分布参数
            next_mu, next_var = self._get_cost_dist_params(
                self._actor_critic.target_cost_critic,
                next_obs,
                next_action,
            )

            # Mean target: standard Bellman
            target_mu = cost + self._cfgs.algo_cfgs.gamma * (1 - done) * next_mu

            # 获取当前 qc（用于方差 target 计算）
            # 注意：这里需要用当前网络的 mu，而不是 target
            curr_mu, _ = self._get_cost_dist_params(
                self._actor_critic.cost_critic,
                obs,
                action,
            )

            # Variance target (基于二阶矩推导)
            # Var_backup = c² + 2γc·next_mu + γ²·next_var + γ²·next_mu² - curr_mu²
            gamma = self._cfgs.algo_cfgs.gamma
            target_var = (
                cost.pow(2)
                + 2 * gamma * cost * next_mu
                + gamma**2 * next_var
                + gamma**2 * next_mu.pow(2)
                - curr_mu.pow(2)
            )
            target_var = torch.clamp(target_var, min=1e-8)  # 确保正数

        # 当前 cost critic 的分布参数
        mu, var = self._get_cost_dist_params(
            self._actor_critic.cost_critic,
            obs,
            action,
        )

        # 数值检查
        if not torch.isfinite(mu).all() or not torch.isfinite(var).all():
            raise RuntimeError('cost_critic outputs NaN/Inf')

        # Mean loss: MSE
        loss_mu = F.mse_loss(mu, target_mu)

        # Variance loss: Wasserstein-style distance
        # L_var = 0.5 * E[var + target_var - 2 * sqrt(var * target_var)]
        loss_var = 0.5 * (var + target_var - 2 * torch.sqrt(var * target_var)).mean()

        # 计算阻尼项 (基于 real actions 的 CVaR 与 cost_limit 的差值)
        damp_scale = float(self._cfgs.algo_cfgs.get('damp_scale', 10.0))
        cost_limit = float(self._cfgs.lagrange_cfgs.cost_limit)
        cvar_real = self._gaussian_cvar(mu.detach(), var.detach())
        self._damp = damp_scale * (cost_limit - cvar_real.mean()).item()

        loss = loss_mu + loss_var

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

        self._logger.store(
            {
                'Loss/Loss_cost_critic': loss.item(),
                'Value/cost_critic': mu.mean().item(),
                'Value/cost_mean': mu.mean().item(),
                'Value/cost_var': var.mean().item(),
                'Value/damp': self._damp,
            },
        )

    # ==================== Actor Loss ====================

    def _loss_pi(self, obs: torch.Tensor) -> torch.Tensor:
        r"""Compute actor loss for WCSAC.

        L = E[ α * log π(a|s) - min(Q1^r, Q2^r)(s,a) + λ * CVaR(s,a) ]

        where CVaR = mu + sigma * pdf_cdf for Gaussian distribution.
        """
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)

        q1_value_r, q2_value_r = self._actor_critic.reward_critic(obs, action)
        loss_entropy = self._alpha * log_prob
        loss_reward = -torch.min(q1_value_r, q2_value_r)

        loss_cost = torch.zeros_like(loss_reward)

        if self._cfgs.algo_cfgs.use_cost:
            mu, var = self._get_cost_dist_params(
                self._actor_critic.cost_critic,
                obs,
                action,
            )

            if torch.isfinite(mu).all() and torch.isfinite(var).all():
                cvar = self._gaussian_cvar(mu, var)

                if torch.isfinite(cvar).all():
                    loss_cost = (self._lagrange_multiplier - self._damp) * cvar

        total_loss = (loss_entropy + loss_reward + loss_cost).mean()

        return total_loss

    # ==================== Lagrange 乘子更新 ====================

    def _update_lagrange_multiplier_wccvar(self, obs: torch.Tensor) -> None:
        """Update Lagrange multiplier based on CVaR constraint violation."""
        cost_limit = float(self._cfgs.lagrange_cfgs.cost_limit)
        lambda_lr = float(self._cfgs.lagrange_cfgs.lambda_lr)

        with torch.no_grad():
            action = self._actor_critic.actor.predict(obs, deterministic=False)

            mu, var = self._get_cost_dist_params(
                self._actor_critic.cost_critic,
                obs,
                action,
            )

            if not torch.isfinite(mu).all() or not torch.isfinite(var).all():
                self._logger.store(
                    {
                        'Value/lagrange_multiplier': self._lagrange_multiplier.item(),
                        'Value/wccvar': 0.0,
                    },
                )
                return

            cvar = self._gaussian_cvar(mu, var)
            if cvar.dim() == 2:
                cvar = cvar.squeeze(-1)

            if not torch.isfinite(cvar).all():
                self._logger.store(
                    {
                        'Value/lagrange_multiplier': self._lagrange_multiplier.item(),
                        'Value/wccvar': 0.0,
                    },
                )
                return

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
        """Log Lagrange-related metrics during warmup (without updating λ)."""
        with torch.no_grad():
            action = self._actor_critic.actor.predict(obs, deterministic=False)
            mu, var = self._get_cost_dist_params(
                self._actor_critic.cost_critic,
                obs,
                action,
            )

            cvar_val = 0.0
            if torch.isfinite(mu).all() and torch.isfinite(var).all():
                cvar = self._gaussian_cvar(mu, var)
                if torch.isfinite(cvar).all():
                    cvar_val = cvar.mean().item()

            self._logger.store(
                {
                    'Value/lagrange_multiplier': self._lagrange_multiplier.item(),
                    'Value/wccvar': cvar_val,
                },
            )

    # ==================== 其他辅助方法 ====================

    def _log_when_not_update(self) -> None:
        """Log default values when not updating."""
        super()._log_when_not_update()

        self._logger.store(
            {
                'Value/wccvar': 0.0,
                'Value/lagrange_multiplier': self._lagrange_multiplier.item(),
                'Value/cost_mean': 0.0,
                'Value/cost_var': 0.0,
                'Value/damp': 0.0,
            },
        )