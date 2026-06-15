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
from torch import optim
from torch.distributions import Normal
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.models.actor_critic.wcsac_actor_q_critic import WCSACActorQCritic


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class WCSAC(SAC):
    """Worst-Case Soft Actor-Critic (WCSAC) algorithm.

    This algorithm extends SAC-Lag by modeling the cost-return distribution as a Gaussian
    and using Conditional Value-at-Risk (CVaR) as the risk measure for constraint satisfaction.

    The Gaussian cost return is represented by two independent critics:
    - cost critic: mean of the cost return
    - cost variance critic: variance of the cost return

    CVaR for Gaussian: CVaR_alpha = mu + sigma * phi(Phi^{-1}(alpha)) / (1 - alpha)

    References:
        - WCSAC: Worst-Case Soft Actor Critic for Safety-Constrained Reinforcement Learning
        - SAC: Soft Actor-Critic (Haarnoja et al., 2018)
    """

    # 类内常量
    _soft_beta: torch.Tensor
    _beta_optimizer: optim.Optimizer
    _soft_alpha: torch.Tensor
    _alpha_optimizer: optim.Optimizer
    _pdf_cdf: float  # CVaR 系数: phi(Phi^{-1}(alpha)) / (1 - alpha)
    _cost_constraint: float
    _damp: float  # 阻尼项, 基于 real actions 的 CVaR 与 cost_limit 的差值

    def _init_model(self) -> None:
        """Initialize the model.

        The original WCSAC uses separate networks for cost mean and variance.
        """
        self._cfgs.model_cfgs.critic['num_critics'] = 2

        self._actor_critic = WCSACActorQCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)

    def _init(self) -> None:
        """Initialize algorithm-specific components."""
        super()._init()

        # The reference code optimizes unconstrained parameters and maps them
        # through softplus to obtain positive alpha and beta.
        beta_init = float(self._cfgs.lagrange_cfgs.lagrangian_multiplier_init)
        self._soft_beta = torch.tensor(
            self._inverse_softplus(beta_init),
            device=self._device,
            requires_grad=True,
        )
        self._beta_optimizer = optim.Adam(
            [self._soft_beta],
            lr=(
                float(self._cfgs.model_cfgs.actor.lr)
                * float(self._cfgs.algo_cfgs.get('cost_penalty_lr_scale', 50.0))
            ),
        )

        alpha_init = float(self._cfgs.algo_cfgs.alpha)
        self._soft_alpha = torch.tensor(
            self._inverse_softplus(alpha_init),
            device=self._device,
            requires_grad=True,
        )
        assert self._cfgs.model_cfgs.actor.lr is not None
        self._alpha_optimizer = optim.Adam(
            [self._soft_alpha],
            lr=float(self._cfgs.model_cfgs.actor.lr),
        )
        self._target_entropy = -torch.prod(torch.Tensor(self._env.action_space.shape)).item()

        # 计算 CVaR 系数: phi(Phi^{-1}(alpha)) / (1 - alpha)
        # 其中 alpha 是风险水平（如 0.5 表示关注上 50% 的尾部风险）
        alpha = float(self._cfgs.lagrange_cfgs.cvar_alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError('cvar_alpha must be strictly between 0 and 1')
        normal = Normal(
            loc=torch.tensor(0.0),
            scale=torch.tensor(1.0),
        )
        z_alpha = normal.icdf(torch.tensor(alpha))
        # phi(z_alpha) = pdf of standard normal at z_alpha
        phi_z_alpha = torch.exp(normal.log_prob(z_alpha))
        self._pdf_cdf = (phi_z_alpha / (1.0 - alpha)).item()

        max_ep_len = int(self._cfgs.algo_cfgs.get('max_ep_len', 1000))
        gamma = float(self._cfgs.algo_cfgs.gamma)
        if gamma == 1.0:
            raise ValueError('WCSAC cost-limit conversion requires gamma < 1')
        self._cost_constraint = (
            float(self._cfgs.lagrange_cfgs.cost_limit)
            * (1.0 - gamma**max_ep_len)
            / (1.0 - gamma)
            / max_ep_len
        )
        self._damp = 0.0

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        """Return x such that softplus(x) is approximately ``value``."""
        value = max(value, 1e-8)
        return torch.log(torch.expm1(torch.tensor(value))).item()

    @property
    def _alpha(self) -> float:
        """Return the positive entropy coefficient."""
        return F.softplus(self._soft_alpha).item()

    @property
    def _lagrange_multiplier(self) -> torch.Tensor:
        """Return the positive WCSAC cost penalty."""
        return F.softplus(self._soft_beta)

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
        obs: torch.Tensor,
        act: torch.Tensor,
        *,
        target: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract Gaussian distribution parameters from cost critic output.

        Args:
            obs: Observation tensor of shape (B, obs_dim).
            act: Action tensor of shape (B, act_dim).
            target: Whether to use target cost critics.

        Returns:
            mu: Mean of cost Q-value, shape (B, 1).
            var: Variance of cost Q-value, shape (B, 1), always positive.
        """
        mean_critic = (
            self._actor_critic.target_cost_critic
            if target
            else self._actor_critic.cost_critic
        )
        var_critic = (
            self._actor_critic.target_cost_var_critic
            if target
            else self._actor_critic.cost_var_critic
        )
        mu = mean_critic(obs, act)[0].unsqueeze(-1)
        var = F.softplus(var_critic(obs, act)[0]).unsqueeze(-1)
        var = torch.clamp(var, min=1e-8, max=1e8)
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
        """Run the reference WCSAC update order on replay-buffer batches."""
        for _ in range(self._cfgs.algo_cfgs.update_iters):
            data = self._buf.sample_batch()
            self._update_count += 1
            obs, act, reward, cost, done, next_obs = (
                data['obs'],
                data['act'],
                data['reward'],
                data['cost'],
                data['done'],
                data['next_obs'],
            )

            self._update_actor(obs)
            self._update_reward_critic(obs, act, reward, done, next_obs)
            if self._cfgs.algo_cfgs.use_cost:
                self._update_cost_critic(obs, act, cost, done, next_obs)
            self._update_alpha(obs)
            if self._cfgs.algo_cfgs.use_cost:
                self._update_beta(obs, act)
            self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

    def _update_actor(self, obs: torch.Tensor) -> None:
        """Update only the policy; alpha is updated later in reference order."""
        DDPG._update_actor(self, obs)

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update both reward critics with the reference WCSAC loss scale."""
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=False)
            next_logp = self._actor_critic.actor.log_prob(next_action)
            next_q1, next_q2 = self._actor_critic.target_reward_critic(
                next_obs,
                next_action,
            )
            next_q = torch.min(next_q1, next_q2) - self._alpha * next_logp
            target_q = reward + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q

        q1, q2 = self._actor_critic.reward_critic(obs, action)
        loss = 0.5 * F.mse_loss(q1, target_q) + 0.5 * F.mse_loss(q2, target_q)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.reward_critic_optimizer.step()
        self._logger.store(
            {
                'Loss/Loss_reward_critic': loss.item(),
                'Value/reward_critic': q1.mean().item(),
            },
        )

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
            next_mu, next_var = self._get_cost_dist_params(next_obs, next_action, target=True)

            # Mean target: standard Bellman
            target_mu = cost + self._cfgs.algo_cfgs.gamma * (1 - done) * next_mu

            # 获取当前 qc（用于方差 target 计算）
            # 注意：这里需要用当前网络的 mu，而不是 target
            curr_mu, _ = self._get_cost_dist_params(obs, action)

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
            target_var = torch.clamp(target_var, min=1e-8, max=1e8)

        # 当前 cost critic 的分布参数
        mu, var = self._get_cost_dist_params(obs, action)

        # 数值检查
        if not torch.isfinite(mu).all() or not torch.isfinite(var).all():
            raise RuntimeError('cost_critic outputs NaN/Inf')

        # Mean loss: MSE
        loss_mu = 0.5 * F.mse_loss(mu, target_mu)

        # Variance loss: Wasserstein-style distance
        # L_var = 0.5 * E[var + target_var - 2 * sqrt(var * target_var)]
        loss_var = 0.5 * (var + target_var - 2 * torch.sqrt(var * target_var)).mean()

        # 计算阻尼项 (基于 real actions 的 CVaR 与 cost_limit 的差值)
        damp_scale = float(self._cfgs.algo_cfgs.get('damp_scale', 10.0))
        cvar_real = self._gaussian_cvar(mu.detach(), var.detach())
        self._damp = damp_scale * (self._cost_constraint - cvar_real.mean()).item()

        loss = loss_mu + loss_var

        if self._cfgs.algo_cfgs.use_critic_norm:
            for critic in (
                self._actor_critic.cost_critic,
                self._actor_critic.cost_var_critic,
            ):
                for param in critic.parameters():
                    loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.cost_critic_optimizer.zero_grad()
        self._actor_critic.cost_var_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                list(self._actor_critic.cost_critic.parameters())
                + list(self._actor_critic.cost_var_critic.parameters()),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.cost_critic_optimizer.step()
        self._actor_critic.cost_var_critic_optimizer.step()

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
            mu, var = self._get_cost_dist_params(obs, action)

            if torch.isfinite(mu).all() and torch.isfinite(var).all():
                cvar = self._gaussian_cvar(mu, var)

                if torch.isfinite(cvar).all():
                    loss_cost = (
                        self._lagrange_multiplier.detach() - self._damp
                    ) * cvar.squeeze(-1)

        total_loss = (loss_entropy + loss_reward + loss_cost).mean()

        return total_loss

    def _update_alpha(self, obs: torch.Tensor) -> None:
        """Update entropy coefficient as in the reference implementation."""
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action).detach()
        entropy = -log_prob.mean()
        alpha_loss = -F.softplus(self._soft_alpha) * (self._target_entropy - entropy)

        self._alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self._alpha_optimizer.step()
        self._logger.store(
            {
                'Loss/alpha_loss': alpha_loss.item(),
                'Value/alpha': self._alpha,
            },
        )

    def _update_beta(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> None:
        """Update the softplus cost penalty using replay-buffer actions."""
        with torch.no_grad():
            mu, var = self._get_cost_dist_params(obs, action)
            cvar = self._gaussian_cvar(mu, var)

        beta_loss = (
            F.softplus(self._soft_beta) * (self._cost_constraint - cvar)
        ).mean()
        self._beta_optimizer.zero_grad()
        beta_loss.backward()
        self._beta_optimizer.step()

        self._logger.store(
            {
                'Value/lagrange_multiplier': self._lagrange_multiplier.item(),
                'Value/wccvar': cvar.mean().item(),
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
