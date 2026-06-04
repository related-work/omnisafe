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
"""Implementation of the Distributional Constrained SAC (DCSAC) algorithm."""
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
class DCSAC(SAC):
    """Distributional Constrained Soft Actor-Critic (DCSAC) algorithm.

    This algorithm extends SAC-Lag by modeling the cost-return distribution as a LogNormal
    and using Worst-Case Conditional Value-at-Risk (WCCVaR) as the risk measure for
    constraint satisfaction.

    References:
        - SAC: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
          with a Stochastic Actor (Haarnoja et al., 2018)
        - Distributional Constrained RL with WCCVaR risk measure
    """

    # 类内常量（若需要限制 λ 上界或单次更新幅度，可在此调整）
    _LAMBDA_MAX: float = 100.0  # λ 上界，默认不做严格限制
    SCALE = 0.01  # 缩放因子
    _lagrange_multiplier: torch.Tensor

    def _init_model(self) -> None:
        """Initialize the model.

        Uses DisConstraintActorQCritic with:
        - reward_critic: dual Q-networks (num_critics=2) for SAC
        - cost_critic: single Q-network with 2-dim output (mu_log, std_raw) for LogNormal
        """
        # SAC 需要双Q reward critic
        self._cfgs.model_cfgs.critic['num_critics'] = 2

        self._actor_critic = DisConstraintActorQCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)

    def _init(self) -> None:
        """Initialize algorithm-specific components.

        Initializes:
        - SAC's entropy temperature (alpha) and its optimizer (if auto_alpha)
        - Lagrange multiplier for cost constraint
        """
        super()._init()

        # 初始化 Lagrange 乘子
        self._lagrange_multiplier = torch.tensor(
            float(self._cfgs.lagrange_cfgs.lagrangian_multiplier_init),
            device=self._device,
        )
        

    def _init_log(self) -> None:
        """Register logging keys for DCSAC-specific metrics."""
        super()._init_log()

        # WCCVaR 与 Lagrange 乘子
        self._logger.register_key('Value/wccvar')
        self._logger.register_key('Value/lagrange_multiplier')

        # cost critic 分布参数（用于调试）
        self._logger.register_key('Value/cost_mu_log')
        self._logger.register_key('Value/cost_std_log')

    # ==================== LogNormal 分布工具函数 ====================

    def _get_cost_dist_params(
        self,
        cost_critic: nn.Module,
        obs: torch.Tensor,
        act: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract LogNormal distribution parameters from cost critic output.

        Args:
            cost_critic: The cost critic network (current or target).
            obs: Observation tensor of shape (B, obs_dim).
            act: Action tensor of shape (B, act_dim).

        Returns:
            mu_log: Mean of log(X+1), shape (B, 1).
            std_log: Std of log(X+1), shape (B, 1), always positive.
        """
        q_out = cost_critic(obs, act)[0]  # [B, 2]
        mu_log = q_out[:, 0:1]  # [B, 1]
        std_raw = q_out[:, 1:2]  # [B, 1]
        std_log = F.softplus(std_raw) + 1e-6  # 保证正数
        std_log = torch.clamp(std_log, max=5.0)
        return mu_log, std_log

    def _lognormal_mean(
        self,
        mu_log: torch.Tensor,
        std_log: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the mean of LogNormal distribution.

        If X+1 ~ LogNormal(mu, sigma), then E[X] = exp(mu + 0.5*sigma^2) - 1.

        Args:
            mu_log: Mean parameter of shape (B, 1).
            std_log: Std parameter of shape (B, 1).

        Returns:
            Mean of X, shape (B, 1).
        """
        exponent = mu_log + 0.5 * std_log.pow(2)
        exponent = torch.clamp(exponent, -10.0, 10.0)  # 数值稳定
        return torch.exp(exponent) - 1.0

    def _lognormal_cvar(
        self,
        mu_log: torch.Tensor,
        std_log: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Compute CVaR of LogNormal distribution (closed-form).

        For X+1 ~ LogNormal(mu, sigma), CVaR_alpha(X) is computed as:
            z_alpha = Phi^{-1}(alpha)
            E[X+1] = exp(mu + sigma^2/2)
            CVaR(X+1) = E[X+1] * Phi(sigma - z_alpha) / (1 - alpha)
            CVaR(X) = CVaR(X+1) - 1

        Args:
            mu_log: Mean parameter of shape (B, 1).
            std_log: Std parameter of shape (B, 1).
            alpha: Risk level in (0, 1), e.g., 0.95.

        Returns:
            CVaR of X, shape (B, 1).
        """
        # 标准正态分布
        normal = Normal(
            loc=torch.tensor(0.0, device=mu_log.device, dtype=mu_log.dtype),
            scale=torch.tensor(1.0, device=mu_log.device, dtype=mu_log.dtype),
        )

        # z_alpha = Phi^{-1}(alpha)
        alpha_tensor = torch.tensor(alpha, device=mu_log.device, dtype=mu_log.dtype)
        z_alpha = normal.icdf(alpha_tensor)

        # E[X+1] = exp(clip(mu + sigma^2/2, -10, 10))
        exponent = mu_log + std_log.pow(2) / 2.0
        exponent = torch.clamp(exponent, -10.0, 10.0)
        
        E_X_plus_1 = torch.exp(exponent)

        # CVaR(X+1) = E[X+1] * Phi(sigma - z_alpha) / (1 - alpha)
        cdf_term = normal.cdf(std_log - z_alpha)
        cvar_shifted = E_X_plus_1 * cdf_term / (1.0 - alpha)

        # CVaR(X) = CVaR(X+1) - 1
        return cvar_shifted - 1.0

    def _update(self) -> None:
        SCALE = 0.01 
        for _ in range(self._cfgs.algo_cfgs.update_iters):
            data = self._buf.sample_batch()
            scaled_cost = data['cost'] * SCALE
            self._update_count += 1

            obs, act, reward, cost, done, next_obs = (
                data['obs'],
                data['act'],
                data['reward'],
                data['cost'],
                data['done'],
                data['next_obs'],
            )

            # 1. 更新 reward critic（SAC 双Q + entropy target）
            self._update_reward_critic(obs, act, reward, done, next_obs)

            # 2. 更新 cost critic（LogNormal 分布建模）
            if self._cfgs.algo_cfgs.use_cost:
                self._update_cost_critic(obs, act, scaled_cost, done, next_obs)

            # 3. 延迟更新 actor
            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                self._update_actor(obs)
                # 4. Polyak 更新 target 网络（包括 reward_critic 和 cost_critic）
                self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

        # 5. 更新 Lagrange 乘子（在所有 update_iters 完成后，每个 _update 调用一次）
        # 使用最后一个 batch 的 obs 来计算 WCCVaR
        # 这与 SAC-Lag 的风格一致：在 _update 末尾更新 λ
        if self._cfgs.algo_cfgs.use_cost:
            if self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
                self._update_lagrange_multiplier_wccvar(obs,scale_factor=SCALE)
            else:
                # warmup 期间记录但不更新 λ
                self._log_lagrange_when_warmup(obs,scale_factor=SCALE)


    # ==================== Critic 更新 ====================

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update reward critic using SAC's double-Q with entropy term.

        This is inherited from SAC and uses:
        - Double Q-networks with min(Q1, Q2) for target
        - Entropy bonus: -alpha * log_prob in target value
        """
        # 直接调用 SAC 父类的实现
        super()._update_reward_critic(obs, action, reward, done, next_obs)

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update cost critic using LogNormal NLL loss.

        The cost critic outputs (mu_log, std_raw) parameterizing a LogNormal distribution.
        Target is computed via Bellman backup using the mean of the target distribution.

        Args:
            obs: Current observations, shape (B, obs_dim).
            action: Actions taken, shape (B, act_dim).
            cost: Immediate costs, shape (B,) or (B, 1).
            done: Terminal flags, shape (B,) or (B, 1).
            next_obs: Next observations, shape (B, obs_dim).
        """
        # 确保 cost 和 done 是 (B, 1)
        if cost.dim() == 1:
            cost = cost.unsqueeze(-1)
        if done.dim() == 1:
            done = done.unsqueeze(-1)

        with torch.no_grad():
            # 使用当前策略采样下一步动作（SAC 风格，stochastic）
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=False)

            # 从 target cost critic 获取分布参数并计算 mean
            next_mu_log, next_std_log = self._get_cost_dist_params(
                self._actor_critic.target_cost_critic,
                next_obs,
                next_action,
            )
            next_mean = self._lognormal_mean(next_mu_log, next_std_log)  # [B, 1]

            # Bellman target: c + γ(1-d) * E[C']
            target_c = cost + self._cfgs.algo_cfgs.gamma * (1 - done) * next_mean
            target_c = torch.clamp(target_c, min=0.0)  # cost-return 应 >= 0

            # 转换为 log 空间（对齐 LogNormal: log(X+1)）
            target_log = torch.log(target_c + 1.0)

        # 当前 cost critic 的分布参数
        mu_log, std_log = self._get_cost_dist_params(
            self._actor_critic.cost_critic,
            obs,
            action,
        )

        # 数值检查
        if not torch.isfinite(mu_log).all() or not torch.isfinite(std_log).all():
            raise RuntimeError('cost_critic outputs NaN/Inf')

        # NLL loss: -log p(target_log | mu_log, std_log)
        dist = Normal(mu_log, std_log)
        loss = -dist.log_prob(target_log).mean()

        # 可选的 critic 正则化
        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        # 反向传播
        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.cost_critic_optimizer.step()

        # 记录日志
        with torch.no_grad():
            pred_mean = self._lognormal_mean(mu_log, std_log)

        self._logger.store(
            {
                'Loss/Loss_cost_critic': loss.mean().item(),
                'Value/cost_critic': pred_mean.mean().item(),
                'Value/cost_mu_log': mu_log.mean().item(),
                'Value/cost_std_log': std_log.mean().item(),
            },
        )

    # ==================== Actor 更新 ====================

    def _update_actor(self, obs: torch.Tensor) -> None:
        """Update actor and alpha, then update Lagrange multiplier.

        The actor loss combines:
        - SAC's entropy-regularized reward objective
        - Lagrangian penalty on WCCVaR of cost

        Key design: Both actor loss and λ update use the SAME sampled action a ~ π(·|s)
        to ensure consistency between the optimization objective and constraint estimation.
        """
        # 计算 actor loss 并更新
        loss = self._loss_pi(obs)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.actor_optimizer.step()

        self._logger.store({'Loss/Loss_pi': loss.mean().item()})

        # 更新 alpha（如果启用自动调节）
        if self._cfgs.algo_cfgs.auto_alpha:
            with torch.no_grad():
                action = self._actor_critic.actor.predict(obs, deterministic=False)
                log_prob = self._actor_critic.actor.log_prob(action)
            alpha_loss = -self._log_alpha * (log_prob + self._target_entropy).mean()

            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

            self._logger.store({'Loss/alpha_loss': alpha_loss.mean().item()})

        self._logger.store({'Value/alpha': self._alpha})

       

    def _loss_pi(self, obs: torch.Tensor) -> torch.Tensor:
        r"""Compute actor loss for DCSAC.

        The loss function is:

        .. math::

            L = \mathbb{E}_{a \sim \pi}[\alpha \log \pi(a|s) - \min(Q_1^r, Q_2^r)(s,a)
                + \lambda \cdot \text{WCCVaR}(s,a)]

        Key design: We use a ~ π(·|s) (reparameterized sampling) for BOTH the reward
        objective and the cost constraint. This ensures that:
        1. The gradient flows through the policy for both terms
        2. The constraint is evaluated on the actions the policy would actually take

        Note: Unlike DCDDPG, we do NOT divide by (1 + λ) for normalization.

        Args:
            obs: Observation tensor of shape (B, obs_dim).

        Returns:
            Scalar loss tensor.
        """
        # 采样动作（reparameterized，梯度可流回 actor）
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)

        # SAC reward objective: alpha * log_prob - min(Q1, Q2)
        q1_value_r, q2_value_r = self._actor_critic.reward_critic(obs, action)
        loss_entropy = self._alpha * log_prob
        loss_reward = -torch.min(q1_value_r, q2_value_r)

        # Cost constraint via WCCVaR
        # 使用同一个 action 计算 WCCVaR，保证 actor loss 与约束估计的一致性
        loss_cost = torch.zeros_like(loss_reward)

        if self._cfgs.algo_cfgs.use_cost:
            mu_log, std_log = self._get_cost_dist_params(
                self._actor_critic.cost_critic,
                obs,
                action,
            )

            # 数值保护：如果 critic 输出异常，跳过 cost 项
            if torch.isfinite(mu_log).all() and torch.isfinite(std_log).all():
                alpha_cvar = float(self._cfgs.lagrange_cfgs.cvar_alpha)
                wccvar = self._lognormal_cvar(mu_log, std_log, alpha_cvar)
                if wccvar.dim() == 2:
                    wccvar = wccvar.squeeze(-1) 
                if torch.isfinite(wccvar).all():
                    # λ 项：即使在 warmup 期间也参与 loss（但 λ 不更新）
                    loss_cost = self._lagrange_multiplier * wccvar

        # 总 loss（不除以 (1+λ)）
        denom = (1.0 + self._lagrange_multiplier).detach()
        total_loss = (loss_entropy + loss_reward + loss_cost).mean()

        return (total_loss/denom)

    # ==================== Lagrange 乘子更新 ====================

    def _update_lagrange_multiplier_wccvar(self, obs: torch.Tensor,scale_factor: float = 1.0) -> None:
        """Update Lagrange multiplier based on WCCVaR constraint violation.

        The update rule is:
            gap = J - cost_limit, where J = mean(WCCVaR)
            λ ← clamp(λ + lr * gap, min=0, max=λ_max)

        Key design: We use a ~ π(·|s) to compute WCCVaR, consistent with the actor loss.
        This ensures the constraint is evaluated on the policy's actual action distribution,
        not on outdated actions from the replay buffer.

        Args:
            obs: Observation tensor of shape (B, obs_dim).
        """
        alpha_cvar = float(self._cfgs.lagrange_cfgs.cvar_alpha)
        cost_limit = float(self._cfgs.lagrange_cfgs.cost_limit)
        lambda_lr = float(self._cfgs.lagrange_cfgs.lambda_lr)

        with torch.no_grad():
            # 采样当前策略的动作（与 actor loss 中一致）
            # 这确保了 λ 更新基于的约束估计与 actor 优化目标一致
            action = self._actor_critic.actor.predict(obs, deterministic=False)

            # 使用当前 cost_critic（不是 target）计算 WCCVaR
            mu_log, std_log = self._get_cost_dist_params(
                self._actor_critic.cost_critic,
                obs,
                action,
            )

            # 数值保护
            if not torch.isfinite(mu_log).all() or not torch.isfinite(std_log).all():
                self._logger.store(
                    {
                        'Value/lagrange_multiplier': self._lagrange_multiplier.item(),
                        'Value/wccvar': 0.0,
                    },
                )
                return

            wccvar_scaled  = self._lognormal_cvar(mu_log, std_log, alpha_cvar)
            if wccvar_scaled .dim() == 2:
                wccvar_scaled  = wccvar_scaled .squeeze(-1)

            if not torch.isfinite(wccvar_scaled ).all():
                self._logger.store(
                    {
                        'Value/lagrange_multiplier': self._lagrange_multiplier.item(),
                        'Value/wccvar': 0.0,
                    },
                )
                return

            # J = batch 平均的 WCCVaR
            J = (wccvar_scaled / scale_factor).mean()
            gap = J - cost_limit

            # 梯度上升更新 λ 并投影到 [0, λ_max]
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

    def _log_lagrange_when_warmup(self, obs: torch.Tensor,scale_factor: float = 1.0) -> None:
        """Log Lagrange-related metrics during warmup (without updating λ).

        Args:
            obs: Observation tensor of shape (B, obs_dim).
        """
        alpha_cvar = float(self._cfgs.lagrange_cfgs.cvar_alpha)

        with torch.no_grad():
            action = self._actor_critic.actor.predict(obs, deterministic=False)
            mu_log, std_log = self._get_cost_dist_params(
                self._actor_critic.cost_critic,
                obs,
                action,
            )

            wccvar_val = 0.0
            if torch.isfinite(mu_log).all() and torch.isfinite(std_log).all():
                wccvar = self._lognormal_cvar(mu_log, std_log, alpha_cvar)
                if torch.isfinite(wccvar).all():
                    # 同样需要还原
                    wccvar_val = (wccvar.mean() / scale_factor).item()

            self._logger.store(
                {
                    'Value/lagrange_multiplier': self._lagrange_multiplier.item(),
                    'Value/wccvar': wccvar_val,
                },
            )

    # ==================== 其他辅助方法 ====================

    def _log_when_not_update(self) -> None:
        """Log default values when not updating (during initial exploration)."""
        super()._log_when_not_update()

        self._logger.store(
            {
                'Value/wccvar': 0.0,
                'Value/lagrange_multiplier': self._lagrange_multiplier.item(),
                'Value/cost_mu_log': 0.0,
                'Value/cost_std_log': 0.0,
            },
        )