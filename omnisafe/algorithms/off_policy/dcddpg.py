from __future__ import annotations


from typing import Any
import os, time
import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn.functional as F
from torch.distributions import Normal

# from omnisafe.adapter.offpolicy_adapter_1 import OffPolicyAdapter_1
from omnisafe.adapter.offpolicy_adapter import OffPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
# from omnisafe.common.buffer.vector_offpolicy_buffer_1 import VectorOffPolicyBuffer_1  #1111
from omnisafe.common.buffer.vector_offpolicy_buffer import VectorOffPolicyBuffer 
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.disconstraint_actor_q_critic import DisConstraintActorQCritic


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class DCDDPG(BaseAlgo):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:
        - Title: Continuous control with deep reinforcement learning
        - Authors: Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess,
            Tom Erez, Yuval Tassa, David Silver, Daan Wierstra.
        - URL: `DDPG <https://arxiv.org/abs/1509.02971>`_
    """

    _epoch: int
 
    def _init_env(self) -> None:
        """Initialize the environment.

        OmniSafe uses :class:`omnisafe.adapter.OffPolicyAdapter` to adapt the environment to this
        algorithm.

        User can customize the environment by inheriting this method.

        Examples:
            >>> def _init_env(self) -> None:
            ...     self._env = CustomAdapter()

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
            AssertionError: If the total number of steps is not divisible by the number of steps per
                epoch.
        """
        self._env: OffPolicyAdapter = OffPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        
        #第一个：steps_per_epoch 必须能均分到每个并行环境上，否则每个 env 每个 epoch 该跑多少步会变成小数。

        #第二个：总步数必须是整 epoch，不然最后一个 epoch 会不完整（日志、lr schedule、保存频率等会变乱）。
        
        assert (
            self._cfgs.algo_cfgs.steps_per_epoch % self._cfgs.train_cfgs.vector_env_nums == 0
        ), 'The number of steps per epoch is not divisible by the number of environments.'

        assert (
            int(self._cfgs.train_cfgs.total_steps) % self._cfgs.algo_cfgs.steps_per_epoch == 0
        ), 'The total number of steps is not divisible by the number of steps per epoch.'
        
        self._epochs: int = int(
            self._cfgs.train_cfgs.total_steps // self._cfgs.algo_cfgs.steps_per_epoch,
        )
        self._epoch: int = 0
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch // self._cfgs.train_cfgs.vector_env_nums
        )

        self._update_cycle: int = self._cfgs.algo_cfgs.update_cycle # 每次与环境交互时，每个并行环境连续走多少步。
        assert (
            self._steps_per_epoch % self._update_cycle == 0
        ), 'The number of steps per epoch is not divisible by the number of steps per sample.'
        self._samples_per_epoch: int = self._steps_per_epoch // self._update_cycle
        self._update_count: int = 0

    def _init_model(self) -> None:
        """Initialize the model.

        OmniSafe uses :class:`omnisafe.models.actor_critic.constraint_actor_q_critic.ConstraintActorQCritic`
        as the default model.

        User can customize the model by inheriting this method.

        Examples:
            >>> def _init_model(self) -> None:
            ...     self._actor_critic = CustomActorQCritic()
        """
        self._cfgs.model_cfgs.critic['num_critics'] = 1
        self._actor_critic: DisConstraintActorQCritic = DisConstraintActorQCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)

    def _init(self) -> None:
        """The initialization of the algorithm.

        User can define the initialization of the algorithm by inheriting this method.

        Examples:
            >>> def _init(self) -> None:
            ...     super()._init()
            ...     self._buffer = CustomBuffer()
            ...     self._model = CustomModel()
        """
        self._buf: VectorOffPolicyBuffer = VectorOffPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._cfgs.algo_cfgs.size,
            batch_size=self._cfgs.algo_cfgs.batch_size,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            penalty_coefficient=self._cfgs.algo_cfgs.get('penalty_coefficient', 0.0),
            device=self._device,
        )
        
        self._lagrange_update_count = 0
        self._lagrange_multiplier = torch.tensor(
        float(self._cfgs.lagrange_cfgs.lagrangian_multiplier_init),
        device=self._device,
    )

        self.lambda_max=1000.0
        self.lambda_delta_max=100.0
        #   # === 方案B：存元数据 ===
        # self._buf.add_field('env_id', (), torch.int64)
        # self._buf.add_field('env_step', (), torch.int64)
        
        # # 111
        # import datetime as dt
        # ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # qc_dir = os.path.join(self._cfgs.logger_cfgs.log_dir, self._cfgs.exp_name)
        # os.makedirs(qc_dir, exist_ok=True)
        # self._qc_tgt_sampled_path = os.path.join(qc_dir, f"qc_target_sampled_{self._env_id}_{ts}.csv")
        # if not os.path.exists(self._qc_tgt_sampled_path):
        #     with open(self._qc_tgt_sampled_path, "w") as f:
        #         f.write("env_step,env_id,q_c_target\n")
                
                
        # self._qc_flush_every = 10000 # sampled 的量更大，建议更大一点
        # self._qc_tgt_sampled_lines: list[str] = []


    def _init_log(self) -> None:
        """Log info about epoch.

        +-------------------------+----------------------------------------------------------------------+
        | Things to log           | Description                                                          |
        +=========================+======================================================================+
        | Train/Epoch             | Current epoch.                                                       |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/EpCost          | Average cost of the epoch.                                           |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/EpRet           | Average return of the epoch.                                         |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/EpLen           | Average length of the epoch.                                         |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/TestEpCost      | Average cost of the evaluate epoch.                                  |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/TestEpRet       | Average return of the evaluate epoch.                                |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/TestEpLen       | Average length of the evaluate epoch.                                |
        +-------------------------+----------------------------------------------------------------------+
        | Value/reward_critic     | Average value in :meth:`rollout` (from critic network) of the epoch. |
        +-------------------------+----------------------------------------------------------------------+
        | Values/cost_critic      | Average cost in :meth:`rollout` (from critic network) of the epoch.  |
        +-------------------------+----------------------------------------------------------------------+
        | Loss/Loss_pi            | Loss of the policy network.                                          |
        +-------------------------+----------------------------------------------------------------------+
        | Loss/Loss_reward_critic | Loss of the reward critic.                                           |
        +-------------------------+----------------------------------------------------------------------+
        | Loss/Loss_cost_critic   | Loss of the cost critic network.                                     |
        +-------------------------+----------------------------------------------------------------------+
        | Train/LR                | Learning rate of the policy network.                                 |
        +-------------------------+----------------------------------------------------------------------+
        | Misc/Seed               | Seed of the experiment.                                              |
        +-------------------------+----------------------------------------------------------------------+
        | Misc/TotalEnvSteps      | Total steps of the experiment.                                       |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Total              | Total time.                                                          |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Rollout            | Rollout time.                                                        |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Update             | Update time.                                                         |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Evaluate           | Evaluate time.                                                       |
        +-------------------------+----------------------------------------------------------------------+
        | FPS                     | Frames per second of the epoch.                                      |
        +-------------------------+----------------------------------------------------------------------+
        """
        self._logger: Logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        what_to_save: dict[str, Any] = {}
        what_to_save['pi'] = self._actor_critic.actor
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer

        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

        self._logger.register_key(
            'Metrics/EpRet',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(
            'Metrics/EpCost',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(
            'Metrics/EpLen',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )

        if self._cfgs.train_cfgs.eval_episodes > 0:
            self._logger.register_key(
                'Metrics/TestEpRet',
                window_length=self._cfgs.logger_cfgs.window_lens,
            )
            self._logger.register_key(
                'Metrics/TestEpCost',
                window_length=self._cfgs.logger_cfgs.window_lens,
            )
            self._logger.register_key(
                'Metrics/TestEpLen',
                window_length=self._cfgs.logger_cfgs.window_lens,
            )

        self._logger.register_key('Train/Epoch')
        self._logger.register_key('Train/LR')

        self._logger.register_key('TotalEnvSteps')

        # log information about actor
        self._logger.register_key('Loss/Loss_pi', delta=True)

        # log information about critic
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward_critic')
        self._logger.register_key('Value/cost_mu_log')
        self._logger.register_key('Value/cost_std_log')


        if self._cfgs.algo_cfgs.use_cost:
            # log information about cost critic
            self._logger.register_key('Loss/Loss_cost_critic', delta=True)
            self._logger.register_key('Value/cost_critic')

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Evaluate')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')
        
        
        self._logger.register_key('Value/lagrange_multiplier')
        self._logger.register_key('Value/wccvar')
        self._logger.register_key('Value/lambda_delta')


        # register environment specific keys
        for env_spec_key in self._env.env_spec_keys:
            self._logger.register_key(env_spec_key)

    def learn(self) -> tuple[float, float, float]:
        """This is main function for algorithm update.

        It is divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.

        Returns:
            ep_ret: average episode return in final epoch.
            ep_cost: average episode cost in final epoch.
            ep_len: average episode length in final epoch.
        """
        self._logger.log('INFO: Start training')
        start_time = time.time()
        step = 0
        for epoch in range(self._epochs):
            self._epoch = epoch
            rollout_time = 0.0
            update_time = 0.0
            epoch_time = time.time()

            for sample_step in range(
                epoch * self._samples_per_epoch,
                (epoch + 1) * self._samples_per_epoch,
            ):
                step = sample_step * self._update_cycle * self._cfgs.train_cfgs.vector_env_nums

                rollout_start = time.time()
                # set noise for exploration
                if self._cfgs.algo_cfgs.use_exploration_noise:
                    self._actor_critic.actor.noise = self._cfgs.algo_cfgs.exploration_noise

                # collect data from environment
                self._env.rollout(
                    rollout_step=self._update_cycle,
                    agent=self._actor_critic,
                    buffer=self._buf,
                    logger=self._logger,
                    use_rand_action=(step <= self._cfgs.algo_cfgs.start_learning_steps),
                )
                rollout_time += time.time() - rollout_start

                # update parameters
                update_start = time.time()
                if step > self._cfgs.algo_cfgs.start_learning_steps:
                    
                    self._update()
                # if we haven't updated the network, log 0 for the loss
                else:
                    self._log_when_not_update()
                update_time += time.time() - update_start

            eval_start = time.time()
            self._env.eval_policy(
                episode=self._cfgs.train_cfgs.eval_episodes,
                agent=self._actor_critic,
                logger=self._logger,
            )
            eval_time = time.time() - eval_start

            self._logger.store({'Time/Update': update_time})
            self._logger.store({'Time/Rollout': rollout_time})
            self._logger.store({'Time/Evaluate': eval_time})

            if (
                step > self._cfgs.algo_cfgs.start_learning_steps
                and self._cfgs.model_cfgs.linear_lr_decay
            ):
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': step + 1,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': self._actor_critic.actor_scheduler.get_last_lr()[0],
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        
        
        # # flush adapter qc logs
        # if hasattr(self._env, "flush_qc_logs"):
        #     self._env.flush_qc_logs()

        # # flush sampled qc-target logs
        # if hasattr(self, "_qc_tgt_sampled_lines") and len(self._qc_tgt_sampled_lines) > 0:
        #     with open(self._qc_tgt_sampled_path, "a") as f:
        #         f.writelines(self._qc_tgt_sampled_lines)
                
        #     self._qc_tgt_sampled_lines.clear()
        
        # if hasattr(self._env, "flush_qc_logs"):
        #     self._env.flush_qc_logs()

        self._env.close()
        self._logger.close()
        return ep_ret, ep_cost, ep_len

    def _update(self) -> None:
        """Update actor, critic.

        -  Get the ``data`` from buffer

        .. note::

            +----------+---------------------------------------+
            | obs      | ``observaion`` stored in buffer.      |
            +==========+=======================================+
            | act      | ``action`` stored in buffer.          |
            +----------+---------------------------------------+
            | reward   | ``reward`` stored in buffer.          |
            +----------+---------------------------------------+
            | cost     | ``cost`` stored in buffer.            |
            +----------+---------------------------------------+
            | next_obs | ``next observaion`` stored in buffer. |
            +----------+---------------------------------------+
            | done     | ``terminated`` stored in buffer.      |
            +----------+---------------------------------------+

        -  Update value net by :meth:`_update_reward_critic`.
        -  Update cost net by :meth:`_update_cost_critic`.
        -  Update policy net by :meth:`_update_actor`.

        The basic process of each update is as follows:

        #. Get the mini-batch data from buffer.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the ``update_iters`` times.
        """
        for _ in range(self._cfgs.algo_cfgs.update_iters):
            data = self._buf.sample_batch()
            self._update_count += 1
 
            obs = data['obs']
            act = data['act']
            reward = data['reward']
            cost = data['cost']
            done = data['done']
            next_obs = data['next_obs']

            
      
            self._update_reward_critic(obs, act, reward, done, next_obs)
            if self._cfgs.algo_cfgs.use_cost:
                # self._update_cost_critic(obs, act, cost, done, next_obs)
                self._update_cost_critic(obs, act, cost, done, next_obs,)
                if self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
                    self._update_lagrange_multiplier_wccvar(obs)
          
                

            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0: #每做 policy_delay 次 mini-batch 更新，才更新一次 actor
                self._update_actor(obs)
                self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update reward critic.

        - Get the TD loss of reward critic.
        - Update critic network by loss.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            action (torch.Tensor): The ``action`` sampled from buffer.
            reward (torch.Tensor): The ``reward`` sampled from buffer.
            done (torch.Tensor): The ``terminated`` sampled from buffer.
            next_obs (torch.Tensor): The ``next observation`` sampled from buffer.
        """


        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=True)
            next_q_value_r = self._actor_critic.target_reward_critic(next_obs, next_action)[0]
            target_q_value_r = reward + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_r
        q_value_r = self._actor_critic.reward_critic(obs, action)[0]
        loss = nn.functional.mse_loss(q_value_r, target_q_value_r)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff
        self._logger.store(
            {
                'Loss/Loss_reward_critic': loss.mean().item(),
                'Value/reward_critic': q_value_r.mean().item(),
            },
        )
        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.reward_critic_optimizer.step()



    def _get_cost_dist_params(self,cost_critic, obs, act):
        q2 = cost_critic(obs, act)[0]   # [B,2]
        mu_log  = q2[:, 0:1]            # [B,1]
        std_raw = q2[:, 1:2]            # [B,1]
        std_log = F.softplus(std_raw) + 1e-6
        return mu_log, std_log
    
    def _lognormal_mean(self,mu_log: torch.Tensor, std_log: torch.Tensor) -> torch.Tensor:
        # E[X] = exp(mu + 0.5*sigma^2) - 1
        x = mu_log + 0.5 * std_log.pow(2)
        x = torch.clamp(x, -10.0, 10.0)  # 数值稳定
        return torch.exp(x) - 1.0
    
    def _lognormal_cvar(self, mu_log: torch.Tensor, std_log: torch.Tensor, alpha: float) -> torch.Tensor:
        """LogNormal CVaR：与 numpy/scipy 逻辑一致
        z_alpha = ppf(alpha)
        E_X_plus_1 = exp(clip(mu + std^2/2, -10, 10))
        cvar_shifted = E_X_plus_1 * cdf(std - z_alpha) / (1-alpha)
        return cvar_shifted - 1
        """
        normal = torch.distributions.Normal(
            loc=torch.tensor(0.0, device=mu_log.device, dtype=mu_log.dtype),
            scale=torch.tensor(1.0, device=mu_log.device, dtype=mu_log.dtype),
        )

        # z_alpha = scipy_norm.ppf(alpha)
        z_alpha = normal.icdf(torch.tensor(alpha, device=mu_log.device, dtype=mu_log.dtype))

        # E_X_plus_1 = exp(clip(mu + std^2/2, -10, 10))
        E_X_plus_1 = torch.exp(torch.clamp(mu_log + std_log.pow(2) / 2.0, -10.0, 10.0))

        # scipy_norm.cdf(std_log - z_alpha)
        cdf_term = normal.cdf(std_log - z_alpha)
        cvar_shifted = E_X_plus_1 * cdf_term / (1.0 - alpha)
        return cvar_shifted - 1.0

    
    
    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
      
    ) -> None:
        """Update cost critic.

        - Get the TD loss of cost critic.
        - Update critic network by loss.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            action (torch.Tensor): The ``action`` sampled from buffer.
            cost (torch.Tensor): The ``cost`` sampled from buffer.
            done (torch.Tensor): The ``terminated`` sampled from buffer.
            next_obs (torch.Tensor): The ``next observation`` sampled from buffer.
        """
        if cost.dim() == 1:
            cost = cost.unsqueeze(-1)
        if done.dim() == 1:
            done = done.unsqueeze(-1)
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=True)
            # next_q_value_c = self._actor_critic.target_cost_critic(next_obs, next_action)[0]
            # target_q_value_c = cost + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_c
            next_mu_log, next_std_log = self._get_cost_dist_params(self._actor_critic.target_cost_critic, next_obs, next_action)
            
            next_mean = self._lognormal_mean(next_mu_log, next_std_log)  # [B,1]
            target_c = cost + self._cfgs.algo_cfgs.gamma * (1 - done) * next_mean
            target_c = torch.clamp(target_c, min=0.0)          # cost-return 应 >=0
            target_log = torch.log(target_c + 1.0)             # 对齐 lognormal: X+1
        mu_log, std_log = self._get_cost_dist_params(self._actor_critic.cost_critic, obs, action)
          
        if not torch.isfinite(mu_log).all() or not torch.isfinite(std_log).all():
            raise RuntimeError("cost_critic outputs NaN/Inf")

        dist = Normal(mu_log, std_log)
        loss = -dist.log_prob(target_log).mean()
        
        # q_value_c = self._actor_critic.cost_critic(obs, action)[0]
        # loss = nn.functional.mse_loss(q_value_c, target_q_value_c)

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
        
        with torch.no_grad():
            pred_mean = self._lognormal_mean(mu_log, std_log)  # [B,1]
            
        self._logger.store(
            {
                'Loss/Loss_cost_critic': loss.mean().item(),
                'Value/cost_critic': pred_mean.mean().item(),
                'Value/cost_mu_log': float(mu_log.mean().item()),
                'Value/cost_std_log': float(std_log.mean().item()),
            },
        )

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
    ) -> None:
        """Update actor.

        - Get the loss of actor.
        - Update actor by loss.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
        """
        loss = self._loss_pi(obs)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.actor_optimizer.step()
        self._logger.store(
            {
                'Loss/Loss_pi': loss.mean().item(),
            },
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """Computing ``pi/actor`` loss.

        The loss function in DDPG is defined as:

        .. math::

            L = -Q^V (s, \pi (s))

        where :math:`Q^V` is the reward critic network, and :math:`\pi` is the policy network.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        action = self._actor_critic.actor.predict(obs, deterministic=True)
        q_r = self._actor_critic.reward_critic(obs, action)[0]
        loss_r = -q_r.mean()

        if not self._cfgs.algo_cfgs.use_cost:
            return loss_r

        
        alpha = float(self._cfgs.lagrange_cfgs.cvar_alpha)

        mu_log, std_log = self._get_cost_dist_params(self._actor_critic.cost_critic, obs, action)
        if (not torch.isfinite(mu_log).all()) or (not torch.isfinite(std_log).all()):
            return loss_r
        wccvar = self._lognormal_cvar(mu_log, std_log, alpha)

        if not torch.isfinite(wccvar).all():
            return loss_r
        loss_c = self._lagrange_multiplier * wccvar.mean()

        # 可选：防止 λ 很大时 loss 尺度太大（ddpg-lag 常见技巧）
        denom = (1.0 + self._lagrange_multiplier).detach()
        return (loss_r + loss_c) / denom

    def _log_when_not_update(self) -> None:
        """Log default value when not update."""
        self._logger.store(
            {
                'Loss/Loss_reward_critic': 0.0,
                'Loss/Loss_pi': 0.0,
                'Value/reward_critic': 0.0,
            },
        )
        if self._cfgs.algo_cfgs.use_cost:
            self._logger.store(
                {
                    'Loss/Loss_cost_critic': 0.0,
                    'Value/cost_critic': 0.0,
                },
            )
            self._logger.store({
            'Value/cost_mu_log': 0.0,
            'Value/cost_std_log': 0.0,
            'Value/lagrange_multiplier': float(self._lagrange_multiplier.item()),
            'Value/wccvar': 0.0,

        })

    def _update_lagrange_multiplier_wccvar(self, obs: torch.Tensor) -> None:
        alpha = float(self._cfgs.lagrange_cfgs.cvar_alpha)
        cost_limit = float(self._cfgs.lagrange_cfgs.cost_limit)
        lambda_lr = float(self._cfgs.lagrange_cfgs.lambda_lr)

        # 你已经在类里写死 self.lambda_max=1000.0
        lambda_max = float(self.lambda_max)

        # 你需要新增一个“单次最大更新幅度”，建议也写成成员变量（或 cfg）
        # 例如在 _init 里： self.lambda_delta_max = 1.0
        delta_max = float(self.lambda_delta_max)

        with torch.no_grad():
            # 用当前策略动作 π(s)
            act_pi = self._actor_critic.actor.predict(obs, deterministic=True)

            # ✅ 用当前 cost_critic（不是 target）
            mu_log, std_log = self._get_cost_dist_params(self._actor_critic.cost_critic, obs, act_pi)

            # 数值保护（建议保留，不然你很难挡住 critic 一次发散把 λ 污染）
            if not torch.isfinite(mu_log).all() or not torch.isfinite(std_log).all():
                return

            wccvar = self._lognormal_cvar(mu_log, std_log, alpha)  # [B,1] 或 [B]
            if wccvar.dim() == 2:
                wccvar = wccvar.squeeze(-1)

            if not torch.isfinite(wccvar).all():
                return

            J = wccvar.mean()              # scalar
            gap = J - cost_limit           # scalar

            # 原始梯度上升步长：Δλ = lr * gap
            delta = lambda_lr * gap

            # ✅ 限制单次更新最大幅度：|Δλ| <= delta_max
            delta = torch.clamp(delta, min=-delta_max, max=delta_max)

            # ✅ 更新并投影到 [0, lambda_max]
            self._lagrange_multiplier = torch.clamp(
                self._lagrange_multiplier + delta,
                min=0.0,
                max=lambda_max,
            )

            self._logger.store({
                'Value/lagrange_multiplier': float(self._lagrange_multiplier.item()),
                'Value/wccvar': float(J.item()),
                'Value/wccvar_gap': float(gap.item()),
                'Value/lambda_delta': float(delta.item()),  # 可选：方便看是不是被 clip 住
            })



    # def _update_lagrange_multiplier_wccvar(self, obs: torch.Tensor) -> None:
    #     alpha = float(self._cfgs.lagrange_cfgs.cvar_alpha)
    #     cost_limit = float(self._cfgs.lagrange_cfgs.cost_limit)
    #     lambda_lr = float(self._cfgs.lagrange_cfgs.lambda_lr)

    #     # 上界（配置里可选）
    #     lambda_max = float(self.lambda_max)

    #     with torch.no_grad():
    #         # ✅ 用 (s, pi(s))，对齐 actor
    #         act_pi = self._actor_critic.actor.predict(obs, deterministic=True)

    #         mu_log, std_log = self._get_cost_dist_params(self._actor_critic.target_cost_critic, obs, act_pi)


    #         # 数值保护：只为了防止 NaN 污染 λ（不改变你的目标，只是保护）
    #         if not torch.isfinite(mu_log).all() or not torch.isfinite(std_log).all():
    #             return

    #         # 计算 WCCVaR（每个样本一个值）
    #         wccvar = self._lognormal_cvar(mu_log, std_log, alpha)  # [B,1]
    #         if wccvar.dim() == 2:
    #             wccvar = wccvar.squeeze(-1)  # [B]

    #         if not torch.isfinite(wccvar).all():
    #             return

    #         J = wccvar.mean()               # 标量：mini-batch 上的风险估计
    #         gap = J - cost_limit            # 标量：>0 则违反约束

    #         # ✅ 手动 SGD ascent + 投影 + 上界
    #         self._lagrange_multiplier = torch.clamp(
    #             self._lagrange_multiplier + lambda_lr * gap,
    #             min=0.0,
    #             max=lambda_max,
    #         )

    #         self._logger.store({
    #             'Value/lagrange_multiplier': float(self._lagrange_multiplier.item()),
    #             'Value/wccvar': float(J.item()),
                
    #         })



