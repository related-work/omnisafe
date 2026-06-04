"""OffPolicy Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any
import os

import numpy as np
import torch

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils.config import Config
from omnisafe.common.buffer.vector_offpolicy_buffer_1 import VectorOffPolicyBuffer_1


class OffPolicyAdapter_1(OnlineAdapter):
    """OffPolicy Adapter for OmniSafe.

    Off-policy training may update the policy before episode ends,
    so this adapter stores current observation in ``_current_obs``.
    """

    _current_obs: torch.Tensor
    _ep_ret: torch.Tensor
    _ep_cost: torch.Tensor
    _ep_len: torch.Tensor

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        super().__init__(env_id, num_envs, seed, cfgs)

        self._current_obs, _ = self.reset()
        self._max_ep_len: int = 10000
        self._reset_log()

        # global step counts total env steps across vector envs
        self._global_env_step = 0

        qc_dir = os.path.join(self._cfgs.logger_cfgs.log_dir, self._cfgs.exp_name)
        os.makedirs(qc_dir, exist_ok=True)

        import datetime as dt
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        env_name = self._env_id
        self._qc_path = os.path.join(qc_dir, f"qc_{env_name}_{ts}.csv")

        # Buffer for qc logs (store on device, flush occasionally)
        self._qc_flush_every = 20000
        self._qc_buf: list[torch.Tensor] = []  # each: (num_envs, 4) on device

        with open(self._qc_path, "w") as f:
            f.write("env_step,env_id,q_c_est,q_target_c\n")

    def eval_policy(  # pylint: disable=too-many-locals
        self,
        episode: int,
        agent: ConstraintActorQCritic,
        logger: Logger,
    ) -> None:
        assert self._eval_env, "Environment for evaluation has not been set!"
        for _ in range(episode):
            ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
            obs, _ = self._eval_env.reset()
            obs = obs.to(self._device)

            done = False
            while not done:
                act = agent.step(obs, deterministic=True)
                obs, reward, cost, terminated, truncated, info = self._eval_env.step(act)
                obs, reward, cost, terminated, truncated = (
                    torch.as_tensor(x, dtype=torch.float32, device=self._device)
                    for x in (obs, reward, cost, terminated, truncated)
                )
                ep_ret += info.get("original_reward", reward).cpu()
                ep_cost += info.get("original_cost", cost).cpu()
                ep_len += 1
                done = bool(terminated[0].item()) or bool(truncated[0].item())

            logger.store(
                {
                    "Metrics/TestEpRet": ep_ret,
                    "Metrics/TestEpCost": ep_cost,
                    "Metrics/TestEpLen": ep_len,
                }
            )

    def rollout(  # pylint: disable=too-many-locals
        self,
        rollout_step: int,
        agent: ConstraintActorQCritic,
        buffer: VectorOffPolicyBuffer_1,
        logger: Logger,
        use_rand_action: bool,
    ) -> None:
        for _ in range(rollout_step):
            if use_rand_action:
                act = (torch.rand(self.action_space.shape) * 2 - 1).unsqueeze(0).to(self._device)  # type: ignore
            else:
                act = agent.step(self._current_obs, deterministic=False)

            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            self._log_value(reward=reward, cost=cost, info=info)

            real_next_obs = next_obs.clone()
            for idx, done in enumerate(torch.logical_or(terminated, truncated)):
                if done:
                    if "final_observation" in info:
                        real_next_obs[idx] = info["final_observation"][idx]
                    self._log_metrics(logger, idx)
                    self._reset_log(idx)

            # ===== env_id / env_step =====
            num_envs = buffer.num_envs
            env_id = torch.arange(num_envs, device=self._device, dtype=torch.int64)  # (num_envs,)
            base = self._global_env_step
            env_step = base + env_id  # (num_envs,)

            # ===== qc logging =====
            start_record_step = 10000
            # safer than attribute access if cfg is dict-like
            gamma_c = self._cfgs.algo_cfgs.get("gamma", 0.99)

            if base >= start_record_step:
                with torch.no_grad():
                    q_c = agent.cost_critic(self._current_obs, act)[0].reshape(num_envs)  # (num_envs,)

                    next_action = agent.actor.predict(real_next_obs, deterministic=True)
                    next_q_tgt = agent.target_cost_critic(real_next_obs, next_action)[0].reshape(num_envs)

                    done_store = torch.logical_or(terminated, truncated)  # (num_envs,)
                    done_f = done_store.float().reshape(num_envs)

                    q_target_all = cost.reshape(num_envs) + gamma_c * (1.0 - done_f) * next_q_tgt

                    # IMPORTANT: use stack to produce (num_envs, 4)
                    rec = torch.stack(
                        [
                            env_step.to(torch.float64),
                            env_id.to(torch.float64),
                            q_c.to(torch.float64),
                            q_target_all.to(torch.float64),
                        ],
                        dim=1,
                    )  # (num_envs, 4)

                    self._qc_buf.append(rec)

                # flush
                if len(self._qc_buf) >= self._qc_flush_every:
                    big = torch.cat(self._qc_buf, dim=0).cpu().numpy()  # (K, 4)
                    self._qc_buf.clear()
                    with open(self._qc_path, "a") as f:
                        np.savetxt(
                            f,
                            big,
                            delimiter=",",
                            fmt=["%.0f", "%.0f", "%.10g", "%.10g"],
                        )

            buffer.store(
                obs=self._current_obs,
                act=act,
                reward=reward,
                cost=cost,
                done=torch.logical_and(terminated, torch.logical_xor(terminated, truncated)),
                next_obs=real_next_obs,
                env_id=env_id,
                env_step=env_step,
            )

            self._global_env_step += num_envs
            self._current_obs = next_obs

    def _log_value(self, reward: torch.Tensor, cost: torch.Tensor, info: dict[str, Any]) -> None:
        self._ep_ret += info.get("original_reward", reward).cpu()
        self._ep_cost += info.get("original_cost", cost).cpu()
        self._ep_len += 1

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        if hasattr(self._env, "spec_log"):
            self._env.spec_log(logger)
        logger.store(
            {
                "Metrics/EpRet": self._ep_ret[idx],
                "Metrics/EpCost": self._ep_cost[idx],
                "Metrics/EpLen": self._ep_len[idx],
            }
        )

    def _reset_log(self, idx: int | None = None) -> None:
        if idx is None:
            self._ep_ret = torch.zeros(self._env.num_envs)
            self._ep_cost = torch.zeros(self._env.num_envs)
            self._ep_len = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[idx] = 0.0
            self._ep_cost[idx] = 0.0
            self._ep_len[idx] = 0.0

    def flush_qc_logs(self) -> None:
        """Flush remaining qc logs to disk."""
        if getattr(self, "_qc_buf", None) and len(self._qc_buf) > 0:
            big = torch.cat(self._qc_buf, dim=0).cpu().numpy()
            self._qc_buf.clear()
            with open(self._qc_path, "a") as f:
                np.savetxt(
                    f,
                    big,
                    delimiter=",",
                    fmt=["%.0f", "%.0f", "%.10g", "%.10g"],
                )
