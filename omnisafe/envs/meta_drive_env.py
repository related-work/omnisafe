# Copyright 2024 OmniSafe Team. All Rights Reserved.
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
"""Environments interface for MetaDrive Safety Environments.

Reference: https://metadrive-simulator.readthedocs.io/en/latest/rl_environments.html#safety-environments

The official MetaDrive ``SafeMetaDriveEnv`` accepts a flat config dict. Key parameters:

    - ``accident_prob`` (float): Probability of accident objects on each block. Default 0.8.
    - ``num_scenarios`` (int): Number of distinct scenarios. Default 100.
    - ``start_seed`` (int): Starting seed for scenario generation. Default 0.
    - ``traffic_density`` (float): Density of traffic vehicles. Default 0.05.
    - ``traffic_mode``: Traffic mode, ``Trigger`` or ``Respawn``.
    - ``horizon`` (int or None): Max steps per episode. None means unlimited.
    - ``crash_vehicle_cost``, ``crash_object_cost``, ``out_of_road_cost`` (float): Cost signals.
    - ``crash_vehicle_done``, ``crash_object_done``, ``out_of_road_done`` (bool): Termination flags.

The ``step()`` method returns ``(obs, reward, terminated, truncated, info)`` where
``info['cost']`` is the per-step cost signal and ``info['arrive_dest']`` indicates success.

For ``num_envs > 1``, subprocess-based AsyncVectorEnv is used because MetaDrive uses a
singleton 3D engine per process.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import torch

from omnisafe.common.logger import Logger
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU

META_DRIVE_AVAILABLE = True
try:
    from metadrive import SafeMetaDriveEnv as _SafeMetaDriveEnv
except ImportError:
    _SafeMetaDriveEnv = None
    META_DRIVE_AVAILABLE = False

GYM_VECTOR_AVAILABLE = True
try:
    from gymnasium.vector import AsyncVectorEnv
except ImportError:
    AsyncVectorEnv = None
    GYM_VECTOR_AVAILABLE = False


def _make_metadrive_env(
    config: dict[str, Any],
    rank: int,
    seed: int | None,
) -> Any:
    """Factory function for a single MetaDrive instance (runs in subprocess).

    Each subprocess gets a unique copy of the config with an offset start_seed
    so that parallel envs explore different scenarios.
    """
    cfg = dict(config)
    start_seed = cfg.get('start_seed', 0)
    num_scenarios = cfg.get('num_scenarios', 1)
    cfg['start_seed'] = start_seed + rank * num_scenarios
    env = _SafeMetaDriveEnv(config=cfg)
    env.logger.setLevel(logging.FATAL)
    if seed is not None:
        env.reset(seed=cfg['start_seed'] + seed % num_scenarios)
    return env


def _make_env_fn(config: dict[str, Any], rank: int):
    """Return a callable that creates a MetaDrive env for the given rank.

    This avoids the late-binding closure issue with lambda in a loop.
    """
    return lambda: _make_metadrive_env(config, rank, None)


@env_register
class SafetyMetaDriveEnv(CMDP):
    """Wrapper for MetaDrive Safety Environments.

    Supports two env_id aliases:
        - ``SafeMetaDrive`` — legacy omnisafe name
        - ``MetaDriveSafe-v0`` — standard naming convention

    The metadrive config dict can be passed via either ``config`` or ``meta_drive_config`` key
    in ``env_cfgs``.

    When ``num_envs > 1``, uses ``gymnasium.vector.AsyncVectorEnv`` to run each MetaDrive
    instance in its own subprocess (required because MetaDrive uses a singleton 3D engine).

    Args:
        env_id (str): Environment id.
        num_envs (int): Number of parallel environments. Defaults to 1.
        device (torch.device): Device for tensor outputs. Defaults to CPU.

    Keyword Args:
        config (dict): MetaDrive configuration dict (preferred key name).
        meta_drive_config (dict): Legacy key name, same as ``config``.
    """

    need_auto_reset_wrapper: bool = True  # set to False below when num_envs > 1
    need_time_limit_wrapper: bool = False

    _support_envs: ClassVar[list[str]] = [
        'SafeMetaDrive',
        'MetaDriveSafe-v0',
    ]

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,
    ) -> None:
        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = torch.device(device)

        if not META_DRIVE_AVAILABLE:
            raise ImportError(
                'Please install MetaDrive to use SafeMetaDrive!\n'
                'Install from PyPI: `pip install metadrive-simulator`.\n'
                'More details: https://github.com/metadriverse/metadrive.',
            )

        # Merge both config keys: 'config' (new) and 'meta_drive_config' (legacy).
        # 'config' takes priority for overlapping keys.
        md_config: dict[str, Any] = {**(kwargs.get('meta_drive_config') or {}), **(kwargs.get('config') or {})}

        if num_envs > 1:
            if not GYM_VECTOR_AVAILABLE:
                raise ImportError(
                    'gymnasium.vector.AsyncVectorEnv is required for num_envs > 1.\n'
                    'Install with: `pip install gymnasium`',
                )
            # Subprocess-based vectorized envs:
            # AutoReset is not needed because AsyncVectorEnv handles reset internally.
            self.need_auto_reset_wrapper = False
            self._is_vector = True
            self._start_seed = md_config.get('start_seed', 0)
            env_fns = [
                _make_env_fn(md_config, i)
                for i in range(num_envs)
            ]
            self._env: Any = AsyncVectorEnv(env_fns, shared_memory=False)
            self._env.reset()  # eager init of subprocesses

            sample_env = _SafeMetaDriveEnv(config=md_config)
            self._num_scenarios = sample_env.config['num_scenarios']
            sample_env.close()
        else:
            self._is_vector = False
            self._start_seed = md_config.get('start_seed', 0)
            self._env = _SafeMetaDriveEnv(config=md_config)
            self._env.logger.setLevel(logging.FATAL)
            self._num_scenarios = self._env.config['num_scenarios']

        self._action_space = self._env.single_action_space if self._is_vector else self._env.action_space
        self._observation_space = self._env.single_observation_space if self._is_vector else self._env.observation_space
        self._metadata = self._env.metadata
        self._last_episode_flags: dict[str, float] = {}
        self._metric_keys_registered: bool = False

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Step the environment.

        Args:
            action (torch.Tensor): Action to take. Shape: (num_envs, act_dim).

        Returns:
            observation, reward, cost, terminated, truncated, info
        """
        if self._is_vector:
            action_np = action.detach().cpu().numpy()
            obs, reward, terminated, truncated, info = self._env.step(action_np)
            cost = info['cost']
            obs, reward, cost, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=self._device)
                for x in (obs, reward, cost, terminated, truncated)
            )
            if 'final_observation' in info:
                fo = info['final_observation']
                if isinstance(fo, np.ndarray):
                    # AsyncVectorEnv returns an object array; convert each element
                    fo = np.array([
                        a if a is not None else np.zeros(obs.shape[-1])
                        for a in fo
                    ])
                    fo = torch.as_tensor(fo, dtype=torch.float32, device=self._device)
                info['final_observation'] = fo
            self._capture_episode_flags_multi(info, terminated, truncated)
            return obs, reward, cost, terminated, truncated, info

        obs, reward, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy(),
        )
        cost = info['cost']
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        if 'final_observation' in info:
            info['final_observation'] = np.array(
                [
                    array if array is not None else np.zeros(obs.shape[-1])
                    for array in info['final_observation']
                ],
            )
            info['final_observation'] = torch.as_tensor(
                info['final_observation'],
                dtype=torch.float32,
                device=self._device,
            )
        self._capture_episode_flags(info)
        return obs, reward, cost, terminated, truncated, info

    def _capture_episode_flags(self, info: dict[str, Any]) -> None:
        """Record per-step episode flags, matching the research_extensions convention."""
        arrive_dest = float(bool(info.get('arrive_dest', False)))
        crash_vehicle = float(bool(info.get('crash_vehicle', False)))
        crash_object = float(bool(info.get('crash_object', False)))
        out_of_road = float(bool(info.get('out_of_road', False)))
        self._last_episode_flags = {
            'Metrics/Success': arrive_dest,
            'Metrics/SuccessRate': arrive_dest,
            'MetaDrive/ArriveDest': arrive_dest,
            'MetaDrive/CrashVehicle': crash_vehicle,
            'MetaDrive/CrashObject': crash_object,
            'MetaDrive/OutOfRoad': out_of_road,
        }

    def _capture_episode_flags_multi(self, info: dict[str, Any], terminated: torch.Tensor, truncated: torch.Tensor) -> None:
        """Record per-step episode flags for vectorized envs.
        Only captures flags from the last env that terminated/truncated in this step,
        which matches how spec_log is called (per-env termination).
        """
        for i in range(self._num_envs):
            if terminated[i] or truncated[i]:
                arrive_dest = float(bool(np.asarray(info.get('arrive_dest', [False] * self._num_envs))[i]))
                crash_vehicle = float(bool(np.asarray(info.get('crash_vehicle', [False] * self._num_envs))[i]))
                crash_object = float(bool(np.asarray(info.get('crash_object', [False] * self._num_envs))[i]))
                out_of_road = float(bool(np.asarray(info.get('out_of_road', [False] * self._num_envs))[i]))
                # Overwrite: spec_log uses the last terminated env's flags
                self._last_episode_flags = {
                    'Metrics/Success': arrive_dest,
                    'Metrics/SuccessRate': arrive_dest,
                    'MetaDrive/ArriveDest': arrive_dest,
                    'MetaDrive/CrashVehicle': crash_vehicle,
                    'MetaDrive/CrashObject': crash_object,
                    'MetaDrive/OutOfRoad': out_of_road,
                }

    def spec_log(self, logger: Logger) -> None:
        """Emit MetaDrive episode-level success and safety metrics, consistent with
        the research_extensions convention."""
        if not self._last_episode_flags:
            return
        if not self._metric_keys_registered:
            for key in self._last_episode_flags:
                logger.register_key(key)
            self._metric_keys_registered = True
        for key, value in self._last_episode_flags.items():
            logger.store({key: value})

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment.

        Args:
            seed (int or None): Seed to reset the environment.
            options (dict or None): Additional options.

        Returns:
            observation, info
        """
        if self._is_vector:
            if seed is not None:
                seeds = [
                    self._start_seed + i * self._num_scenarios + seed % self._num_scenarios
                    for i in range(self._num_envs)
                ]
            else:
                seeds = None
            obs, info = self._env.reset(seed=seeds, options=options)
            return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

        obs, info = self._env.reset(seed=seed)
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    def set_seed(self, seed: int) -> None:
        """Set the seed for the environment.

        Maps the training seed to valid MetaDrive scenario indices within
        [start_seed, start_seed + num_scenarios) for each parallel env.
        """
        if self._is_vector:
            seeds = [
                self._start_seed + i * self._num_scenarios + seed % self._num_scenarios
                for i in range(self._num_envs)
            ]
            self._env.reset(seed=seeds)
            return

        scenario_seed = self._start_seed + seed % self._num_scenarios
        self.reset(seed=scenario_seed)

    def render(self) -> Any:
        """Render a frame."""
        return self._env.render()

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
