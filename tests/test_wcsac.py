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
"""Tests for the Gaussian WCSAC migration."""

import math
from types import SimpleNamespace

import torch
from gymnasium.spaces import Box

from omnisafe.algorithms.off_policy.wcsac import WCSAC
from omnisafe.algorithms.off_policy.wcsac_iqn import WCSAC_IQN
from omnisafe.models.actor_critic.wcsac_actor_q_critic import WCSACActorQCritic
from omnisafe.models.actor_critic.wcsac_iqn_actor_q_critic import WCSACIQNActorQCritic
from omnisafe.utils.config import Config


def _model_cfgs() -> Config:
    return Config(
        weight_initialization_mode='xavier_uniform',
        actor_type='gaussian_sac',
        linear_lr_decay=False,
        actor=Config(hidden_sizes=[32, 32], activation='relu', lr=1e-3),
        critic=Config(hidden_sizes=[32, 32], activation='relu', lr=1e-3, num_critics=2),
    )


def test_wcsac_cost_critics_are_independent() -> None:
    """Cost mean and variance must not share parameters."""
    model = WCSACActorQCritic(
        obs_space=Box(low=-1.0, high=1.0, shape=(4,)),
        act_space=Box(low=-1.0, high=1.0, shape=(2,)),
        model_cfgs=_model_cfgs(),
        epochs=10,
    )

    mean_params = {id(param) for param in model.cost_critic.parameters()}
    var_params = {id(param) for param in model.cost_var_critic.parameters()}
    assert mean_params.isdisjoint(var_params)

    obs = torch.randn(8, 4)
    act = torch.randn(8, 2)
    assert model.cost_critic(obs, act)[0].shape == (8,)
    assert model.cost_var_critic(obs, act)[0].shape == (8,)


def test_wcsac_softplus_initialization() -> None:
    """Inverse softplus must reproduce the reference alpha/beta initialization."""
    expected = math.log(2.0)
    raw = WCSAC._inverse_softplus(expected)
    actual = torch.nn.functional.softplus(torch.tensor(raw)).item()
    assert math.isclose(actual, expected, rel_tol=1e-6)


def test_wcsac_target_cost_critics_follow_polyak_update() -> None:
    """Both cost target networks must track their online counterparts."""
    model = WCSACActorQCritic(
        obs_space=Box(low=-1.0, high=1.0, shape=(4,)),
        act_space=Box(low=-1.0, high=1.0, shape=(2,)),
        model_cfgs=_model_cfgs(),
        epochs=10,
    )

    with torch.no_grad():
        for param in model.cost_critic.parameters():
            param.add_(1.0)
        for param in model.cost_var_critic.parameters():
            param.sub_(1.0)

    model.polyak_update(1.0)

    for source, target in (
        (model.cost_critic, model.target_cost_critic),
        (model.cost_var_critic, model.target_cost_var_critic),
    ):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            assert torch.equal(source_param, target_param)


def test_wcsac_iqn_model_and_target_update() -> None:
    """IQN must own a registered cost critic and matching target."""
    cfg = _model_cfgs()
    cfg.critic['iqn_embedding_dim'] = 16
    model = WCSACIQNActorQCritic(
        obs_space=Box(low=-1.0, high=1.0, shape=(4,)),
        act_space=Box(low=-1.0, high=1.0, shape=(2,)),
        model_cfgs=cfg,
        epochs=10,
    )
    obs = torch.randn(8, 4)
    act = torch.randn(8, 2)
    tau = torch.rand(8, 7)
    assert model.cost_critic(obs, act, tau).shape == (8, 7, 1)

    with torch.no_grad():
        for param in model.cost_critic.parameters():
            param.add_(1.0)
    model.polyak_update(1.0)
    for source_param, target_param in zip(
        model.cost_critic.parameters(),
        model.target_cost_critic.parameters(),
    ):
        assert torch.equal(source_param, target_param)


def test_wcsac_iqn_samples_upper_tail() -> None:
    """With alpha=0.9, IQN CVaR samples must all lie in [0.9, 1]."""

    class RecordingCritic:
        def __init__(self) -> None:
            self.tau: torch.Tensor | None = None

        def __call__(
            self,
            obs: torch.Tensor,
            act: torch.Tensor,
            tau: torch.Tensor,
        ) -> torch.Tensor:
            del obs, act
            self.tau = tau
            return tau.unsqueeze(-1)

    algo = object.__new__(WCSAC_IQN)
    critic = RecordingCritic()
    algo._actor_critic = SimpleNamespace(cost_critic=critic)
    algo._cfgs = Config(
        algo_cfgs=Config(cvar_quantile_samples=128),
        lagrange_cfgs=Config(cvar_alpha=0.9),
    )
    cvar = algo._compute_cvar(torch.zeros(4, 3), torch.zeros(4, 2))

    assert critic.tau is not None
    assert torch.all(critic.tau >= 0.9)
    assert torch.all(critic.tau <= 1.0)
    assert cvar.shape == (4, 1)
