#!/usr/bin/env python3
"""Run one WCSAC or WCSAC-IQN training job."""

from __future__ import annotations

import argparse
import os
from typing import Any

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import omnisafe
import torch


ALGOS = ('WCSAC', 'WCSAC_IQN')
DEFAULT_ENV_ID = 'SafetyPointCircle2-v0'
STEPS_PER_EPOCH = 10_000


def build_custom_cfgs(
    algo: str,
    env_id: str,
    seed: int,
    device: str,
    total_steps: int,
    save_every_steps: int,
    log_root: str,
) -> dict[str, Any]:
    """Build the configuration for one training run."""
    if total_steps <= 0:
        raise ValueError('total_steps must be positive')
    if save_every_steps <= 0 or save_every_steps % STEPS_PER_EPOCH != 0:
        raise ValueError(
            f'save_every_steps must be a positive multiple of {STEPS_PER_EPOCH}',
        )

    log_dir = os.path.join(log_root, algo, env_id)
    custom_cfgs: dict[str, Any] = {
        'seed': seed,
        'train_cfgs': {
            'device': device,
            'total_steps': total_steps,
            'torch_threads': 1,
        },
        'algo_cfgs': {
            'steps_per_epoch': STEPS_PER_EPOCH,
            'update_cycle': 100,
            'update_iters': 100,
            'start_learning_steps': 500,
            'reward_normalize': False,
            'cost_normalize': False,
            'alpha': 0.693147,
            'auto_alpha': True,
            'policy_delay': 1,
            'warmup_epochs': 0,
            'max_ep_len': 1000,
            'cost_penalty_lr_scale': 50.0,
        },
        'model_cfgs': {
            'weight_initialization_mode': 'xavier_uniform',
            'actor': {
                'hidden_sizes': [256, 256],
                'activation': 'relu',
                'lr': 1e-3,
            },
            'critic': {
                'hidden_sizes': [256, 256],
                'activation': 'relu',
                'lr': 1e-3,
            },
        },
        'logger_cfgs': {
            'use_wandb': False,
            'use_tensorboard': True,
            'log_dir': log_dir,
            'save_model_freq': save_every_steps // STEPS_PER_EPOCH,
        },
        'lagrange_cfgs': {
            'cost_limit': 5.0,
            'lagrangian_multiplier_init': 0.693147,
            'lambda_lr': 0.05,
            'lambda_optimizer': 'Adam',
            'cvar_alpha': 0.9,
        },
    }

    if algo == 'WCSAC_IQN':
        custom_cfgs['algo_cfgs'].update(
            {
                'iqn_n_quantiles': 32,
                'iqn_kappa': 1.0,
                'cvar_quantile_samples': 32,
            },
        )
        custom_cfgs['model_cfgs']['critic']['iqn_embedding_dim'] = 64

    return custom_cfgs


def main() -> None:
    """Parse arguments and run one training job."""
    parser = argparse.ArgumentParser(description='Single WCSAC/WCSAC-IQN training run')
    parser.add_argument('--algo', choices=ALGOS, default='WCSAC')
    parser.add_argument('--env', default=DEFAULT_ENV_ID)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--save-every-steps', type=int, default=500_000)
    parser.add_argument('--log-root', default='./runs/single')
    args = parser.parse_args()

    use_cuda = (
        not args.cpu
        and torch.cuda.is_available()
        and 0 <= args.gpu < torch.cuda.device_count()
    )
    device = f'cuda:{args.gpu}' if use_cuda else 'cpu'
    custom_cfgs = build_custom_cfgs(
        algo=args.algo,
        env_id=args.env,
        seed=args.seed,
        device=device,
        total_steps=args.steps,
        save_every_steps=args.save_every_steps,
        log_root=args.log_root,
    )

    print(f'Starting {args.algo} on {args.env}')
    print(f'  device: {device}')
    print(f'  seed: {args.seed}')
    print(f'  total steps: {args.steps:,}')
    print(f'  save every: {args.save_every_steps:,} steps')
    print(f'  log directory: {custom_cfgs["logger_cfgs"]["log_dir"]}')

    agent = omnisafe.Agent(args.algo, args.env, custom_cfgs=custom_cfgs)
    reward, cost, ep_len = agent.learn()
    print(f'Finished: reward={reward:.2f}, cost={cost:.2f}, ep_len={ep_len:.2f}')


if __name__ == '__main__':
    main()
