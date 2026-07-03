#!/usr/bin/env python3
"""Optuna search dedicated to WCSAC-IQN."""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch
import yaml

try:
    from hpo import run_wcsac_hpo as common
except ModuleNotFoundError:
    import run_wcsac_hpo as common


ALGO = 'WCSAC_IQN'


def suggest_params(trial: 'optuna.Trial') -> dict[str, Any]:
    """Sample WCSAC-IQN and distributional critic hyperparameters."""
    learning_rate = trial.suggest_categorical(
        'learning_rate',
        [1e-4, 3e-4, 1e-3],
    )
    lr_scale = trial.suggest_categorical(
        'cost_penalty_lr_scale',
        [1.0, 10.0, 50.0],
    )
    return {
        'model_cfgs:actor:lr': learning_rate,
        'model_cfgs:critic:lr': learning_rate,
        'algo_cfgs:batch_size': trial.suggest_categorical(
            'batch_size',
            [64, 128, 256, 512],
        ),
        'algo_cfgs:polyak': trial.suggest_float('polyak', 0.001, 0.02),
        'algo_cfgs:cost_penalty_lr_scale': lr_scale,
        'lagrange_cfgs:lambda_lr': learning_rate * lr_scale,
        'lagrange_cfgs:lagrangian_multiplier_init': trial.suggest_categorical(
            'lagrangian_multiplier_init',
            [0.3, 0.693147, 1.0],
        ),
        'algo_cfgs:iqn_n_quantiles': trial.suggest_categorical(
            'iqn_n_quantiles',
            [8, 16, 32, 64],
        ),
        'algo_cfgs:iqn_kappa': trial.suggest_float('iqn_kappa', 0.1, 2.0),
        'algo_cfgs:cvar_quantile_samples': trial.suggest_categorical(
            'cvar_quantile_samples',
            [8, 16, 32, 64],
        ),
        'model_cfgs:critic:iqn_embedding_dim': trial.suggest_categorical(
            'iqn_embedding_dim',
            [32, 64, 128],
        ),
    }


def _resolve_devices(args: argparse.Namespace) -> tuple[list[int], int, int]:
    """Resolve available GPUs and Optuna worker count."""
    if args.cpu:
        return [], 1, 1

    available = set(range(torch.cuda.device_count()))
    requested = (
        [int(gpu.strip()) for gpu in args.gpus.split(',')]
        if args.gpus
        else common.AVAILABLE_GPUS
    )
    gpus = [gpu for gpu in requested if gpu in available]
    if not gpus:
        print('未检测到可用 GPU，自动回退到 CPU。')
        return [], 1, 1

    seed_workers = min(max(args.seed_workers, 1), len(common.SEEDS), len(gpus))
    max_trial_jobs = max(len(gpus) // seed_workers, 1)
    requested_jobs = args.parallel if args.parallel else min(common.N_JOBS, max_trial_jobs)
    n_jobs = min(requested_jobs, max_trial_jobs)
    if requested_jobs != n_jobs:
        print(
            f'每个 trial 并行 {seed_workers} 个 seed，'
            f'{len(gpus)} 张 GPU 最多同时跑 {n_jobs} 个 trial。',
        )
    return gpus, n_jobs, seed_workers


def main() -> None:
    """Run WCSAC-IQN HPO."""
    parser = argparse.ArgumentParser(description='WCSAC-IQN HPO with Optuna')
    parser.add_argument('--env', default='SafetyPointCircle2-v0', help='环境，逗号分隔')
    parser.add_argument('--trials', type=int, default=common.N_TRIALS)
    parser.add_argument('--gpus', default=None, help='GPU 编号，逗号分隔')
    parser.add_argument('--parallel', type=int, default=None)
    parser.add_argument('--seed-workers', type=int, default=len(common.SEEDS))
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument(
        '--run-id',
        default=None,
        help='运行标识，传入后结果存到 OUTPUT_DIR/<run-id>/ 子目录，与历史运行隔离',
    )
    parser.add_argument(
        '--cost-limit',
        type=float,
        default=None,
        help='cost 约束阈值（默认见 common.COST_LIMIT，模块顶部）',
    )
    args = parser.parse_args()

    if args.cost_limit is not None:
        common.COST_LIMIT = args.cost_limit

    envs = [env.strip() for env in args.env.split(',')]
    gpus, n_jobs, seed_workers = _resolve_devices(args)
    output_dir = (
        os.path.join(common.OUTPUT_DIR, args.run_id)
        if args.run_id
        else common.OUTPUT_DIR
    )
    os.makedirs(output_dir, exist_ok=True)

    results = [
        common.run_hpo_for_env(
            ALGO,
            env_id,
            args.trials,
            output_dir,
            gpus,
            n_jobs,
            param_suggester=suggest_params,
            seed_workers=seed_workers,
        )
        for env_id in envs
    ]
    summary_path = os.path.join(output_dir, 'wcsac_iqn_hpo_summary.yaml')
    with open(summary_path, 'w', encoding='utf-8') as file:
        yaml.safe_dump(results, file, allow_unicode=True, sort_keys=False)
    print(f'WCSAC-IQN HPO 完成，结果保存到 {summary_path}')


if __name__ == '__main__':
    main()
