#!/usr/bin/env python3
"""
WCSAC / WCSAC_IQN 超参数自动搜索（Optuna）。

每个 trial 在 3 个随机种子 [111, 222, 333] 下各训练 100w 步，
取平均 reward/cost 作为目标值。约束：mean_cost ≤ 5.0。

用法:
    python examples/benchmarks/run_wcsac_hpo.py          # 所有环境
    python examples/benchmarks/run_wcsac_hpo.py --env SafetyPointGoal1-v0  # 单个环境
    python examples/benchmarks/run_wcsac_hpo.py --algo WCSAC               # 单个算法
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np
import yaml

import omnisafe


# ==============================================================================
# 配置
# ==============================================================================

ENV_IDS = [
    'SafetyPointGoal1-v0',
    'SafetyPointCircle2-v0',
    'SafetyHopperVelocity-v1',
    'SafetyHumanoidVelocity-v1',
    'SafetyAntVelocity-v1',
    'SafetyPointButton1-v0',
]

ALGOS = ['WCSAC', 'WCSAC_IQN']

# 每个 trial 的训练步数（全量）
TOTAL_STEPS = 1_000_000

# 每个 trial 的随机种子列表
SEEDS = [111, 222, 333]

# 默认使用所有可用 GPU，设为 [] 或 None 则用 CPU
AVAILABLE_GPUS = list(range(8))  # 8 张卡

# 并行 trial 数（建议 = GPU 数量，每卡最多跑 1 个 trial）
N_JOBS = 8

# 每个 (algo, env) 组合的 trial 数
N_TRIALS = 30

# 结果保存目录
OUTPUT_DIR = './hpo_results'

# ---- 固定不变的参数 ----
COST_LIMIT = 5.0
ALPHA = 0.693147     # 原实现 softplus(0)
GAMMA = 0.99         # 折扣因子
HIDDEN_SIZES = [256, 256]


# ==============================================================================
# 超参数搜索空间
# ==============================================================================

def _build_trial_name(trial_number: int) -> str:
    """构建简洁的 TensorBoard trial 目录名。"""
    return f'trial_{trial_number:03d}'


def suggest_params(trial: 'optuna.Trial', algo: str) -> dict[str, Any]:
    """为一个 trial 采样超参数。WCSAC: 6 参数，WCSAC_IQN: 10 参数。"""
    params = {
        # 原实现对 actor、critics 和 alpha 使用相同基础学习率。
        'model_cfgs:actor:lr': trial.suggest_categorical(
            'learning_rate', [1e-4, 3e-4, 1e-3],
        ),
        # 批量大小
        'algo_cfgs:batch_size': trial.suggest_categorical(
            'batch_size', [64, 128, 256, 512],
        ),
        # 目标网络软更新系数
        'algo_cfgs:polyak': trial.suggest_float('polyak', 0.001, 0.02),
        # Lagrange 乘子初始值
        'lagrange_cfgs:lagrangian_multiplier_init': trial.suggest_categorical(
            'lagrangian_multiplier_init', [0.3, 0.693147, 1.0],
        ),
    }
    params['model_cfgs:critic:lr'] = params['model_cfgs:actor:lr']
    lr_scale = trial.suggest_categorical('cost_penalty_lr_scale', [1.0, 10.0, 50.0])
    params['algo_cfgs:cost_penalty_lr_scale'] = lr_scale
    params['lagrange_cfgs:lambda_lr'] = params['model_cfgs:actor:lr'] * lr_scale
    if algo == 'WCSAC_IQN':
        params.update({
            'algo_cfgs:iqn_n_quantiles': trial.suggest_categorical(
                'iqn_n_quantiles', [8, 16, 32, 64],
            ),
            'algo_cfgs:iqn_kappa': trial.suggest_float(
                'iqn_kappa', 0.1, 2.0,
            ),
            'algo_cfgs:cvar_quantile_samples': trial.suggest_categorical(
                'cvar_quantile_samples', [8, 16, 32, 64],
            ),
            'model_cfgs:critic:iqn_embedding_dim': trial.suggest_categorical(
                'iqn_embedding_dim', [32, 64, 128],
            ),
        })
    return params


def make_custom_cfgs(
    algo: str,
    env_id: str,
    params: dict[str, Any],
    log_dir: str,
    seed: int,
    gpu_id: int | None = None,
) -> dict[str, Any]:
    """根据搜索参数构建 OmniSafe custom_cfgs 字典。"""
    device = f'cuda:{gpu_id}' if gpu_id is not None else 'cpu'
    custom_cfgs: dict[str, Any] = {
        'train_cfgs': {
            'device': device,
            'total_steps': TOTAL_STEPS,
            'torch_threads': 1,
        },
        'algo_cfgs': {
            'gamma': GAMMA,
            'alpha': ALPHA,
            'auto_alpha': True,
            'cost_normalize': False,
            'policy_delay': 1,
            'warmup_epochs': 0,
            'max_ep_len': 1000,
            'update_cycle': 100,
            'update_iters': 100,
            'start_learning_steps': 500,
            'steps_per_epoch': 10000,
        },
        'model_cfgs': {
            'weight_initialization_mode': 'xavier_uniform',
            'actor': {
                'hidden_sizes': HIDDEN_SIZES,
                'activation': 'relu',
            },
            'critic': {
                'hidden_sizes': HIDDEN_SIZES,
                'activation': 'relu',
            },
        },
        'logger_cfgs': {
            'use_wandb': False,
            'use_tensorboard': True,
            'log_dir': log_dir,
            # 50 epochs * 10,000 steps_per_epoch = 500,000 environment steps.
            'save_model_freq': 50,
        },
        'lagrange_cfgs': {
            'cost_limit': COST_LIMIT,
            'lambda_optimizer': 'Adam',
            'cvar_alpha': 0.9,
        },
        'seed': seed,
    }

    # 将搜索参数写入对应段
    for key, value in params.items():
        if key.startswith('algo_cfgs:'):
            custom_cfgs['algo_cfgs'][key.split(':')[1]] = value
        elif key.startswith('model_cfgs:'):
            section, field = key.split(':')[1], key.split(':')[2]
            custom_cfgs['model_cfgs'][section][field] = value
        elif key.startswith('lagrange_cfgs:'):
            custom_cfgs['lagrange_cfgs'][key.split(':')[1]] = value

    return custom_cfgs


def run_single_seed(
    algo: str,
    env_id: str,
    params: dict[str, Any],
    log_dir: str,
    seed: int,
    gpu_id: int | None = None,
) -> tuple[float, float]:
    """单种子训练，返回 (reward, cost)。"""
    import torch
    custom_cfgs = make_custom_cfgs(algo, env_id, params, log_dir, seed, gpu_id)

    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    reward, cost, _ep_len = agent.learn()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(reward), float(cost)


def _run_seed_job(
    job: tuple[str, str, dict[str, Any], str, int, int | None],
) -> tuple[int, float, float]:
    """Run one seed in a child process."""
    algo, env_id, params, log_dir, seed, gpu_id = job
    reward, cost = run_single_seed(algo, env_id, params, log_dir, seed, gpu_id)
    return seed, reward, cost


def _select_seed_gpus(
    trial_number: int,
    gpus: list[int],
    seed_workers: int,
    max_parallel_trials: int,
) -> list[int | None]:
    """Assign one GPU per seed while keeping concurrent trials disjoint."""
    if not gpus:
        return [None for _ in SEEDS]
    if seed_workers <= 1:
        return [gpus[trial_number % len(gpus)] for _ in SEEDS]

    max_parallel_trials = max(max_parallel_trials, 1)
    trial_slot = trial_number % max_parallel_trials
    start = trial_slot * seed_workers
    gpu_chunk = gpus[start:start + seed_workers]
    if not gpu_chunk:
        gpu_chunk = gpus[:seed_workers]
    return [gpu_chunk[idx % len(gpu_chunk)] for idx, _seed in enumerate(SEEDS)]


def objective(
    trial: 'optuna.Trial',
    algo: str,
    env_id: str,
    base_log_dir: str,
    gpus: list[int],
    param_suggester: Callable[[Any], dict[str, Any]] | None = None,
    seed_workers: int = 1,
    max_parallel_trials: int = 1,
) -> float:
    """目标函数：3 个种子取平均，cost 超标时给惩罚。"""
    params = (
        param_suggester(trial)
        if param_suggester is not None
        else suggest_params(trial, algo)
    )

    # TensorBoard 只用 trial 和 seed 区分运行，参数保留在 Optuna 中。
    trial_dir = _build_trial_name(trial.number)
    seed_gpus = _select_seed_gpus(
        trial.number,
        gpus,
        seed_workers,
        max_parallel_trials,
    )

    rewards = []
    costs = []

    jobs = [
        (
            algo,
            env_id,
            params,
            os.path.join(base_log_dir, trial_dir, f'seed_{seed:03d}'),
            seed,
            seed_gpus[idx],
        )
        for idx, seed in enumerate(SEEDS)
    ]
    seed_workers = max(1, min(seed_workers, len(SEEDS)))
    if seed_workers == 1:
        for job in jobs:
            seed = job[4]
            try:
                _seed, reward, cost = _run_seed_job(job)
                rewards.append(reward)
                costs.append(cost)
            except Exception as exc:
                print(f'[Trial {trial.number}] seed={seed} 异常: {exc}')
                rewards.append(-1e6)
                costs.append(1e6)
    else:
        context = mp.get_context('spawn')
        with ProcessPoolExecutor(
            max_workers=seed_workers,
            mp_context=context,
        ) as executor:
            futures = {executor.submit(_run_seed_job, job): job[4] for job in jobs}
            results: dict[int, tuple[float, float]] = {}
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    _seed, reward, cost = future.result()
                    results[_seed] = (reward, cost)
                except Exception as exc:
                    print(f'[Trial {trial.number}] seed={seed} 异常: {exc}')
                    results[seed] = (-1e6, 1e6)
            for seed in SEEDS:
                reward, cost = results[seed]
                rewards.append(reward)
                costs.append(cost)

    mean_reward = float(np.mean(rewards))
    mean_cost = float(np.mean(costs))
    std_reward = float(np.std(rewards))
    std_cost = float(np.std(costs))

    # 保存结果
    trial.set_user_attr('mean_reward', mean_reward)
    trial.set_user_attr('mean_cost', mean_cost)
    trial.set_user_attr('std_reward', std_reward)
    trial.set_user_attr('std_cost', std_cost)
    trial.set_user_attr('full_params', params)

    # 目标：cost 在限制内最大化 reward，超标给惩罚
    if mean_cost <= COST_LIMIT:
        value = mean_reward
    else:
        penalty = 1000.0 * (mean_cost / COST_LIMIT - 1.0)
        value = mean_reward - penalty

    print(
        f'[Trial {trial.number:03d}] algo={algo}  env={env_id}\n'
        f'  params: actor_lr={params["model_cfgs:actor:lr"]:.1e}  '
        f'critic_lr={params["model_cfgs:critic:lr"]:.1e}  '
        f'batch={params["algo_cfgs:batch_size"]}  '
        f'polyak={params["algo_cfgs:polyak"]:.4f}\n'
        f'           lambda_lr={params["lagrange_cfgs:lambda_lr"]:.2e}  '
        f'mult_init={params["lagrange_cfgs:lagrangian_multiplier_init"]}\n'
        f'  reward: {rewards}  ->  mean={mean_reward:.2f} ± {std_reward:.2f}\n'
        f'  cost:   {[f"{c:.2f}" for c in costs]}  ->  mean={mean_cost:.2f} ± {std_cost:.2f}\n'
        f'  value={value:.2f}',
    )
    return value


def run_hpo_for_env(
    algo: str,
    env_id: str,
    n_trials: int,
    output_dir: str,
    gpus: list[int],
    n_jobs: int,
    param_suggester: Callable[[Any], dict[str, Any]] | None = None,
    seed_workers: int = 1,
) -> dict[str, Any]:
    """对单个 (algo, env_id) 组合运行 HPO。"""
    import optuna

    study_name = f'{algo}_{env_id}'
    storage_path = os.path.join(output_dir, f'{study_name}.db')

    print(f'\n{"=" * 60}')
    print(f'启动 HPO: {study_name}')
    print(f'Trials: {n_trials}  |  步数: {TOTAL_STEPS:,}  |  Seeds: {SEEDS}')
    print(f'GPU: {gpus}  |  并行 trial: {n_jobs}  |  seed workers: {seed_workers}')
    print(
        '搜索参数: learning_rate, batch_size, polyak, '
        'cost_penalty_lr_scale, multiplier_init',
    )
    print(f'{"=" * 60}\n')

    study = optuna.create_study(
        study_name=study_name,
        storage=f'sqlite:///{storage_path}',
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    base_log_dir = os.path.join(output_dir, study_name)
    os.makedirs(base_log_dir, exist_ok=True)

    study.optimize(
        lambda trial: objective(
            trial,
            algo,
            env_id,
            base_log_dir,
            gpus,
            param_suggester,
            seed_workers,
            n_jobs,
        ),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    best = study.best_trial
    best_params = best.user_attrs.get('full_params', {})
    best_cfgs = make_custom_cfgs(algo, env_id, best_params, log_dir='./runs', seed=0)

    result = {
        'algo': algo,
        'env_id': env_id,
        'best_value': best.value,
        'mean_reward': best.user_attrs.get('mean_reward', float('nan')),
        'mean_cost': best.user_attrs.get('mean_cost', float('nan')),
        'std_reward': best.user_attrs.get('std_reward', float('nan')),
        'std_cost': best.user_attrs.get('std_cost', float('nan')),
        'best_params': best_params,
        'best_custom_cfgs': best_cfgs,
        'n_trials': len(study.trials),
    }

    print(f'\n--- 最佳: {study_name} ---')
    print(f'  Value:  {best.value:.2f}')
    print(f'  Reward: {result["mean_reward"]:.2f} ± {result["std_reward"]:.2f}')
    print(f'  Cost:   {result["mean_cost"]:.2f} ± {result["std_cost"]:.2f}')
    print(f'  Params: {best_params}')

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='WCSAC HPO with Optuna')
    parser.add_argument('--algo', type=str, default=None, help='只跑指定算法，逗号分隔多个')
    parser.add_argument('--env', type=str, default=None, help='只跑指定环境，逗号分隔多个')
    parser.add_argument('--trials', type=int, default=N_TRIALS, help='每个组合的 trial 数')
    parser.add_argument('--gpus', type=str, default=None, help='GPU 编号，逗号分隔（默认 0-7）')
    parser.add_argument('--parallel', type=int, default=None, help='并行 trial 数（默认 8）')
    parser.add_argument('--cpu', action='store_true', help='强制使用 CPU')
    parser.add_argument(
        '--run-id',
        default=None,
        help='运行标识，传入后结果存到 OUTPUT_DIR/<run-id>/ 子目录，与历史运行隔离',
    )
    parser.add_argument(
        '--cost-limit',
        type=float,
        default=None,
        help=f'cost 约束阈值（默认 {COST_LIMIT}）',
    )
    args = parser.parse_args()

    if args.cost_limit is not None:
        global COST_LIMIT
        COST_LIMIT = args.cost_limit

    algos = [a.strip() for a in args.algo.split(',')] if args.algo else ALGOS
    envs = [e.strip() for e in args.env.split(',')] if args.env else ENV_IDS
    n_trials = args.trials

    if args.cpu:
        gpus = []
        n_jobs = 1
    else:
        import torch

        available = list(range(torch.cuda.device_count()))
        requested = (
            [int(g.strip()) for g in args.gpus.split(',')]
            if args.gpus
            else AVAILABLE_GPUS
        )
        gpus = [gpu for gpu in requested if gpu in available]
        n_jobs = args.parallel if args.parallel else min(N_JOBS, max(len(gpus), 1))
        if not gpus:
            print('未检测到可用 GPU，自动回退到 CPU。')
            n_jobs = 1

    output_dir = (
        os.path.join(OUTPUT_DIR, args.run_id) if args.run_id else OUTPUT_DIR
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f'GPU 分配: {gpus if gpus else "CPU"}')
    print(f'并行数:   {n_jobs}')
    print(f'环境:     {envs}')
    print(f'算法:     {algos}')

    all_results = []

    for algo in algos:
        for env_id in envs:
            result = run_hpo_for_env(algo, env_id, n_trials, output_dir, gpus, n_jobs)
            all_results.append(result)

    # ---- 汇总保存 ----
    summary = {
        'hpo_config': {
            'total_steps': TOTAL_STEPS,
            'seeds': SEEDS,
            'n_trials': n_trials,
            'cost_limit': COST_LIMIT,
            'alpha': ALPHA,
            'gamma': GAMMA,
            'hidden_sizes': HIDDEN_SIZES,
            'search_params': [
                'learning_rate: categorical [1e-4, 3e-4, 1e-3]',
                'batch_size: categorical [64, 128, 256, 512]',
                'polyak: uniform [0.001, 0.02]',
                'cost_penalty_lr_scale: categorical [1, 10, 50]',
                'lagrangian_multiplier_init: categorical [0.3, 0.693147, 1.0]',
                # ---- WCSAC_IQN only ----
                'iqn_n_quantiles: categorical [8, 16, 32, 64]',
                'iqn_kappa: uniform [0.1, 2.0]',
                'cvar_quantile_samples: categorical [8, 16, 32, 64]',
                'iqn_embedding_dim: categorical [32, 64, 128]',
            ],
        },
        'results': [
            {
                'algo': r['algo'],
                'env_id': r['env_id'],
                'best_value': r['best_value'],
                'mean_reward': r['mean_reward'],
                'mean_cost': r['mean_cost'],
                'std_reward': r['std_reward'],
                'std_cost': r['std_cost'],
                'best_params': r['best_params'],
            }
            for r in all_results
        ],
    }

    summary_path = os.path.join(output_dir, 'hpo_summary.yaml')
    with open(summary_path, 'w', encoding='utf-8') as f:
        yaml.dump(summary, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f'\n{"=" * 60}')
    print(f'HPO 完成！共 {len(all_results)} 组搜索。')
    print(f'结果汇总: {summary_path}')
    print(f'{"=" * 60}')

    print(f'\n{"Algo":<16} {"Env":<30} {"Reward":>14} {"Cost":>14} {"Value":>12}')
    print('-' * 86)
    for r in all_results:
        print(
            f'{r["algo"]:<16} {r["env_id"]:<30} '
            f'{r["mean_reward"]:>8.2f} ± {r["std_reward"]:>4.2f}  '
            f'{r["mean_cost"]:>8.2f} ± {r["std_cost"]:>4.2f}  '
            f'{r["best_value"]:>12.2f}',
        )


if __name__ == '__main__':
    main()
