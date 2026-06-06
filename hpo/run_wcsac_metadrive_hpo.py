#!/usr/bin/env python3
"""
WCSAC on SafeMetaDrive 超参数自动搜索（Optuna）。

每个 trial 在 3 个随机种子 [111, 222, 333] 下各训练 50w 步，
取平均 reward/cost 作为目标值。约束：mean_cost ≤ 0.0（零碰撞容忍）。

用法:
    python hpo/run_wcsac_metadrive_hpo.py
    python hpo/run_wcsac_metadrive_hpo.py --trials 20
    python hpo/run_wcsac_metadrive_hpo.py --gpus 0,1,2,3 --parallel 4
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import yaml

import omnisafe


# ==============================================================================
# 配置
# ==============================================================================

ENV_ID = 'SafeMetaDrive'
ALGO = 'WCSAC'

# 每个 trial 的训练步数（MetaDrive 默认 50w）
TOTAL_STEPS = 500_000

# 每个 trial 的随机种子列表
SEEDS = [111, 222, 333]

# 默认 GPU 配置
AVAILABLE_GPUS = list(range(8))
N_JOBS = 8

# 每个组合的 trial 数
N_TRIALS = 30

# 结果保存目录
OUTPUT_DIR = './hpo_results'

# ---- 固定不变的参数 ----
COST_LIMIT = 0.0     # MetaDrive: 零碰撞容忍
ALPHA = 0.2          # SAC 默认温度系数
GAMMA = 0.99         # 折扣因子
HIDDEN_SIZES = [256, 256, 256]

# MetaDrive 环境固定参数（不参与搜索）
METADRIVE_CONFIG = {
    'horizon': 1000,
    'num_scenarios': 100,
    'accident_prob': 0.1,
    'traffic_density': 0.10,
    'crash_vehicle_cost': 1.0,
    'crash_object_cost': 1.0,
    'out_of_road_cost': 1.0,
    'start_seed': 1000,
    'image_observation': False,
    'vehicle_config': {
        'lidar': {'num_lasers': 240, 'distance': 50},
    },
}


# ==============================================================================
# 超参数搜索空间
# ==============================================================================

def _build_trial_name(trial_number: int, params: dict[str, Any]) -> str:
    """构建含参数值的 trial 目录名，方便 TensorBoard 查看。"""
    alr = params['model_cfgs:actor:lr']
    clr = params['model_cfgs:critic:lr']
    bs = params['algo_cfgs:batch_size']
    pk = params['algo_cfgs:polyak']
    llr = params['lagrange_cfgs:lambda_lr']
    mi = params['lagrange_cfgs:lagrangian_multiplier_init']
    return (
        f'trial_{trial_number:03d}_'
        f'alr{alr:.0e}_clr{clr:.0e}_'
        f'bs{bs}_pk{pk:.4f}_'
        f'llr{llr:.2e}_mi{mi}'
    )


def suggest_params(trial: 'optuna.Trial') -> dict[str, Any]:
    """为一个 trial 采样超参数（6 个搜索参数）。"""
    return {
        # Actor 学习率
        'model_cfgs:actor:lr': trial.suggest_categorical(
            'actor_lr', [1e-5, 1e-4, 3e-4, 1e-3],
        ),
        # Critic 学习率
        'model_cfgs:critic:lr': trial.suggest_categorical(
            'critic_lr', [1e-5, 1e-4, 3e-4, 1e-3],
        ),
        # 批量大小
        'algo_cfgs:batch_size': trial.suggest_categorical(
            'batch_size', [64, 128, 256, 512],
        ),
        # 目标网络软更新系数
        'algo_cfgs:polyak': trial.suggest_float('polyak', 0.001, 0.02),
        # Lagrange 乘子学习率
        'lagrange_cfgs:lambda_lr': trial.suggest_float(
            'lambda_lr', 3e-4, 3e-3, log=True,
        ),
        # Lagrange 乘子初始值
        'lagrange_cfgs:lagrangian_multiplier_init': trial.suggest_categorical(
            'lagrangian_multiplier_init', [0.01, 0.1, 0.5, 1.0],
        ),
    }


def make_custom_cfgs(
    params: dict[str, Any],
    log_dir: str,
    seed: int,
    gpu_id: int | None = None,
) -> dict[str, Any]:
    """构建 OmniSafe custom_cfgs 字典（MetaDrive 专用）。"""
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
            'steps_per_epoch': 2_000,
            'update_cycle': 100,
            'update_iters': 200,
            'reward_normalize': True,
            'cost_normalize': True,
            'warmup_epochs': 10,
        },
        'model_cfgs': {
            'actor': {'hidden_sizes': HIDDEN_SIZES},
            'critic': {'hidden_sizes': HIDDEN_SIZES},
        },
        'logger_cfgs': {
            'use_wandb': False,
            'use_tensorboard': True,
            'log_dir': log_dir,
            'save_model_freq': 10_000_000,
        },
        'lagrange_cfgs': {
            'cost_limit': COST_LIMIT,
            'lambda_optimizer': 'Adam',
            'cvar_alpha': 0.9,
        },
        'env_cfgs': {
            'meta_drive_config': METADRIVE_CONFIG,
        },
        'seed': seed,
    }

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
    params: dict[str, Any],
    log_dir: str,
    seed: int,
    gpu_id: int | None = None,
) -> tuple[float, float]:
    """单种子训练，返回 (reward, cost)。"""
    import torch
    custom_cfgs = make_custom_cfgs(params, log_dir, seed, gpu_id)

    agent = omnisafe.Agent(ALGO, ENV_ID, custom_cfgs=custom_cfgs)
    reward, cost, _ep_len = agent.learn()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(reward), float(cost)


def objective(
    trial: 'optuna.Trial',
    base_log_dir: str,
    gpus: list[int],
) -> float:
    """目标函数：3 种子取平均，cost 超标给惩罚。"""
    params = suggest_params(trial)
    gpu_id = gpus[trial.number % len(gpus)] if gpus else None
    trial_dir = _build_trial_name(trial.number, params)

    rewards = []
    costs = []

    for seed in SEEDS:
        log_dir = os.path.join(base_log_dir, trial_dir, f'seed_{seed:03d}')
        try:
            reward, cost = run_single_seed(params, log_dir, seed, gpu_id)
            rewards.append(reward)
            costs.append(cost)
        except Exception as exc:
            print(f'[Trial {trial.number}] seed={seed} 异常: {exc}')
            rewards.append(-1e6)
            costs.append(1e6)

    mean_reward = float(np.mean(rewards))
    mean_cost = float(np.mean(costs))
    std_reward = float(np.std(rewards))
    std_cost = float(np.std(costs))

    trial.set_user_attr('mean_reward', mean_reward)
    trial.set_user_attr('mean_cost', mean_cost)
    trial.set_user_attr('std_reward', std_reward)
    trial.set_user_attr('std_cost', std_cost)
    trial.set_user_attr('full_params', params)

    # MetaDrive: cost_limit=0.0，任何碰撞都是违规
    if mean_cost <= COST_LIMIT:
        value = mean_reward
    else:
        # cost 超标给惩罚（MetaDrive 惩罚更严厉）
        penalty = 2000.0 * (mean_cost - COST_LIMIT)
        value = mean_reward - penalty

    print(
        f'[Trial {trial.number:03d}] {ALGO} on {ENV_ID}\n'
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


def run_hpo(
    n_trials: int,
    output_dir: str,
    gpus: list[int],
    n_jobs: int,
) -> dict[str, Any]:
    """对 SafeMetaDrive 运行 HPO。"""
    import optuna

    study_name = f'{ALGO}_{ENV_ID}'
    storage_path = os.path.join(output_dir, f'{study_name}.db')

    print(f'\n{"=" * 60}')
    print(f'启动 HPO: {study_name}')
    print(f'Trials: {n_trials}  |  步数: {TOTAL_STEPS:,}  |  Seeds: {SEEDS}')
    print(f'GPU: {gpus}  |  并行: {n_jobs} jobs  |  cost_limit: {COST_LIMIT}')
    print(f'搜索参数: actor_lr, critic_lr, batch_size, polyak, lambda_lr, multiplier_init')
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
        lambda trial: objective(trial, base_log_dir, gpus),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    best = study.best_trial
    best_params = best.user_attrs.get('full_params', {})
    best_cfgs = make_custom_cfgs(best_params, log_dir='./runs', seed=0)

    result = {
        'algo': ALGO,
        'env_id': ENV_ID,
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
    parser = argparse.ArgumentParser(description='WCSAC HPO on SafeMetaDrive')
    parser.add_argument('--trials', type=int, default=N_TRIALS, help='Trial 数')
    parser.add_argument('--gpus', type=str, default=None, help='GPU 编号，逗号分隔')
    parser.add_argument('--parallel', type=int, default=None, help='并行数')
    parser.add_argument('--cpu', action='store_true', help='CPU 模式')
    args = parser.parse_args()

    if args.cpu:
        gpus = []
        n_jobs = 1
    else:
        gpus = [int(g.strip()) for g in args.gpus.split(',')] if args.gpus else AVAILABLE_GPUS
        n_jobs = args.parallel if args.parallel else N_JOBS

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'GPU 分配: {gpus if gpus else "CPU"}')
    print(f'并行数:   {n_jobs}')

    result = run_hpo(args.trials, OUTPUT_DIR, gpus, n_jobs)

    summary = {
        'hpo_config': {
            'algo': ALGO,
            'env_id': ENV_ID,
            'total_steps': TOTAL_STEPS,
            'seeds': SEEDS,
            'n_trials': args.trials,
            'cost_limit': COST_LIMIT,
            'alpha': ALPHA,
            'gamma': GAMMA,
            'hidden_sizes': HIDDEN_SIZES,
            'search_params': [
                'actor_lr: categorical [1e-5, 1e-4, 3e-4, 1e-3]',
                'critic_lr: categorical [1e-5, 1e-4, 3e-4, 1e-3]',
                'batch_size: categorical [64, 128, 256, 512]',
                'polyak: uniform [0.001, 0.02]',
                'lambda_lr: log-uniform [3e-4, 3e-3]',
                'lagrangian_multiplier_init: categorical [0.01, 0.1, 0.5, 1.0]',
            ],
        },
        'result': {
            'best_value': result['best_value'],
            'mean_reward': result['mean_reward'],
            'mean_cost': result['mean_cost'],
            'std_reward': result['std_reward'],
            'std_cost': result['std_cost'],
            'best_params': result['best_params'],
        },
    }

    summary_path = os.path.join(OUTPUT_DIR, f'{ALGO}_{ENV_ID}_summary.yaml')
    with open(summary_path, 'w', encoding='utf-8') as f:
        yaml.dump(summary, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f'\n结果汇总: {summary_path}')


if __name__ == '__main__':
    main()
