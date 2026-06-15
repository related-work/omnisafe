#!/usr/bin/env python3
"""
WCSAC / WCSAC_IQN on SafeMetaDrive 超参数自动搜索（Optuna）。

每个 trial 在 3 个随机种子 [111, 222, 333] 下各训练 50w 步，
取平均 reward/cost 作为目标值。约束：mean_cost ≤ 0.0（零碰撞容忍）。

用法:
    python hpo/run_wcsac_metadrive_hpo.py
    python hpo/run_wcsac_metadrive_hpo.py --algo WCSAC_IQN
    python hpo/run_wcsac_metadrive_hpo.py --trials 20 --gpus 0,1,2,3
"""
from __future__ import annotations

import argparse
import copy
import os
from typing import Any

# MetaDrive/Panda3D must be configured before importing OmniSafe.
os.environ.setdefault('RENDER_OFFSCREEN', '1')


def _configure_panda3d() -> None:
    """Configure Panda3D before MetaDrive is imported."""
    try:
        from panda3d.core import loadPrcFileData

        loadPrcFileData('', 'window-type offscreen')
        loadPrcFileData('', 'audio-library-name null')
    except ImportError:
        pass


_configure_panda3d()

import numpy as np
import yaml

import omnisafe


# ==============================================================================
# 配置
# ==============================================================================

ENV_ID = 'SafeMetaDrive'
ALGOS = ['WCSAC', 'WCSAC_IQN']

# 每个 trial 的训练步数
TOTAL_STEPS = 1_000_000

# 每个 trial 的随机种子列表
SEEDS = [111, 222, 333]

# MetaDrive uses a process-global Panda3D ShowBase, so run one trial per process.
AVAILABLE_GPUS = list(range(8))
N_JOBS = 1

# 每个组合的 trial 数
N_TRIALS = 30

# 结果保存目录
OUTPUT_DIR = './hpo_results'

# ---- 固定不变的参数 ----
COST_LIMIT = 1.0     # 参考 on-policy 脚本
GAMMA = 0.99
ALPHA = 0.693147
HIDDEN_SIZES = [256, 256]

# MetaDrive 环境固定参数（参考 run_onpolicy_safemetadrive_zn.py）
METADRIVE_CONFIG = {
    'horizon': 1000,
    'num_scenarios': 10,        # 参考 on-policy
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

def _build_trial_name(trial_number: int) -> str:
    """构建简洁的 TensorBoard trial 目录名。"""
    return f'trial_{trial_number:03d}'


def suggest_params(trial: 'optuna.Trial', algo: str) -> dict[str, Any]:
    """Sample parameters while preserving the WCSAC optimizer relationships."""
    params = {
        'model_cfgs:actor:lr': trial.suggest_categorical(
            'learning_rate', [1e-4, 3e-4, 1e-3],
        ),
        'algo_cfgs:batch_size': trial.suggest_categorical(
            'batch_size', [64, 128, 256, 512],
        ),
        'algo_cfgs:polyak': trial.suggest_float('polyak', 0.001, 0.02),
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
            'auto_alpha': True,
            'steps_per_epoch': 10_000,
            'update_cycle': 100,
            'update_iters': 100,
            'start_learning_steps': 500,
            'reward_normalize': False,
            'cost_normalize': False,
            'policy_delay': 1,
            'warmup_epochs': 0,
            'max_ep_len': 1000,
        },
        'model_cfgs': {
            'weight_initialization_mode': 'xavier_uniform',
            'actor': {'hidden_sizes': HIDDEN_SIZES, 'activation': 'relu'},
            'critic': {'hidden_sizes': HIDDEN_SIZES, 'activation': 'relu'},
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
        'env_cfgs': {
            'meta_drive_config': copy.deepcopy(METADRIVE_CONFIG),
        },
        'seed': seed,
    }

    # 搜索参数写入
    for key, value in params.items():
        if key.startswith('algo_cfgs:'):
            custom_cfgs['algo_cfgs'][key.split(':')[1]] = value
        elif key.startswith('model_cfgs:'):
            section, field = key.split(':')[1], key.split(':')[2]
            custom_cfgs['model_cfgs'][section][field] = value
        elif key.startswith('lagrange_cfgs:'):
            custom_cfgs['lagrange_cfgs'][key.split(':')[1]] = value

    return custom_cfgs


def _read_metadrive_metrics(log_dir: str) -> dict[str, float]:
    """从训练日志 CSV 读取最后 epoch 的 MetaDrive 专用指标。"""
    import glob

    import pandas as pd

    metrics: dict[str, float] = {}
    try:
        pattern = os.path.join(log_dir, '**', 'progress.csv')
        csv_files = glob.glob(pattern, recursive=True)
        if not csv_files:
            return metrics
        df = pd.read_csv(csv_files[0])
        if len(df) < 2:
            return metrics
        # 取最后一行（有时最后一行是训练不完整的，取倒数第 2 行）
        row = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        for col in df.columns:
            col_lower = col.lower()
            if any(k in col_lower for k in ('success', 'crash', 'outofroad', 'arrive')):
                metrics[col] = float(row[col])
    except Exception:
        pass
    return metrics


def run_single_seed(
    algo: str,
    params: dict[str, Any],
    log_dir: str,
    seed: int,
    gpu_id: int | None = None,
) -> tuple[float, float, dict[str, float]]:
    """单种子训练，返回 (reward, cost, metadrive_metrics)。"""
    import torch
    custom_cfgs = make_custom_cfgs(algo, params, log_dir, seed, gpu_id)

    agent = omnisafe.Agent(algo, ENV_ID, custom_cfgs=custom_cfgs)
    reward, cost, _ep_len = agent.learn()

    # 读取 MetaDrive 专用指标
    md_metrics = _read_metadrive_metrics(log_dir)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(reward), float(cost), md_metrics


def objective(
    trial: 'optuna.Trial',
    algo: str,
    base_log_dir: str,
    gpus: list[int],
) -> float:
    """目标函数：3 种子取平均，cost 超标给惩罚。"""
    params = suggest_params(trial, algo)
    gpu_id = gpus[trial.number % len(gpus)] if gpus else None
    # TensorBoard 只用 trial 和 seed 区分运行，参数保留在 Optuna 中。
    trial_dir = _build_trial_name(trial.number)

    rewards = []
    costs = []

    all_md_metrics: list[dict[str, float]] = []

    for seed in SEEDS:
        log_dir = os.path.join(base_log_dir, trial_dir, f'seed_{seed:03d}')
        try:
            reward, cost, md_metrics = run_single_seed(algo, params, log_dir, seed, gpu_id)
            rewards.append(reward)
            costs.append(cost)
            all_md_metrics.append(md_metrics)
        except Exception as exc:
            print(f'[Trial {trial.number}] seed={seed} 异常: {exc}')
            rewards.append(-1e6)
            costs.append(1e6)
            all_md_metrics.append({})

    mean_reward = float(np.mean(rewards))
    mean_cost = float(np.mean(costs))
    std_reward = float(np.std(rewards))
    std_cost = float(np.std(costs))

    trial.set_user_attr('mean_reward', mean_reward)
    trial.set_user_attr('mean_cost', mean_cost)
    trial.set_user_attr('std_reward', std_reward)
    trial.set_user_attr('std_cost', std_cost)
    trial.set_user_attr('full_params', params)
    # 提取 MetaDrive 指标平均值
    if all_md_metrics and all_md_metrics[0]:
        for key in all_md_metrics[0]:
            vals = [m.get(key, float('nan')) for m in all_md_metrics]
            trial.set_user_attr(f'md_{key.replace("/", "_")}', float(np.nanmean(vals)))

    if mean_cost <= COST_LIMIT:
        value = mean_reward
    else:
        penalty = 2000.0 * (mean_cost - COST_LIMIT)
        value = mean_reward - penalty

    # 构建 MetaDrive 指标摘要
    md_summary = ''
    if all_md_metrics and all_md_metrics[0]:
        md_summary = '\n  MetaDrive: '
        md_summary += ' | '.join(
            f'{key.split("/")[-1]}='
            f'{np.nanmean([m.get(key, float("nan")) for m in all_md_metrics]):.3f}'
            for key in sorted(all_md_metrics[0].keys())
        )

    print(
        f'[Trial {trial.number:03d}] {algo} on {ENV_ID}\n'
        f'  params: actor_lr={params["model_cfgs:actor:lr"]:.1e}  '
        f'critic_lr={params["model_cfgs:critic:lr"]:.1e}  '
        f'batch={params["algo_cfgs:batch_size"]}  '
        f'polyak={params["algo_cfgs:polyak"]:.4f}\n'
        f'           lambda_lr={params["lagrange_cfgs:lambda_lr"]:.2e}  '
        f'mult_init={params["lagrange_cfgs:lagrangian_multiplier_init"]}\n'
        f'  reward: {rewards}  ->  mean={mean_reward:.2f} ± {std_reward:.2f}\n'
        f'  cost:   {[f"{c:.2f}" for c in costs]}  ->  mean={mean_cost:.2f} ± {std_cost:.2f}'
        f'{md_summary}\n'
        f'  value={value:.2f}',
    )
    return value


def run_hpo(
    algo: str,
    n_trials: int,
    output_dir: str,
    gpus: list[int],
    n_jobs: int,
) -> dict[str, Any]:
    """对 SafeMetaDrive 运行 HPO。"""
    import optuna

    study_name = f'{algo}_{ENV_ID}'
    storage_path = os.path.join(output_dir, f'{study_name}.db')

    print(f'\n{"=" * 60}')
    print(f'启动 HPO: {study_name}')
    print(f'Trials: {n_trials}  |  步数: {TOTAL_STEPS:,}  |  Seeds: {SEEDS}')
    print(f'GPU: {gpus}  |  并行: {n_jobs} jobs  |  cost_limit: {COST_LIMIT}')
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
        lambda trial: objective(trial, algo, base_log_dir, gpus),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    best = study.best_trial
    best_params = best.user_attrs.get('full_params', {})
    best_cfgs = make_custom_cfgs(algo, best_params, log_dir='./runs', seed=0)

    result = {
        'algo': algo,
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
    parser = argparse.ArgumentParser(description='WCSAC/WCSAC_IQN HPO on SafeMetaDrive')
    parser.add_argument(
        '--algo',
        type=str,
        default=None,
        help='算法，逗号分隔（默认 WCSAC,WCSAC_IQN）',
    )
    parser.add_argument('--trials', type=int, default=N_TRIALS, help='Trial 数')
    parser.add_argument('--gpus', type=str, default=None, help='GPU 编号，逗号分隔')
    parser.add_argument('--parallel', type=int, default=None, help='并行数')
    parser.add_argument('--cpu', action='store_true', help='CPU 模式')
    args = parser.parse_args()

    algos = [a.strip() for a in args.algo.split(',')] if args.algo else ALGOS

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
        n_jobs = args.parallel if args.parallel else N_JOBS
        if not gpus:
            print('未检测到可用 GPU，自动回退到 CPU。')
        if n_jobs != 1:
            print('MetaDrive 使用进程级 Panda3D 状态，并行 trial 已强制设为 1。')
            n_jobs = 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'GPU 分配: {gpus if gpus else "CPU"}')
    print(f'并行数:   {n_jobs}')
    print(f'算法:     {algos}')

    all_results = []

    for algo in algos:
        result = run_hpo(algo, args.trials, OUTPUT_DIR, gpus, n_jobs)
        all_results.append(result)

    # ---- 汇总保存 ----
    summary = {
        'hpo_config': {
            'env_id': ENV_ID,
            'total_steps': TOTAL_STEPS,
            'seeds': SEEDS,
            'n_trials': args.trials,
            'cost_limit': COST_LIMIT,
            'gamma': GAMMA,
            'hidden_sizes': HIDDEN_SIZES,
            'num_scenarios': METADRIVE_CONFIG['num_scenarios'],
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

    summary_path = os.path.join(OUTPUT_DIR, 'SafeMetaDrive_hpo_summary.yaml')
    with open(summary_path, 'w', encoding='utf-8') as f:
        yaml.dump(summary, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f'\n{"=" * 60}')
    print(f'HPO 完成！共 {len(all_results)} 组搜索。')
    print(f'结果汇总: {summary_path}')
    print(f'{"=" * 60}')

    print(f'\n{"Algo":<16} {"Env":<20} {"Reward":>14} {"Cost":>14} {"Value":>12}')
    print('-' * 76)
    for r in all_results:
        print(
            f'{r["algo"]:<16} {r["env_id"]:<20} '
            f'{r["mean_reward"]:>8.2f} ± {r["std_reward"]:>4.2f}  '
            f'{r["mean_cost"]:>8.2f} ± {r["std_cost"]:>4.2f}  '
            f'{r["best_value"]:>12.2f}',
        )


if __name__ == '__main__':
    main()
