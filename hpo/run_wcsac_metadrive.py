#!/usr/bin/env python3
"""
WCSAC / WCSAC_IQN on SafeMetaDrive 单次启动脚本。

用法:
    python hpo/run_wcsac_metadrive.py
    python hpo/run_wcsac_metadrive.py --algo WCSAC_IQN
    python hpo/run_wcsac_metadrive.py --seed 42 --gpu 0
"""
import argparse
import copy
import os

# Panda3D must be configured before importing OmniSafe/MetaDrive.
os.environ.setdefault('RENDER_OFFSCREEN', '1')


def _configure_panda3d() -> None:
    """Configure Panda3D for headless training before MetaDrive is imported."""
    try:
        from panda3d.core import loadPrcFileData

        loadPrcFileData('', 'window-type offscreen')
        loadPrcFileData('', 'audio-library-name null')
    except ImportError:
        pass


_configure_panda3d()

import omnisafe
import torch


ENV_ID = 'SafeMetaDrive'
ALGOS = ['WCSAC', 'WCSAC_IQN']

# SafeMetaDrive 默认参数
DEFAULT_CFGS = {
    'train_cfgs': {
        'device': 'cuda:0',
        'total_steps': 1_000_000,
        'torch_threads': 1,
    },
    'algo_cfgs': {
        'steps_per_epoch': 10_000,
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
        'actor': {'hidden_sizes': [256, 256], 'activation': 'relu', 'lr': 1e-3},
        'critic': {'hidden_sizes': [256, 256], 'activation': 'relu', 'lr': 1e-3},
    },
    'logger_cfgs': {
        'use_wandb': False,
        'use_tensorboard': True,
        'log_dir': './runs',
        'save_model_freq': 50,
    },
    'lagrange_cfgs': {
        'cost_limit': 1.0,
        'lagrangian_multiplier_init': 0.693147,
        'lambda_lr': 0.05,
        'lambda_optimizer': 'Adam',
        'cvar_alpha': 0.9,
    },
    'env_cfgs': {
        'meta_drive_config': {
            'horizon': 1000,
            'num_scenarios': 10,
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
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description='WCSAC/WCSAC_IQN on SafeMetaDrive')
    parser.add_argument('--algo', type=str, default='WCSAC', choices=ALGOS, help='算法')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--gpu', type=int, default=0, help='GPU 编号')
    parser.add_argument('--steps', type=int, default=500_000, help='总训练步数')
    args = parser.parse_args()

    custom_cfgs = copy.deepcopy(DEFAULT_CFGS)
    use_cuda = torch.cuda.is_available() and args.gpu < torch.cuda.device_count()
    custom_cfgs['train_cfgs']['device'] = f'cuda:{args.gpu}' if use_cuda else 'cpu'
    custom_cfgs['train_cfgs']['total_steps'] = args.steps
    custom_cfgs['seed'] = args.seed

    # WCSAC_IQN 需要额外的 IQN 参数
    if args.algo == 'WCSAC_IQN':
        custom_cfgs['algo_cfgs']['iqn_n_quantiles'] = 32
        custom_cfgs['algo_cfgs']['iqn_kappa'] = 1.0
        custom_cfgs['algo_cfgs']['cvar_quantile_samples'] = 32
        custom_cfgs['model_cfgs']['critic']['iqn_embedding_dim'] = 64

    print(f'启动: {args.algo} on {ENV_ID}')
    print(
        f'  Device: {custom_cfgs["train_cfgs"]["device"]}  '
        f'|  Seed: {args.seed}  |  Steps: {args.steps:,}',
    )

    agent = omnisafe.Agent(args.algo, ENV_ID, custom_cfgs=custom_cfgs)
    reward, cost, ep_len = agent.learn()

    print(f'训练完成!  reward={reward:.2f}  cost={cost:.2f}  ep_len={ep_len:.2f}')


if __name__ == '__main__':
    main()
