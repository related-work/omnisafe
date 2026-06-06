#!/usr/bin/env python3
"""
WCSAC on SafeMetaDrive 单次启动脚本。

用法:
    python hpo/run_wcsac_metadrive.py
    python hpo/run_wcsac_metadrive.py --seed 42
    python hpo/run_wcsac_metadrive.py --gpu 0
"""
import argparse

import omnisafe


ENV_ID = 'SafeMetaDrive'
ALGO = 'WCSAC'

# SafeMetaDrive 默认最优参数（WCSAC 配置中的值）
DEFAULT_CFGS = {
    'train_cfgs': {
        'device': 'cuda:0',
        'total_steps': 500_000,
        'torch_threads': 1,
    },
    'algo_cfgs': {
        'steps_per_epoch': 2_000,
        'update_cycle': 100,
        'update_iters': 200,
        'reward_normalize': True,
        'cost_normalize': True,
        'warmup_epochs': 10,
    },
    'model_cfgs': {
        'actor': {'hidden_sizes': [256, 256, 256], 'lr': 3e-4},
        'critic': {'hidden_sizes': [256, 256, 256], 'lr': 1e-4},
    },
    'logger_cfgs': {
        'use_wandb': False,
        'use_tensorboard': True,
        'log_dir': './runs',
        'save_model_freq': 50,
    },
    'lagrange_cfgs': {
        'cost_limit': 0.0,
        'lagrangian_multiplier_init': 0.01,
        'lambda_lr': 1e-4,
        'lambda_optimizer': 'Adam',
        'cvar_alpha': 0.9,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description='WCSAC on SafeMetaDrive')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--gpu', type=int, default=0, help='GPU 编号')
    parser.add_argument('--steps', type=int, default=500_000, help='总训练步数')
    args = parser.parse_args()

    custom_cfgs = DEFAULT_CFGS.copy()
    custom_cfgs['train_cfgs']['device'] = f'cuda:{args.gpu}'
    custom_cfgs['train_cfgs']['total_steps'] = args.steps
    custom_cfgs['seed'] = args.seed

    print(f'启动: {ALGO} on {ENV_ID}')
    print(f'  GPU: {args.gpu}  |  Seed: {args.seed}  |  Steps: {args.steps:,}')

    agent = omnisafe.Agent(ALGO, ENV_ID, custom_cfgs=custom_cfgs)
    reward, cost, ep_len = agent.learn()

    print(f'训练完成!  reward={reward:.2f}  cost={cost:.2f}  ep_len={ep_len:.2f}')


if __name__ == '__main__':
    main()
