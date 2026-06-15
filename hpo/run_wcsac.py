#!/usr/bin/env python3
"""使用 ExperimentGrid 批量启动 WCSAC / WCSAC_IQN 在多个环境上训练。"""
import warnings

import torch

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train


if __name__ == '__main__':
    eg = ExperimentGrid(exp_name='WCSAC_Benchmark')

    # ---- 算法 ----
    eg.add('algo', ['WCSAC', 'WCSAC_IQN'])

    # ---- 环境 ----
    # SafetyPointGoal1-v0, SafetyPointCircle2-v0 在配置中有环境专属调优参数
    # 其余环境使用 defaults 默认参数
    eg.add('env_id', [
        'SafetyPointGoal1-v0',
        'SafetyPointCircle2-v0',
        'SafetyHopperVelocity-v1',
        'SafetyHumanoidVelocity-v1',
        'SafetyAntVelocity-v1',
        'SafetyPointButton1-v0',
    ])

    # ---- 随机种子 ----
    eg.add('seed', [0, 111, 222])

    # ---- 日志 ----
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('logger_cfgs:use_tensorboard', [True])

    # ---- 训练参数 ----
    eg.add('train_cfgs:torch_threads', [1])
    eg.add('train_cfgs:total_steps', [1_000_000])
    eg.add('algo_cfgs:steps_per_epoch', [10_000])
    eg.add('algo_cfgs:update_cycle', [100])
    eg.add('algo_cfgs:update_iters', [100])
    eg.add('algo_cfgs:start_learning_steps', [500])
    eg.add('algo_cfgs:cost_normalize', [False])
    eg.add('algo_cfgs:alpha', [0.693147])
    eg.add('algo_cfgs:cost_penalty_lr_scale', [50.0])
    eg.add('model_cfgs:weight_initialization_mode', ['xavier_uniform'])
    eg.add('model_cfgs:actor:hidden_sizes', [[256, 256]])
    eg.add('model_cfgs:actor:activation', ['relu'])
    eg.add('model_cfgs:actor:lr', [1e-3])
    eg.add('model_cfgs:critic:hidden_sizes', [[256, 256]])
    eg.add('model_cfgs:critic:activation', ['relu'])
    eg.add('model_cfgs:critic:lr', [1e-3])
    eg.add('lagrange_cfgs:lagrangian_multiplier_init', [0.693147])
    eg.add('lagrange_cfgs:cvar_alpha', [0.9])

    # ---- GPU 配置 ----
    avaliable_gpus = list(range(torch.cuda.device_count()))
    gpu_id = [0]  # 单卡; 多卡示例: [0, 1, 2, 3]; CPU: None

    if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
        warnings.warn('GPU 不可用，回退到 CPU。', stacklevel=1)
        gpu_id = None
    else:
        print(f'使用 GPU: {gpu_id}')

    # ---- 启动 ----
    # num_pool 控制并行进程数，建议 <= GPU 数量 × 每个 GPU 能跑的环境数
    eg.run(train, num_pool=4, gpu_id=gpu_id)
