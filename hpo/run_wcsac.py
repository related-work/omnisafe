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
