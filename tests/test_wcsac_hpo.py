"""Tests for WCSAC HPO configuration generation."""

from hpo import (
    run_wcsac_gaussian_hpo,
    run_wcsac_hpo,
    run_wcsac_iqn_hpo,
    run_wcsac_metadrive_hpo,
)


class FakeTrial:
    """Small Optuna trial substitute for deterministic config tests."""

    def suggest_categorical(self, name, choices):
        del name
        return choices[0]

    def suggest_float(self, name, low, high, log=False):
        del name, high, log
        return low


def _assert_common_cfg(cfg) -> None:
    assert cfg['algo_cfgs']['cost_normalize'] is False
    assert cfg['algo_cfgs']['alpha'] == 0.693147
    assert cfg['algo_cfgs']['update_cycle'] == 100
    assert cfg['algo_cfgs']['update_iters'] == 100
    assert cfg['logger_cfgs']['save_model_freq'] == 50
    assert cfg['model_cfgs']['weight_initialization_mode'] == 'xavier_uniform'
    assert cfg['model_cfgs']['actor']['activation'] == 'relu'
    assert cfg['model_cfgs']['critic']['activation'] == 'relu'
    assert cfg['model_cfgs']['actor']['lr'] == cfg['model_cfgs']['critic']['lr']
    assert cfg['lagrange_cfgs']['cvar_alpha'] == 0.9


def test_general_hpo_builds_valid_iqn_config() -> None:
    params = run_wcsac_hpo.suggest_params(FakeTrial(), 'WCSAC_IQN')
    cfg = run_wcsac_hpo.make_custom_cfgs(
        'WCSAC_IQN',
        'SafetyPointGoal1-v0',
        params,
        '/tmp/wcsac-hpo-test',
        0,
    )
    _assert_common_cfg(cfg)
    assert cfg['algo_cfgs']['iqn_n_quantiles'] == 8
    assert cfg['model_cfgs']['critic']['iqn_embedding_dim'] == 32


def test_hpo_tensorboard_names_only_include_trial_and_seed() -> None:
    assert run_wcsac_hpo._build_trial_name(7) == 'trial_007'
    assert run_wcsac_metadrive_hpo._build_trial_name(12) == 'trial_012'


def test_hpo_parallel_seed_gpu_assignment_uses_disjoint_chunks() -> None:
    assert run_wcsac_hpo._select_seed_gpus(0, list(range(8)), 3, 2) == [0, 1, 2]
    assert run_wcsac_hpo._select_seed_gpus(1, list(range(8)), 3, 2) == [3, 4, 5]
    assert run_wcsac_hpo._select_seed_gpus(2, list(range(8)), 3, 2) == [0, 1, 2]
    assert run_wcsac_hpo._select_seed_gpus(3, [7], 1, 1) == [7, 7, 7]


def test_gaussian_and_iqn_use_separate_search_spaces() -> None:
    gaussian = run_wcsac_gaussian_hpo.suggest_params(FakeTrial())
    iqn = run_wcsac_iqn_hpo.suggest_params(FakeTrial())

    assert 'algo_cfgs:iqn_n_quantiles' not in gaussian
    assert 'model_cfgs:critic:iqn_embedding_dim' not in gaussian
    assert iqn['algo_cfgs:iqn_n_quantiles'] == 8
    assert iqn['model_cfgs:critic:iqn_embedding_dim'] == 32


def test_metadrive_hpo_builds_valid_iqn_config() -> None:
    params = run_wcsac_metadrive_hpo.suggest_params(FakeTrial(), 'WCSAC_IQN')
    cfg = run_wcsac_metadrive_hpo.make_custom_cfgs(
        'WCSAC_IQN',
        params,
        '/tmp/wcsac-metadrive-hpo-test',
        0,
    )
    _assert_common_cfg(cfg)
    assert cfg['algo_cfgs']['steps_per_epoch'] == 10_000
    assert cfg['env_cfgs']['meta_drive_config']['horizon'] == 1000
