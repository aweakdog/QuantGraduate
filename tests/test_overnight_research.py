import json
from pathlib import Path

import pytest

from scripts.run_overnight_research import ENSEMBLE, PROFILES, SEEDS, compare, make_tasks, summarize


def test_corr_plan_has_complete_pairs_and_explicit_dependencies():
    tasks = make_tasks(Path('/tmp/research'), 'corr', '2026-09-10')
    names = {t['id'] for t in tasks}
    assert len(tasks) == len(names) == 210
    assert len({t['output'] for t in tasks}) == len(tasks)
    assert all(set(t['deps']) <= names for t in tasks)
    for seed in SEEDS:
        for model in PROFILES:
            for cap in ['C00', 'C80', 'C70', 'C60']:
                task = next(t for t in tasks if t['id'] == f'N1_C_{model}_{cap}_s{seed}')
                assert task['deps'] == [f'N1_C_{model}_TRAIN_s{seed}']
                assert '--load-preds' in task['command']
    for model in PROFILES:
        task = next(t for t in tasks if t['id'] == f'N1_C_{model}_ENSEMBLE')
        assert task['deps'] == [f'N1_C_{model}_TRAIN_s{s}' for s in ENSEMBLE]


def test_t2a_uses_same_augmented_matrix_for_baseline_and_experiment():
    tasks = make_tasks(Path('/tmp/research'), 't2a', '2026-09-10')
    assert len(tasks) == 20
    for task in tasks:
        command = task['command']
        assert command[command.index('--train-file') + 1] == 'training_data_pit_v24_tick1_t2a.parquet'
        assert command[command.index('--test-end') + 1] == '2026-09-10'
        assert '--load-preds' not in command
        assert task['deps'] == []


def test_training_and_replay_configs_match_except_execution_cap():
    tasks = make_tasks(Path('/tmp/research'), 'corr', '2026-09-10')
    base = next(t for t in tasks if t['id'] == 'N1_C_A_TRAIN_s42')['command']
    replay = next(t for t in tasks if t['id'] == 'N1_C_A_C70_s42')['command']
    for flag in ['--train-file', '--features-from', '--pit-universe', '--label', '--test-start', '--test-end', '--skip-boards', '--slippage', '--ind-cap', '--initial-capital']:
        assert base[base.index(flag) + 1] == replay[replay.index(flag) + 1]


def sample_result(total=20, drawdown=-10, event=-0.09):
    return {'summary': {'total_return_pct': total, 'max_dd_pct': drawdown, 'avg_deployed_pct': 90,
                        'avg_holdings': 3, 'n_trades': 100, 'total_cost_pct': 5},
            'daily': [{'date': date, 'daily_ret': value} for date, value in
                      [('2026-08-18', 0.02), ('2026-08-19', event), ('2026-08-20', 0.01)]]}


def test_event_decomposition_and_mismatched_dates():
    base, arm = sample_result(), sample_result(event=-0.06)
    difference = compare(arm, base)
    assert difference['event_0819_delta_pp'] == pytest.approx(3)
    assert difference['excluding_0819_delta_pp'] == pytest.approx(0)
    arm['daily'] = arm['daily'][:-1]
    with pytest.raises(ValueError, match='dates differ'):
        compare(arm, base)


def test_partial_summary_never_passes_twenty_seed_gate(tmp_path):
    processed = tmp_path / 'data/processed'
    processed.mkdir(parents=True)
    for i, seed in enumerate(SEEDS):
        suffix = f'_s{seed}_ts2023-09-19_te2026-09-10_cap100000.json'
        for arm, result in [('C00', sample_result()), ('C80', sample_result(21, -7))]:
            (processed / f'wf_daily_N1_C_A_{arm}{suffix}').write_text(json.dumps(result))
        out = summarize(tmp_path, 'corr', '2026-09-10')
        assert out['rows'][0]['n_pairs'] == i + 1
        assert out['rows'][0]['passes_corr_numeric_gate'] == (i == 19)
    assert (tmp_path / 'summary_corr.json').exists()


@pytest.mark.parametrize('phase', ['label', 'prune'])
def test_n2_plan_is_ten_seed_three_arm_and_never_changes_execution(phase):
    tasks = make_tasks(Path('/tmp/research'), phase, '2026-09-11')
    assert len(tasks) == len({t['id'] for t in tasks}) == 60
    assert len({t['output'] for t in tasks}) == 60
    for task in tasks:
        command = task['command']
        assert command[command.index('--train-file') + 1] == 'training_data_pit_v24_tick1.parquet'
        assert command[command.index('--exec-mode') + 1] == 't1close'
        assert command[command.index('--hold-days') + 1] == '5'
        assert command[command.index('--regime-filter') + 1] == 'breadth'
        assert '--save-preds' in command and '--load-preds' not in command
        if phase == 'prune':
            assert '--label-alignment' not in command
        else:
            mode = command[command.index('--label-alignment') + 1]
            assert mode in ['legacy', 'common', 't1close']


def test_n2_shared_start_is_used_in_commands_and_output_names():
    tasks = make_tasks(Path('/tmp/research'), 'label', '2026-09-11', test_start='2023-09-20')
    for task in tasks:
        command = task['command']
        assert command[command.index('--test-start') + 1] == '2023-09-20'
        assert '_ts2023-09-20_' in task['output']


@pytest.mark.parametrize('test_start', ['2023-09-19', '2023-09-20'])
def test_label_summary_separates_purge_cost_from_target_change(tmp_path, test_start):
    processed = tmp_path / 'data/processed'
    processed.mkdir(parents=True)
    for seed in SEEDS[:10]:
        for arm, total in [('CURRENT', 20), ('CONTROL', 18), ('ALIGNED', 25)]:
            path = processed / f'wf_daily_N2_L_A_{arm}_s{seed}_ts{test_start}_te2026-09-11_cap100000.json'
            path.write_text(json.dumps(sample_result(total=total)))
    rows = summarize(tmp_path, 'label', '2026-09-11', test_start)['rows']
    values = {r['arm']: r['median']['total_return_pct'] for r in rows}
    assert values == {'CONTROL-CURRENT': -2, 'ALIGNED-CONTROL': 7, 'ALIGNED-CURRENT': 5}
    assert all(r['n_pairs'] == 10 and not r.get('adoption_ready', False) for r in rows)


def test_n3_plan_only_adds_new_seeds_and_dissection_arms():
    """N3: FULL/NOEXO 只补后 10 个种子(文件名与 N2 同构); purge6/common5 只用前 10 个种子;
    执行层与 N2 一字不改, 不产生任何 load-preds 重放"""
    tasks = make_tasks(Path('/tmp/research'), 'n3', '2026-09-11', test_start='2023-09-20')
    assert len(tasks) == len({t['id'] for t in tasks}) == len({t['output'] for t in tasks}) == 80
    confirm = [t for t in tasks if t['id'].startswith('N2_F_')]
    dissect = [t for t in tasks if t['id'].startswith('N2_L_')]
    assert len(confirm) == 40 and len(dissect) == 40
    assert {int(t['id'].rsplit('_s', 1)[1]) for t in confirm} == set(SEEDS[10:20])
    assert {int(t['id'].rsplit('_s', 1)[1]) for t in dissect} == set(SEEDS[:10])
    for task in tasks:
        command = task['command']
        assert command[command.index('--test-start') + 1] == '2023-09-20' and '_ts2023-09-20_' in task['output']
        assert command[command.index('--exec-mode') + 1] == 't1close'
        assert command[command.index('--hold-days') + 1] == '5'
        assert command[command.index('--regime-breadth') + 1] == '0.40'
        assert '--save-preds' in command and '--load-preds' not in command
    modes = {command[command.index('--label-alignment') + 1] for command in (t['command'] for t in dissect)}
    assert modes == {'purge6', 'common5'}
    assert all('--label-alignment' not in t['command'] for t in confirm)
    noexo = next(t for t in confirm if '_NOEXO_' in t['id'])
    assert noexo['command'][noexo['command'].index('--features-from') + 1].startswith('features_N2_')
    # 与 N2 产出的文件名完全同构, 才能拼成 20 种子配对
    n2 = make_tasks(Path('/tmp/research'), 'prune', '2026-09-11', test_start='2023-09-20')
    n2_names = {Path(t['output']).name.replace(f'_s{s}_', '_sX_') for t in n2 for s in SEEDS[:10] if f'_s{s}_' in Path(t['output']).name}
    n3_names = {Path(t['output']).name.replace(f'_s{s}_', '_sX_') for t in confirm for s in SEEDS[10:20] if f'_s{s}_' in Path(t['output']).name}
    assert n3_names <= n2_names


def test_n4_plan_confirms_purge6_and_adds_dose_response_only():
    tasks = make_tasks(Path('/tmp/research'), 'n4', '2026-09-11', test_start='2023-09-20')
    assert len(tasks) == len({t['id'] for t in tasks}) == len({t['output'] for t in tasks}) == 60
    by_mode = {}
    for task in tasks:
        command = task['command']
        mode = command[command.index('--label-alignment') + 1]
        by_mode.setdefault(mode, set()).add(int(task['id'].rsplit('_s', 1)[1]))
        assert command[command.index('--test-start') + 1] == '2023-09-20' and '_ts2023-09-20_' in task['output']
        assert command[command.index('--exec-mode') + 1] == 't1close' and command[command.index('--hold-days') + 1] == '5'
        assert '--features-from' in command and command[command.index('--features-from') + 1].startswith('features_V24PUT_T1')
        assert '--save-preds' in command and '--load-preds' not in command
    assert by_mode == {'purge6': set(SEEDS[10:20]), 'purge7': set(SEEDS[:10]), 'purge8': set(SEEDS[:10])}


def test_n4_summary_uses_full_as_equivalent_baseline_for_new_seeds(tmp_path):
    processed = tmp_path / 'data/processed'
    processed.mkdir(parents=True)
    suffix = '_ts2023-09-20_te2026-09-11_cap100000.json'
    for seed in SEEDS[:10]:
        (processed / f'wf_daily_N2_L_A_CURRENT_s{seed}{suffix}').write_text(json.dumps(sample_result(20)))
        (processed / f'wf_daily_N2_L_A_PURGE6_s{seed}{suffix}').write_text(json.dumps(sample_result(30, -8)))
        (processed / f'wf_daily_N2_L_A_PURGE7_s{seed}{suffix}').write_text(json.dumps(sample_result(21)))
        (processed / f'wf_daily_N2_L_A_PURGE8_s{seed}{suffix}').write_text(json.dumps(sample_result(19)))
    out = summarize(tmp_path, 'n4', '2026-09-11', '2023-09-20')
    row = next(r for r in out['rows'] if r['arm'] == 'PURGE6-CURRENT')
    assert row['n_pairs'] == 10 and row['passes_gate'] is False and row['complete'] is False
    for seed in SEEDS[10:20]:                    # 新种子: 基线只有 N3 的 FULL(同 legacy 模型)
        (processed / f'wf_daily_N2_F_A_FULL_s{seed}{suffix}').write_text(json.dumps(sample_result(20)))
        (processed / f'wf_daily_N2_L_A_PURGE6_s{seed}{suffix}').write_text(json.dumps(sample_result(30, -8)))
    out = summarize(tmp_path, 'n4', '2026-09-11', '2023-09-20')
    row = next(r for r in out['rows'] if r['arm'] == 'PURGE6-CURRENT')
    assert row['n_pairs'] == 20 and row['passes_gate'] is True
    assert out['purge6_both_points_pass'] is None       # B 点没有结果, 联合判定不能成立
    values = {r['arm']: r['median']['total_return_pct'] for r in out['rows']}
    assert values['PURGE7-CURRENT'] == 1 and values['PURGE8-CURRENT'] == -1
    assert out['adoption_ready'] is False


def test_n3_summary_pools_twenty_seeds_and_never_passes_partial(tmp_path):
    processed = tmp_path / 'data/processed'
    processed.mkdir(parents=True)
    suffix = '_ts2023-09-20_te2026-09-11_cap100000.json'
    for seed in SEEDS[:10]:                       # N2 已有的基线与对照
        for arm, total in [('CURRENT', 20), ('CONTROL', 30), ('PURGE6', 22), ('COMMON5', 27)]:
            (processed / f'wf_daily_N2_L_A_{arm}_s{seed}{suffix}').write_text(json.dumps(sample_result(total=total)))
    for i, seed in enumerate(SEEDS):
        for arm, result in [('FULL', sample_result()), ('NOEXO', sample_result(28, -8))]:
            (processed / f'wf_daily_N2_F_A_{arm}_s{seed}{suffix}').write_text(json.dumps(result))
        out = summarize(tmp_path, 'n3', '2026-09-11', '2023-09-20')
        row = next(r for r in out['rows'] if r['arm'] == 'NOEXO-FULL')
        assert row['n_pairs'] == i + 1 and row['passes_gate'] == (i == 19)
        assert out['adoption_ready'] is False
    values = {r['arm']: r['median']['total_return_pct'] for r in out['rows']}
    assert values['PURGE6-CURRENT'] == 2 and values['COMMON5-CURRENT'] == 7 and values['CONTROL-CURRENT'] == 10
    assert out['noexo_both_points_pass'] is None          # B 点还没有结果, 不能宣布联合通过
    assert (tmp_path / 'summary_n3.json').exists()
