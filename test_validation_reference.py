"""Dependency-free guards for the incumbent and declared win rate target."""
import ast
from contextlib import redirect_stderr
import io
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import validate_winrate


ROOT = Path(__file__).resolve().parent


def result(win_rate=25.53, net_return=60.23, profit_factor=2.50):
    return {
        'full_period': {
            'win_rate_pct': win_rate, 'closed_positions': 47,
            'profit_factor': profit_factor, 'return_pct': net_return,
            'excess_over_cash_pp': net_return - 20, 'max_drawdown_pct': -6.0,
        },
        'annual': {
            str(year): {'start': f'{year}-01-03', 'end': f'{year}-12-30',
                        'excess_over_cash_pp': 2.0}
            for year in range(2022, 2026)
        },
        'stress': {
            'higher_slippage': {'excess_over_cash_pp': 10.0, 'return_pct': 45.0},
            'exclude_vic_pdr': {'excess_over_cash_pp': 8.0, 'return_pct': 40.0},
        },
    }


class DeclaredWinRateTargetTests(unittest.TestCase):
    def test_one_percentage_point_is_not_five(self):
        report = validate_winrate.evaluate_candidate(result(26.53), result(), 5.0)
        self.assertEqual(report['win_rate_gain_pp'], 1.0)
        self.assertFalse(report['checks']['higher_win_rate'])
        self.assertFalse(report['historical_pass'])
        self.assertTrue(all(value for key, value in report['checks'].items()
                            if key != 'higher_win_rate'))

    def test_exact_five_percentage_point_boundary_passes(self):
        report = validate_winrate.evaluate_candidate(result(30.53), result(), 5.0)
        self.assertEqual(report['win_rate_gain_pp'], 5.0)
        self.assertEqual(report['min_win_rate_gain_pp'], 5.0)
        self.assertTrue(report['historical_pass'])

    def test_just_below_five_points_fails(self):
        report = validate_winrate.evaluate_candidate(result(30.5299), result(), 5.0)
        self.assertFalse(report['checks']['higher_win_rate'])

    def test_default_preserves_legacy_any_positive_gain(self):
        self.assertTrue(validate_winrate.evaluate_candidate(result(25.54), result())['historical_pass'])

    def test_zero_gain_fails_even_with_zero_minimum(self):
        self.assertFalse(validate_winrate.evaluate_candidate(result(), result(), 0)['checks']['higher_win_rate'])

    def test_unknown_win_rate_cannot_pass(self):
        report = validate_winrate.evaluate_candidate(result(None), result(), 5.0)
        self.assertIsNone(report['win_rate_gain_pp'])
        self.assertFalse(report['historical_pass'])

    def test_win_rate_target_cannot_offset_profit_loss(self):
        for field, value, key in (
            ('return_pct', 60.22, 'net_return_at_least_baseline_and_above_cash'),
            ('profit_factor', 2.49, 'profit_factor_at_least_baseline_and_above_one'),
            ('max_drawdown_pct', -6.01, 'drawdown_no_worse_than_baseline'),
            ('closed_positions', 35, 'at_least_30_closes_and_75pct_of_baseline'),
        ):
            candidate = result(35.0)
            candidate['full_period'][field] = value
            with self.subTest(field=field):
                report = validate_winrate.evaluate_candidate(candidate, result(), 5.0)
                self.assertFalse(report['checks'][key])
                self.assertFalse(report['historical_pass'])


class IncumbentReferenceTests(unittest.TestCase):
    def modes(self):
        return {'baseline': result(20.0, 32.24, 1.38),
                'balanced': result(), 'profit_guard': result(30.53)}

    def test_balanced_is_reference_and_every_other_mode_is_assessed(self):
        modes = validate_winrate.assess_modes(self.modes())
        self.assertNotIn('assessment', modes['balanced'])
        self.assertIn('assessment', modes['baseline'])
        self.assertEqual(modes['profit_guard']['assessment']['win_rate_gain_pp'], 5.0)
        for mode, summary in modes.items():
            with self.subTest(mode=mode):
                self.assertEqual(summary['reference_mode'], 'balanced')
                self.assertEqual(summary['min_win_rate_gain_pp'], 5.0)

    def test_beating_baseline_cannot_hide_losing_incumbent_profit(self):
        modes = self.modes()
        modes['profit_guard']['full_period']['return_pct'] = 50.0
        self.assertTrue(validate_winrate.evaluate_candidate(
            modes['profit_guard'], modes['baseline'], 5.0)['historical_pass'])
        validate_winrate.assess_modes(modes)
        assessment = modes['profit_guard']['assessment']
        self.assertFalse(assessment['checks']['net_return_at_least_baseline_and_above_cash'])
        self.assertFalse(assessment['historical_pass'])

    def test_beating_baseline_win_rate_is_insufficient_against_balanced(self):
        modes = self.modes()
        modes['profit_guard']['full_period']['win_rate_pct'] = 26.0
        self.assertTrue(validate_winrate.evaluate_candidate(
            modes['profit_guard'], modes['baseline'], 5.0)['historical_pass'])
        validate_winrate.assess_modes(modes)
        self.assertFalse(modes['profit_guard']['assessment']['checks']['higher_win_rate'])

    def test_explicit_reference_replaces_any_prior_assessment(self):
        modes = validate_winrate.assess_modes(self.modes())
        validate_winrate.assess_modes(modes, 'baseline', 0.0)
        self.assertNotIn('assessment', modes['baseline'])
        self.assertIn('assessment', modes['balanced'])
        self.assertEqual(modes['profit_guard']['assessment']['win_rate_gain_pp'], 10.53)

    def test_summary_lists_metrics_checks_and_failure_conclusion(self):
        modes = self.modes()
        modes['profit_guard']['full_period']['win_rate_pct'] = 26.53
        validate_winrate.assess_modes(modes)
        text = validate_winrate.comparison_summary(modes, 'balanced', 5.0)
        for mode in modes:
            self.assertIn(f'| {mode} |', text)
        self.assertIn('Conclusion: profit_guard historical criteria FAIL against balanced.', text)
        self.assertIn('No automatic promotion', text)
        self.assertIn('Net return >= reference', text)


class ValidationCliTests(unittest.TestCase):
    def test_defaults_declare_balanced_and_five_points(self):
        args = validate_winrate.build_parser().parse_args([])
        self.assertEqual(validate_winrate.MODES, ('baseline', 'balanced', 'profit_guard'))
        self.assertEqual(args.reference_mode, 'balanced')
        self.assertEqual(args.min_win_rate_gain_pp, 5.0)

    def test_invalid_targets_fail_before_dependency_probe_or_artifacts(self):
        for value in ('-1', 'nan', 'inf', '-inf', 'NaN', '1e999', 'not-a-number'):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), \
                    patch.object(validate_winrate.importlib.util, 'find_spec') as probe, \
                    patch.object(Path, 'mkdir') as mkdir:
                with self.assertRaises(SystemExit) as caught:
                    validate_winrate.main(['--min-win-rate-gain-pp=' + value])
                self.assertEqual(caught.exception.code, 2)
                probe.assert_not_called()
                mkdir.assert_not_called()

    def test_finite_nonnegative_targets_are_accepted(self):
        for value in ('0', '0.5', '5', '100'):
            with self.subTest(value=value):
                args = validate_winrate.build_parser().parse_args(['--min-win-rate-gain-pp', value])
                self.assertEqual(args.min_win_rate_gain_pp, float(value))
                self.assertTrue(math.isfinite(args.min_win_rate_gain_pp))

    def test_unknown_reference_fails_before_dependencies(self):
        with redirect_stderr(io.StringIO()), \
                patch.object(validate_winrate.importlib.util, 'find_spec') as probe:
            with self.assertRaises(SystemExit) as caught:
                validate_winrate.main(['--reference-mode=typo'])
        self.assertEqual(caught.exception.code, 2)
        probe.assert_not_called()


class ValidationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = (ROOT/'.github/workflows/validate_strategy.yml').read_text(encoding='utf-8')

    def test_workflow_is_manual_read_only_and_has_no_secrets(self):
        self.assertIn('name: Compare balanced and profit guard', self.workflow)
        self.assertIn('workflow_dispatch:', self.workflow)
        self.assertNotIn('schedule:', self.workflow)
        self.assertIn('contents: read', self.workflow)
        self.assertNotIn('contents: write', self.workflow)
        self.assertNotIn('secrets.', self.workflow)
        self.assertNotIn('git push', self.workflow)
        self.assertNotIn('train_bot.py', self.workflow)
        self.assertNotIn('TELEGRAM', self.workflow)

    def test_workflow_tests_before_dependencies_and_downloads_once(self):
        self.assertLess(self.workflow.index('python run_tests.py'),
                        self.workflow.index('pip install'))
        self.assertEqual(self.workflow.count('python download_all_data.py'), 1)
        self.assertLess(self.workflow.index('python download_all_data.py'),
                        self.workflow.index('python validate_winrate.py'))
        self.assertIn("python-version: '3.11'", self.workflow)

    def test_inputs_are_quoted_environment_values_and_artifacts_always_upload(self):
        self.assertIn("default: '2026-11-19'", self.workflow)
        self.assertIn("default: 'FLC,HVG,PVF,ROS'", self.workflow)
        self.assertIn('END_DATE: ${{ inputs.end_date }}', self.workflow)
        self.assertIn('ALLOW_MISSING: ${{ inputs.allow_missing }}', self.workflow)
        self.assertIn('--end-date "$END_DATE"', self.workflow)
        self.assertIn('--allow-missing "$ALLOW_MISSING"', self.workflow)
        self.assertIn('--reference-mode balanced', self.workflow)
        self.assertIn('--min-win-rate-gain-pp 5.0', self.workflow)
        self.assertIn('if: always()', self.workflow)
        self.assertIn('uses: actions/upload-artifact@v4', self.workflow)

    def test_comparison_download_declares_2021_warmup_and_quoted_inclusive_window(self):
        history_input = self.workflow.split('      history_start:\n', 1)[1].split(
            '      allow_missing:', 1)[0]
        self.assertIn("default: '2021-01-01'", history_input)
        self.assertIn('200 prior VNINDEX bars', history_input)
        download = self.workflow.split(
            '- name: Download one shared OHLCV snapshot', 1)[1].split('- name:', 1)[0]
        self.assertIn('HISTORY_START: ${{ inputs.history_start }}', download)
        self.assertIn('END_DATE: ${{ inputs.end_date }}', download)
        self.assertIn('python download_all_data.py --start-date "$HISTORY_START" '
                      '--end-date "$END_DATE"', download)
        self.assertNotIn('${{ inputs.', download.split('run:', 1)[1])

    def test_download_window_does_not_move_the_fixed_2022_replay_start(self):
        self.assertEqual(validate_winrate.BACKTEST_START, '2022-01-01')
        self.assertNotIn('--start-date', validate_winrate.build_parser().format_help())
        comparison = self.workflow.split(
            '- name: Compare all modes against balanced', 1)[1].split('- name:', 1)[0]
        self.assertNotIn('--start-date', comparison)
        self.assertNotIn('BACKTEST_START_DATE:', self.workflow)
        self.assertNotIn('HISTORY_START', comparison)
        rows = ([{'Date': f'2021-{month:02d}-{day:02d}'}
                 for month in range(1, 11) for day in range(1, 21)]
                + [{'Date': '2022-01-01'}, {'Date': '2022-12-30'}])
        validate_winrate.validate_market_coverage(rows)
        with self.assertRaises(validate_winrate.OHLCVValidationError) as caught:
            validate_winrate.validate_market_coverage(rows[1:])
        self.assertIn('200 market rows strictly before 2022-01-01', str(caught.exception))

    def test_training_history_keeps_2012_origin_and_frozen_model_artifacts(self):
        tree = ast.parse((ROOT / 'train_bot.py').read_text(encoding='utf-8'))
        base_masks = [node.value for node in ast.walk(tree)
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'base_train_mask'
                              for target in node.targets)]
        self.assertEqual(len(base_masks), 1)
        strings = {node.value for node in ast.walk(base_masks[0])
                   if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        self.assertTrue({'2012-01-01', '2022-01-01', 'Date', 'Future_Date'} <= strings)
        expanding_starts = [node.value.value for node in ast.walk(tree)
                            if isinstance(node, ast.Assign)
                            and any(isinstance(target, ast.Name)
                                    and target.id == 'start_train_date'
                                    for target in node.targets)
                            and isinstance(node.value, ast.Constant)]
        self.assertEqual(expanding_starts, ['2012-01-01'])
        for name in ['xgboost_ultimate.json', 'xgboost_live_latest.json'] + [
                f'xgboost_model_{year}.json' for year in range(2022, 2027)]:
            with self.subTest(model=name):
                self.assertTrue((ROOT / name).is_file())
        self.assertNotIn('train_bot.py', self.workflow)


if __name__ == '__main__':
    unittest.main()
