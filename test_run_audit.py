"""Mode and provenance tests that require no market or package dependencies."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from execution_policy import get_policy
import run_audit


ROOT = Path(__file__).resolve().parents[1]


class EntryModeResolutionTests(unittest.TestCase):
    def test_default_is_balanced(self):
        self.assertEqual(run_audit.resolve_entry_mode({}), 'balanced')

    def test_all_four_explicit_modes_and_normalization(self):
        for mode in ('baseline', 'quality', 'balanced', 'index_trend'):
            with self.subTest(mode=mode):
                self.assertEqual(run_audit.resolve_entry_mode({'ENTRY_MODE': mode.upper()+' '}), mode)

    def test_invalid_or_empty_mode_does_not_fall_back(self):
        for value in ('balnced', '', ' '):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_audit.resolve_entry_mode({'ENTRY_MODE': value})

    def test_matching_expected_mode_is_accepted(self):
        self.assertEqual(run_audit.resolve_entry_mode({'ENTRY_MODE': ' Balanced ',
                         'RUN_EXPECTED_ENTRY_MODE': 'BALANCED'}), 'balanced')

    def test_wrong_expected_mode_fails_before_trading(self):
        with self.assertRaisesRegex(ValueError, 'ENTRY_MODE=baseline, expected balanced'):
            run_audit.resolve_entry_mode({'ENTRY_MODE': 'baseline', 'RUN_EXPECTED_ENTRY_MODE': 'balanced'})

    def test_unrelated_environment_values_are_not_returned(self):
        value = run_audit.resolve_entry_mode({'TELEGRAM_BOT_TOKEN':'fake-private-token',
                                              'DNSE_API_SECRET':'fake-private-key'})
        self.assertEqual(value, 'balanced')

    def test_policies_retain_distinct_modes_and_risk(self):
        self.assertFalse(get_policy('baseline').candle_filter)
        self.assertTrue(get_policy('quality').candle_filter)
        balanced = get_policy('balanced')
        self.assertTrue(balanced.candle_filter)
        self.assertEqual(balanced.risk_per_trade, 0.015)
        self.assertEqual(balanced.min_close_location, 0.60)
        self.assertEqual(balanced.max_open_gap_atr, 0.75)
        self.assertTrue(get_policy('index_trend').index_slopes)


class HashManifestTests(unittest.TestCase):
    def test_file_sha256_hashes_binary_content_in_chunks(self):
        content = b'first\x00\xff\r\n' * 200_000
        with patch.object(Path, 'open', return_value=io.BytesIO(content)):
            self.assertEqual(run_audit.sha256_file('unused'), hashlib.sha256(content).hexdigest())

    def test_frame_hash_uses_canonical_csv_with_dates_and_precision(self):
        class FakeFrame:
            def to_csv(self, **kwargs):
                self.kwargs = kwargs
                return 'Date,Close\n2026-11-19,12.5\n'
        frame = FakeFrame()
        expected = hashlib.sha256(frame.to_csv().encode('utf-8')).hexdigest()
        self.assertEqual(run_audit.frame_sha256(frame), expected)
        self.assertEqual(frame.kwargs, {'index':True, 'date_format':'%Y-%m-%d', 'float_format':'%.17g'})

    def test_manifest_records_used_models_frames_and_policy_without_environment(self):
        class FakeFrame:
            def to_csv(self, **kwargs):
                return 'Date,Close\n2026-11-19,12.5\n'
        config = {'backtest_start_date':'2022-01-01', 'stop_floor':-0.085}
        with patch.dict(os.environ, {'TELEGRAM_BOT_TOKEN':'fake-private-token',
                                     'DNSE_API_SECRET':'fake-private-key'}), \
             patch.object(run_audit, 'package_versions', return_value={'python':'test'}):
            result = run_audit.build_manifest(ROOT, 'balanced',
                    ['xgboost_model_2025.json', 'xgboost_model_2025.json'],
                    {'close':FakeFrame()}, config, ['VIC','BID'], ['ROS','FLC'])
        self.assertEqual(result['entry_mode'], 'balanced')
        self.assertEqual(result['entry_policy']['risk_per_trade'], .015)
        self.assertEqual(result['effective_config'], config)
        self.assertEqual(result['loaded_tickers'], ['BID','VIC'])
        self.assertEqual(result['missing_tickers'], ['FLC','ROS'])
        self.assertEqual(result['versions'], {'python':'test'})
        self.assertIn('xgboost_model_2025.json', result['sha256'])
        self.assertIn('run_audit.py', result['sha256'])
        self.assertEqual(result['sha256']['xgboost_model_2025.json'],
                         run_audit.sha256_file(ROOT/'xgboost_model_2025.json'))
        self.assertEqual(result['input_frames_sha256']['close'],run_audit.frame_sha256(FakeFrame()))
        rendered = json.dumps(result)
        self.assertNotIn('fake-private-token', rendered)
        self.assertNotIn('fake-private-key', rendered)
        self.assertNotIn('TELEGRAM_BOT_TOKEN', rendered)
        self.assertIn('not completed-trade win probability', result['ai_target'])

    def test_main_passes_allowlisted_config_not_whole_environment(self):
        tree = ast.parse((ROOT/'quant_bot.py').read_text(encoding='utf-8'))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'build_manifest']
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0].args[4], ast.Dict)
        keys = [key.value for key in calls[0].args[4].keys]
        self.assertTrue(keys)
        self.assertFalse(any(any(token in key.lower() for token in ('secret','token','password','chat_id'))
                             for key in keys))

    def test_manifest_writer_is_strict_utf8_json(self):
        with patch.object(Path, 'write_text') as write:
            run_audit.write_manifest('unused.json', {'mode':'balanced', 'description':'kiểm tra'})
        text = write.call_args.args[0]
        self.assertEqual(json.loads(text)['description'], 'kiểm tra')
        self.assertEqual(write.call_args.kwargs['encoding'], 'utf-8')
        with patch.object(Path, 'write_text') as write, self.assertRaises(ValueError):
            run_audit.write_manifest('unused.json', {'bad':float('nan')})
        write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
