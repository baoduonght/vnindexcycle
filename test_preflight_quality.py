"""Strict preflight diagnostics must never turn bad bars into replay results."""
from contextlib import redirect_stderr
import csv
from datetime import date, timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import re
import types
import unittest
from unittest.mock import patch
import uuid

from ohlcv_quality import OHLCVValidationError
import validate_winrate


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == 'tests' else HERE
FIELDS = ('Date', 'Open', 'High', 'Low', 'Close', 'Volume')


def bars(start='2012-01-01', count=60):
    first = date.fromisoformat(start)
    return [dict(zip(FIELDS, [(first + timedelta(days=i)).isoformat(),
                              '100', '110', '90', '105', '1000']))
            for i in range(count)]


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path.read_bytes()


class PreflightQualityTests(unittest.TestCase):
    def setUp(self):
        # Default mkdir inherits ordinary ACLs. Python 3.14 TemporaryDirectory
        # applies owner-only Windows ACLs that can exclude a sandbox process.
        self.root = ROOT.parent/('preflight_quality_test_' + uuid.uuid4().hex)
        self.root.mkdir()
        self.source = self.root/'VNINDEX.csv'
        self.output = self.root/'validation'
        self.addCleanup(self.cleanup_fixture)

    def cleanup_fixture(self):
        for path in (self.source, self.output/'preflight_report.json'):
            path.unlink(missing_ok=True)
        try:
            self.output.rmdir()
        except FileNotFoundError:
            pass
        self.root.rmdir()

    def report(self):
        return json.loads((self.output/'preflight_report.json').read_text(encoding='utf-8'),
                          parse_constant=lambda value: self.fail(f'non-JSON constant: {value}'))

    def fail_preflight(self, rows, end_date='2026-11-19', symbol='VNINDEX'):
        original = write_csv(self.source, rows)
        with self.assertRaises(OHLCVValidationError):
            validate_winrate.validate_input_file(self.source, symbol, end_date, self.output)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(list(self.output.iterdir()), [self.output/'preflight_report.json'])
        report = self.report()
        self.assertEqual(report['phase'], 'preflight')
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(report['replay_started'])
        self.assertEqual(report['symbol'], symbol)
        self.assertEqual(report['source_file'], str(self.source.resolve()))
        self.assertEqual(report['sha256'], hashlib.sha256(original).hexdigest())
        return report

    def test_original_bad_date_reports_ohlc_values_and_exact_csv_row(self):
        rows = bars()
        rows[40]['High'] = '99'
        report = self.fail_preflight(rows)
        self.assertEqual(report['input_row_count'], 60)
        self.assertEqual(report['candidate_row_count'], 60)
        self.assertEqual(report['invalid_row_count'], 1)
        issue = report['issues'][0]
        self.assertEqual(issue['csv_row'], 42)
        self.assertEqual(issue['date'], '2012-02-10')
        self.assertIn('High is below max(Open, Close)', issue['reasons'])
        self.assertEqual(issue['raw_values']['High'], '99')
        self.assertEqual(issue['parsed_values']['High'], 99.0)

    def test_duplicate_date_retains_both_record_locations(self):
        rows = bars()
        rows[40]['Date'] = rows[39]['Date']
        issue = self.fail_preflight(rows)['issues'][0]
        self.assertEqual(issue['csv_row'], 42)
        self.assertEqual(issue['first_csv_row'], 41)
        self.assertIn('duplicate', ' '.join(issue['reasons']))

    def test_all_bad_rows_and_nonfinite_evidence_are_reported(self):
        rows = bars()
        rows[0]['Open'] = 'NaN'
        rows[1]['High'] = 'inf'
        rows[2]['Volume'] = '-1'
        rows[3]['Date'] = '2012-02-30'
        report = self.fail_preflight(rows)
        self.assertEqual(report['invalid_row_count'], 4)
        self.assertEqual(report['error_count'], 4)
        self.assertEqual(report['unparseable_date_count'], 1)
        self.assertEqual(report['issues'][0]['parsed_values']['Open'], 'NaN')
        reasons = ' '.join(reason for issue in report['issues'] for reason in issue['reasons'])
        for message in ('Open must be finite', 'High must be finite',
                        'Volume must be nonnegative', 'valid ISO date'):
            self.assertIn(message, reasons)

    def test_cutoff_does_not_hide_malformed_rows_or_renumber_evidence(self):
        rows = bars('2022-01-01')
        rows[-1]['High'] = '50'
        report = self.fail_preflight(rows, end_date='2022-02-21')
        self.assertEqual(report['candidate_row_count'], 52)
        self.assertEqual(report['issues'][0]['csv_row'], 61)

    def test_insufficient_selected_rows_is_a_structured_failure(self):
        report = self.fail_preflight(bars('2022-01-01'), end_date='2022-01-31', symbol='IPO')
        self.assertEqual(report['candidate_row_count'], 31)
        self.assertEqual(report['invalid_row_count'], 0)
        self.assertIn('fewer than 51 rows', ' '.join(report['issues'][0]['reasons']))

    def test_truncated_market_history_cannot_shorten_fixed_backtest(self):
        report = self.fail_preflight(bars('2022-01-01'))
        self.assertEqual(report['market_coverage']['warmup_rows'], 0)
        self.assertEqual(report['market_coverage']['in_period_rows'], 60)
        self.assertIn('200 market rows strictly before 2022-01-01', report['error'])

    def test_market_must_have_a_row_in_the_requested_backtest_period(self):
        report = self.fail_preflight(bars('2021-01-01', count=220))
        self.assertEqual(report['market_coverage']['warmup_rows'], 220)
        self.assertEqual(report['market_coverage']['in_period_rows'], 0)
        self.assertIn('at least one market row', report['error'])

    def test_market_warmup_boundary_requires_all_200_prior_rows(self):
        prior = bars('2021-01-01', count=200)
        current = bars('2022-01-01', count=1)
        validate_winrate.validate_market_coverage(prior + current)
        with self.assertRaises(OHLCVValidationError) as caught:
            validate_winrate.validate_market_coverage(prior[:-1] + current)
        self.assertIn('found 199', str(caught.exception))

    def test_valid_market_coverage_and_ipo_stock_create_no_failure_artifacts(self):
        for symbol, rows in (('VNINDEX', bars('2021-01-01', count=366)),
                             ('IPO', bars('2022-01-01', count=51))):
            with self.subTest(symbol=symbol):
                original = write_csv(self.source, rows)
                normalized = validate_winrate.validate_input_file(
                    self.source, symbol, '2022-03-31', self.output)
                self.assertEqual(len(normalized), len(rows))
                self.assertEqual(tuple(normalized[0]), FIELDS)
                self.assertIsInstance(normalized[0]['Open'], float)
                self.assertEqual(self.source.read_bytes(), original)
                self.assertFalse(self.output.exists())

    def test_decode_failure_still_keeps_source_hash_and_diagnostics(self):
        original = b'\xff\xfe\x00\x01broken CSV'
        self.source.write_bytes(original)
        with self.assertRaises(OHLCVValidationError):
            validate_winrate.validate_input_file(self.source, 'VNINDEX', '2026-11-19', self.output)
        report = self.report()
        self.assertEqual(report['sha256'], hashlib.sha256(original).hexdigest())
        self.assertIn('cannot read OHLCV CSV', report['error'])
        self.assertEqual(self.source.read_bytes(), original)

    def test_cli_exits_two_with_cause_report_and_no_replay_artifacts(self):
        rows = bars()
        rows[40]['High'] = '99'
        original = write_csv(self.source, rows)
        stderr = io.StringIO()
        # Simulate dependency presence only; strict validation and all I/O remain real.
        universe = types.ModuleType('vn30_universe')
        universe.ALL_HISTORICAL_VN30_TICKERS = []
        with patch.object(validate_winrate.importlib.util, 'find_spec', return_value=object()), \
                patch.dict('sys.modules', {'vn30_universe': universe}), \
                patch.dict(os.environ), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                validate_winrate.main(['--data-dir', str(self.root), '--output', str(self.output)])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn('2012-02-10', stderr.getvalue())
        self.assertIn('High is below max(Open, Close)', stderr.getvalue())
        self.assertIn(str(self.output/'preflight_report.json'), stderr.getvalue())
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(list(self.output.iterdir()), [self.output/'preflight_report.json'])

    def test_missing_dependencies_stop_before_any_output_creation(self):
        with patch.object(validate_winrate.importlib.util, 'find_spec', return_value=None), \
                patch.object(Path, 'mkdir') as mkdir, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                validate_winrate.main(['--data-dir', str(self.root), '--output', str(self.output)])
        self.assertEqual(caught.exception.code, 2)
        mkdir.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_cutoff_before_fixed_start_fails_before_dependency_probe(self):
        with patch.object(validate_winrate.importlib.util, 'find_spec') as probe, \
                redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                validate_winrate.main(['--end-date', '2021-12-31', '--output', str(self.output)])
        self.assertEqual(caught.exception.code, 2)
        probe.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_always_upload_captures_bad_input_when_download_or_preflight_fails(self):
        workflow = (ROOT/'.github/workflows/validate_strategy.yml').read_text(encoding='utf-8')
        step = workflow.split('      - name: Upload reports and frozen snapshot', 1)[1]
        self.assertIn('if: always()', step)
        self.assertRegex(step, re.compile(
            r'path: \|\s*\n\s+validation/\s*\n\s+data/\s*\n\s+data_quality_report\.json'))
        self.assertIn('if-no-files-found: warn', step)


if __name__ == '__main__':
    unittest.main()
