"""Dependency-free tests of the downloader's actual validation wiring.

Only selected function bodies are compiled from the downloader AST; importing it
would require network SDKs and create a data directory. All bars below, including
the deliberately bad 2012-02-10 bar, are invented fixtures, not vendor evidence.
"""

import ast
import copy
import csv
from datetime import date, datetime, timedelta
import io
from pathlib import Path
import re
import unittest

from ohlcv_quality import (
    OHLCVValidationError, encode_csv, select_valid_source, validate_rows,
)


ROOT = Path(__file__).resolve().parent
DOWNLOADER = ROOT / 'download_all_data.py'
TREE = ast.parse(DOWNLOADER.read_text(encoding='utf-8'), filename=str(DOWNLOADER))


def function_named(name):
    return next(node for node in TREE.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def calls_named(node, name):
    return [child for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name) and child.func.id == name]


def compile_function(name, namespace):
    module = ast.Module(body=[function_named(name)], type_ignores=[])
    exec(compile(module, str(DOWNLOADER), 'exec'), namespace)
    return namespace[name]


def bars(count=60):
    first = date(2012, 1, 1)
    return [{'Date': (first + timedelta(days=i)).isoformat(),
             'Open': '100', 'High': '110', 'Low': '90',
             'Close': '105', 'Volume': '1000'}
            for i in range(count)]


class FrameDouble:
    """The small DataFrame surface used by the wrapper, with visible mutations."""

    def __init__(self, rows):
        self.rows = copy.deepcopy(rows)
        self.requested_orientations = []

    def to_dict(self, orient):
        if orient != 'records':
            raise AssertionError(f'unexpected DataFrame orientation: {orient}')
        self.requested_orientations.append(orient)
        # Expose the original mappings so in-place repair cannot escape this test.
        return self.rows

    def __getitem__(self, key):
        return [row[key] for row in self.rows]

    def __setitem__(self, key, values):
        values = list(values)
        if len(values) != len(self.rows):
            raise AssertionError('column assignment changed the number of rows')
        for row, value in zip(self.rows, values):
            row[key] = value


class PandasDouble:
    def __init__(self):
        self.constructed = []
        self.date_inputs = []

    def DataFrame(self, rows):
        frame = FrameDouble(rows)
        self.constructed.append(frame)
        return frame

    def to_datetime(self, values):
        self.date_inputs.append(list(values))
        return [datetime.fromisoformat(value) for value in values]


class WrapperHarness:
    def __init__(self, candidates):
        self.pd = PandasDouble()
        self.visited = []
        self.rejected = []
        self.writes = []
        self.report = {'symbols': {}}
        self.report_path = Path('fixture-quality-report.json')

        def providers(symbol):
            self.symbol = symbol
            for source, frame in candidates:
                self.visited.append(source)
                yield frame, source

        def symbol_report(symbol):
            return self.report['symbols'].setdefault(symbol, {'attempts': []})

        def record_rejection(symbol, source, rows, error):
            self.rejected.append((symbol, source, rows, error))
            symbol_report(symbol)['attempts'].append({'source': source,
                                                      'status': 'rejected'})

        def record_write(path, payload):
            self.writes.append((path, copy.deepcopy(payload)))

        self.namespace = {
            'pd': self.pd, 'select_valid_source': select_valid_source,
            '_iter_ticker_sources': providers, '_symbol_report': symbol_report,
            '_record_rejected_source': record_rejection,
            'write_json_atomic': record_write,
            'QUALITY_REPORT_PATH': self.report_path,
            'DATA_QUALITY_REPORT': self.report,
        }
        self.fetch = compile_function('fetch_ticker_multi_source', self.namespace)


class DownloadWrapperTests(unittest.TestCase):
    def test_invalid_primary_falls_back_through_actual_wrapper_and_stops(self):
        bad_rows = bars()
        bad_rows[40]['High'] = '99'
        primary = FrameDouble(bad_rows)
        secondary = FrameDouble(bars())

        def candidates():
            yield 'bad primary', primary
            yield 'good secondary', secondary
            self.fail('wrapper requested another provider after a valid dataset')

        harness = WrapperHarness(candidates())
        frame, source = harness.fetch('SYNTH')
        self.assertEqual(source, 'good secondary')
        self.assertEqual(harness.visited, ['bad primary', 'good secondary'])
        self.assertEqual(len(frame.rows), 60)
        self.assertEqual(primary.requested_orientations, ['records'])
        self.assertEqual(secondary.requested_orientations, ['records'])
        self.assertEqual([entry[1] for entry in harness.rejected], ['bad primary'])
        self.assertEqual(len(harness.pd.constructed), 1)

    def test_rejected_payload_and_row_evidence_are_not_repaired_or_mutated(self):
        raw = bars()
        raw[40].update(High='99', Note='invented invalid fixture')
        before = copy.deepcopy(raw)
        primary = FrameDouble(raw)
        harness = WrapperHarness([('primary', primary),
                                  ('secondary', FrameDouble(bars()))])
        harness.fetch('SYNTH')
        symbol, source, rejected, error = harness.rejected[0]
        self.assertEqual((symbol, source), ('SYNTH', 'primary'))
        self.assertEqual(primary.rows, before)
        self.assertEqual(rejected, before)
        self.assertIsInstance(error, OHLCVValidationError)
        issue = error.issues[0]
        self.assertEqual(issue['date'], '2012-02-10')
        self.assertEqual(issue['csv_row'], 42)
        self.assertEqual(issue['raw_values'], before[40])
        self.assertIn('High is below max(Open, Close)', issue['reasons'])

    def test_all_invalid_providers_return_none_and_persist_failed_status(self):
        high_bad, volume_bad = bars(), bars()
        high_bad[0]['High'] = '99'
        volume_bad[1]['Volume'] = '-1'
        harness = WrapperHarness([('high-bad', FrameDouble(high_bad)),
                                  ('volume-bad', FrameDouble(volume_bad))])
        self.assertEqual(harness.fetch('SYNTH'), (None, None))
        self.assertEqual([entry[1] for entry in harness.rejected],
                         ['high-bad', 'volume-bad'])
        self.assertEqual(harness.report['symbols']['SYNTH']['status'],
                         'failed_no_valid_source')
        self.assertEqual(harness.pd.constructed, [])
        path, persisted = harness.writes[-1]
        self.assertEqual(path, harness.report_path)
        self.assertEqual(persisted, harness.report)
        self.assertNotIn('source', persisted['symbols']['SYNTH'])

    def test_no_provider_data_is_a_reported_failure(self):
        harness = WrapperHarness([])
        self.assertEqual(harness.fetch('SYNTH'), (None, None))
        self.assertEqual(harness.rejected, [])
        self.assertEqual(harness.pd.constructed, [])
        self.assertEqual(harness.writes[-1][1]['symbols']['SYNTH']['status'],
                         'failed_no_valid_source')

    def test_short_valid_provider_cannot_bypass_minimum_or_merge_with_next(self):
        harness = WrapperHarness([('short', FrameDouble(bars(50))),
                                  ('complete', FrameDouble(bars(60)))])
        frame, source = harness.fetch('SYNTH')
        self.assertEqual(source, 'complete')
        self.assertEqual(len(frame.rows), 60)
        issue = harness.rejected[0][3].issues[0]
        self.assertEqual(issue['raw_values'], {'row_count': 50, 'min_rows': 51})

    def test_accepted_provider_returns_datetime_frame_source_and_saved_metadata(self):
        raw = bars(51)
        raw[0]['Volume'] = '0'
        raw[0]['ProviderNote'] = 'raw metadata must remain untouched'
        original = FrameDouble(raw)
        harness = WrapperHarness([('accepted provider', original)])
        frame, source = harness.fetch('SYNTH')
        self.assertEqual(source, 'accepted provider')
        self.assertEqual(original.rows, raw)
        self.assertEqual(frame.rows[0]['Date'], datetime(2012, 1, 1))
        self.assertEqual(frame.rows[-1]['Date'], datetime(2012, 2, 20))
        self.assertEqual(frame.rows[0]['Volume'], 0.0)
        self.assertIsInstance(frame.rows[0]['Open'], float)
        self.assertNotIn('ProviderNote', frame.rows[0])
        self.assertEqual(harness.rejected, [])
        metadata = harness.writes[-1][1]['symbols']['SYNTH']
        self.assertEqual(metadata, {'attempts': [], 'status': 'validated',
                                    'source': 'accepted provider', 'rows': 51})
        self.assertEqual(harness.pd.date_inputs,
                         [[row['Date'] for row in raw]])

    def test_missing_volume_rejects_entire_provider_instead_of_defaulting_to_zero(self):
        raw = bars()
        del raw[40]['Volume']
        primary = FrameDouble(raw)
        harness = WrapperHarness([('missing volume', primary),
                                  ('complete', FrameDouble(bars()))])
        _, source = harness.fetch('SYNTH')
        self.assertEqual(source, 'complete')
        issue = harness.rejected[0][3].issues[0]
        self.assertIn('missing fields: Volume', issue['reasons'])
        self.assertNotIn('Volume', primary.rows[40])
        self.assertNotIn('Volume', issue['raw_values'])

    def test_conflicting_duplicate_rejects_provider_and_keeps_both_locations(self):
        raw = bars()
        raw[40].update(Date=raw[39]['Date'], Close='104')
        harness = WrapperHarness([('duplicate', FrameDouble(raw)),
                                  ('complete', FrameDouble(bars()))])
        frame, source = harness.fetch('SYNTH')
        self.assertEqual(source, 'complete')
        self.assertEqual(len(frame.rows), 60)
        issue = harness.rejected[0][3].issues[0]
        self.assertEqual((issue['csv_row'], issue['first_csv_row']), (42, 41))
        self.assertEqual(issue['raw_values']['Close'], '104')
        self.assertIn('duplicate Date', ' '.join(issue['reasons']))

    def test_diagnostic_failure_propagates_instead_of_silently_accepting_fallback(self):
        raw = bars()
        raw[0]['High'] = '99'
        harness = WrapperHarness([('bad', FrameDouble(raw)),
                                  ('good', FrameDouble(bars()))])

        def unwritable(*_args):
            raise OSError('cannot preserve rejected source')

        harness.namespace['_record_rejected_source'] = unwritable
        with self.assertRaisesRegex(OSError, 'cannot preserve rejected source'):
            harness.fetch('SYNTH')
        self.assertEqual(harness.visited, ['bad'])
        self.assertEqual(harness.pd.constructed, [])
        self.assertEqual(harness.writes, [])


class DownloadIntegrationContractTests(unittest.TestCase):
    def test_every_provider_branch_yields_without_early_dataset_returns(self):
        iterator = function_named('_iter_ticker_sources')
        yields = [node for node in ast.walk(iterator) if isinstance(node, ast.Yield)]
        self.assertEqual(len(yields), 5)
        self.assertTrue(all(isinstance(node.value, ast.Tuple)
                            and len(node.value.elts) == 2 for node in yields))
        returns = [node for node in ast.walk(iterator)
                   if isinstance(node, ast.Return) and node.value is not None]
        self.assertEqual(returns, [], 'provider datasets must reach validation before selection')

    def test_provider_adapters_do_not_choose_a_conflicting_duplicate_or_round_prices(self):
        iterator = function_named('_iter_ticker_sources')
        destructive = [node.func.attr for node in ast.walk(iterator)
                       if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute)
                       and node.func.attr in {'drop_duplicates', 'round', 'clip'}]
        self.assertEqual(destructive, [])

    def test_provider_adapters_do_not_manufacture_missing_volume(self):
        iterator = function_named('_iter_ticker_sources')
        volume_names = {'v', 'volume', 'Volume'}
        fabricated = []
        for node in ast.walk(iterator):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and node.value.value == 0:
                for target in node.targets:
                    if isinstance(target, ast.Subscript) \
                            and isinstance(target.slice, ast.Constant) \
                            and target.slice.value in volume_names:
                        fabricated.append(node.lineno)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in {'get', 'setdefault'} and len(node.args) >= 2:
                key, default = node.args[:2]
                if isinstance(key, ast.Constant) and key.value in volume_names \
                        and isinstance(default, ast.Constant) and default.value == 0:
                    fabricated.append(node.lineno)
        self.assertEqual(fabricated, [])

    def save_pipeline(self):
        """Compile the actual per-symbol validation-to-write statements in main."""
        main = function_named('main')
        for branch in ast.walk(main):
            if not isinstance(branch, ast.If):
                continue
            for end, statement in enumerate(branch.body):
                if not isinstance(statement, ast.Expr) \
                        or not calls_named(statement, 'write_text_atomic'):
                    continue
                call = calls_named(statement, 'write_text_atomic')[0]
                if not (call.args and isinstance(call.args[0], ast.Name)
                        and call.args[0].id == 'file_path'):
                    continue
                start = next(index for index, node in enumerate(branch.body[:end])
                             if isinstance(node, ast.Assign)
                             and calls_named(node, 'validate_rows'))
                module = ast.Module(body=branch.body[start:end + 1], type_ignores=[])
                return compile(module, str(DOWNLOADER), 'exec')
        self.fail('could not locate the per-symbol CSV validation and atomic write path')

    def run_save_pipeline(self, raw, encoder=encode_csv):
        writes = []
        namespace = {
            'df': FrameDouble(raw), 'sym': 'SYNTH', 'file_path': 'fixture/SYNTH.csv',
            'validate_rows': validate_rows, 'encode_csv': encoder,
            'csv': csv, 'io': io,
            'write_text_atomic': lambda path, text: writes.append((path, text)),
        }
        exec(self.save_pipeline(), namespace)
        return writes, namespace

    def test_exact_serialized_prices_revalidate_and_reach_the_atomic_writer(self):
        raw = bars()
        raw[40].update(Open=100.00000000000003, High=110.12345678901235,
                       Low=90.00000000000001, Close=105.12345678901235,
                       Volume=123456.123456789)
        before = copy.deepcopy(raw)
        writes, namespace = self.run_save_pipeline(raw)
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][0], 'fixture/SYNTH.csv')
        text = writes[0][1]
        self.assertIn('100.00000000000003', text)
        restored = validate_rows(csv.DictReader(io.StringIO(text)), 'SYNTH', min_rows=51)
        self.assertEqual(restored, validate_rows(raw, 'SYNTH', min_rows=51))
        self.assertEqual(namespace['df'].rows, before)
        self.assertEqual(raw, before)

    def test_bad_serialization_is_rejected_before_any_atomic_dataset_write(self):
        writes = []

        def corrupt_encoder(rows):
            corrupted = copy.deepcopy(rows)
            corrupted[40]['High'] = 99.0
            return encode_csv(corrupted)

        namespace = {
            'df': FrameDouble(bars()), 'sym': 'SYNTH',
            'file_path': 'fixture/SYNTH.csv', 'validate_rows': validate_rows,
            'encode_csv': corrupt_encoder, 'csv': csv, 'io': io,
            'write_text_atomic': lambda *args: writes.append(args),
        }
        with self.assertRaises(OHLCVValidationError) as caught:
            exec(self.save_pipeline(), namespace)
        self.assertEqual(caught.exception.issues[0]['date'], '2012-02-10')
        self.assertEqual(writes, [])

    def test_real_rejection_report_links_to_unmodified_candidate_evidence(self):
        raw = bars()
        raw[40]['High'] = '99'
        before = copy.deepcopy(raw)
        with self.assertRaises(OHLCVValidationError) as caught:
            validate_rows(raw, 'SYNTH', min_rows=51)
        error = caught.exception
        report = {'symbols': {'SYNTH': {'attempts': []}}}
        writes = []
        report_path = Path('fixture-quality-report.json')
        namespace = {
            're': re, 'Path': Path, 'DATA_DIR': 'fixture-data',
            'QUALITY_REPORT_PATH': report_path, 'DATA_QUALITY_REPORT': report,
            '_symbol_report': lambda symbol: report['symbols'][symbol],
            'write_json_atomic': lambda path, payload:
                writes.append((Path(path), copy.deepcopy(payload))),
            'print': lambda *_args, **_kwargs: None,
        }
        record = compile_function('_record_rejected_source', namespace)
        record('SYNTH', 'provider / primary', raw, error)
        evidence_path = Path('fixture-data') / '_rejected' / 'SYNTH_provider_primary.json'
        self.assertEqual([path for path, _ in writes], [evidence_path, report_path])
        evidence = writes[0][1]
        self.assertEqual(evidence['candidate_rows'], before)
        self.assertEqual(evidence['issues'], error.issues)
        self.assertEqual(evidence['symbol'], 'SYNTH')
        self.assertEqual(evidence['source'], 'provider / primary')
        attempt = writes[1][1]['symbols']['SYNTH']['attempts'][0]
        self.assertEqual(attempt['candidate_evidence'], str(evidence_path))
        self.assertEqual(attempt['status'], 'rejected')
        self.assertEqual(attempt['rows'], 60)
        self.assertEqual(attempt['issues'], error.issues)
        self.assertEqual(raw, before)


if __name__ == '__main__':
    unittest.main()
