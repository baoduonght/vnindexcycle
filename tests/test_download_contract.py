"""Dependency-free tests of the downloader's actual validation wiring.

Only selected function bodies are compiled from the downloader AST; importing it
would require network SDKs and create a data directory. Bars are synthetic test
fixtures. The window fixture reuses reported anomaly dates and envelope values;
it is not a captured provider payload or evidence that a live download succeeds.
"""

import argparse
import ast
import copy
import csv
from contextlib import redirect_stderr
import datetime as datetime_module
from datetime import date, datetime, timedelta
import io
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

from ohlcv_quality import (
    OHLCVValidationError, encode_csv, select_valid_source, validate_rows,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == 'tests' else HERE
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


def bars(count=60, first=date(2012, 1, 1)):
    return [{'Date': (first + timedelta(days=i)).isoformat(),
             'Open': '100', 'High': '110', 'Low': '90',
             'Close': '105', 'Volume': '1000'}
            for i in range(count)]


def compile_window_helpers(namespace=None):
    namespace = {} if namespace is None else namespace
    namespace['datetime'] = datetime_module
    compile_function('build_download_window', namespace)
    compile_function('_filter_source_window', namespace)
    return namespace


class YahooTimestampDouble:
    """Small aware-timestamp surface needed by the Yahoo normalizer."""

    def __init__(self, value):
        self.value = value

    def tz_convert(self, timezone_name):
        if timezone_name == 'Asia/Ho_Chi_Minh':
            offset = datetime_module.timezone(datetime_module.timedelta(hours=7))
        elif timezone_name == 'UTC':
            offset = datetime_module.timezone.utc
        else:
            raise AssertionError(f'unexpected fixture timezone: {timezone_name}')
        self.value = self.value.astimezone(offset)
        return self

    def date(self):
        return self.value.date()


class YahooPandasDouble:
    def to_datetime(self, value, *, unit, utc):
        if unit != 's' or utc is not True:
            raise AssertionError('Yahoo timestamps must be interpreted as UTC seconds')
        return YahooTimestampDouble(datetime.fromtimestamp(
            value, tz=datetime_module.timezone.utc))

    def DataFrame(self, rows, *, columns):
        return SimpleNamespace(rows=rows, columns=list(columns))


class VnstockDateSeriesDouble:
    """Small pandas datetime-series surface used by the Vnstock normalizer."""

    def __init__(self, values):
        self.values = list(values)
        self.dt = self

    def __iter__(self):
        return iter(self.values)

    @property
    def tz(self):
        return self.values[0].tzinfo

    def tz_convert(self, timezone_name):
        if timezone_name != 'Asia/Ho_Chi_Minh':
            raise AssertionError(f'unexpected timezone: {timezone_name}')
        vietnam = datetime_module.timezone(datetime_module.timedelta(hours=7))
        if self.tz is None:
            raise TypeError('cannot convert timezone-naive timestamps')
        return VnstockDateSeriesDouble(value.astimezone(vietnam) for value in self)

    def tz_localize(self, timezone):
        if timezone is not None:
            raise AssertionError('expected timezone removal')
        return VnstockDateSeriesDouble(value.replace(tzinfo=None) for value in self)

    def normalize(self):
        return VnstockDateSeriesDouble(
            value.replace(hour=0, minute=0, second=0, microsecond=0)
            for value in self)


class VnstockIndexDouble:
    def __init__(self, values, *, dtype='int64', name=None):
        self.values = list(values)
        self.dtype = dtype
        self.name = name

    def __len__(self):
        return len(self.values)


class VnstockFrameDouble:
    """The frame operations used to normalize Vnstock history."""

    def __init__(self, rows, *, index=None):
        self.rows = copy.deepcopy(rows)
        self.index = index or VnstockIndexDouble(range(len(self.rows)))

    @property
    def columns(self):
        return list(dict.fromkeys(key for row in self.rows for key in row))

    @property
    def empty(self):
        return not self.rows

    def __len__(self):
        return len(self.rows)

    def copy(self):
        return VnstockFrameDouble(self.rows, index=self.index)

    def sort_values(self, column):
        return VnstockFrameDouble(sorted(self.rows, key=lambda row: row[column]))

    def reset_index(self, *, drop=False):
        if drop:
            return VnstockFrameDouble(
                self.rows, index=VnstockIndexDouble(range(len(self.rows))))
        key = self.index.name or 'index'
        return VnstockFrameDouble([{key: value, **row}
                                   for value, row in zip(self.index.values, self.rows,
                                                         strict=True)])

    def rename(self, *, columns, inplace):
        renamed = [{columns.get(key, key): value for key, value in row.items()}
                   for row in self.rows]
        if inplace:
            self.rows = renamed
            return None
        return VnstockFrameDouble(renamed)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return VnstockFrameDouble([{column: row[column] for column in key}
                                   for row in self.rows])

    def __setitem__(self, key, values):
        for row, value in zip(self.rows, values, strict=True):
            row[key] = value


class VnstockPandasDouble:
    def __init__(self):
        self.api = SimpleNamespace(types=SimpleNamespace(
            is_datetime64_any_dtype=lambda dtype:
                str(dtype).startswith('datetime64[')))

    def DataFrame(self, data, *, index):
        columns = list(data)
        values = [list(data[column]) for column in columns]
        if any(len(column_values) != len(index) for column_values in values):
            raise AssertionError('constructed columns and index must have equal lengths')
        rows = [dict(zip(columns, row_values, strict=True))
                for row_values in zip(*values, strict=True)]
        return VnstockFrameDouble(rows, index=index)

    def to_datetime(self, values, *, errors):
        if errors != 'raise':
            raise AssertionError('Vnstock dates must reject invalid input')
        return VnstockDateSeriesDouble(datetime.fromisoformat(value)
                                       if isinstance(value, str) else value
                                       for value in values)


def compile_yahoo_helpers():
    namespace = {'re': re, 'pd': YahooPandasDouble()}
    compile_function('_yahoo_symbol', namespace)
    compile_function('_normalize_yahoo_chart', namespace)
    return namespace


def yahoo_payload(timestamps=None, currency='VND', **arrays):
    """Synthetic Yahoo-shaped response, independent of live symbol coverage."""
    if timestamps is None:
        timestamps = [int(datetime(2021, 1, 1, 12, tzinfo=datetime_module.timezone.utc).timestamp())]
    quote = {'open': [100.0], 'high': [110.0], 'low': [90.0],
             'close': [105.0], 'volume': [1000]}
    quote.update(arrays)
    meta = {'exchangeTimezoneName': 'Asia/Ho_Chi_Minh'}
    if currency is not None:
        meta['currency'] = currency
    return {'chart': {'result': [{
        'timestamp': timestamps,
        'meta': meta,
        'indicators': {'quote': [quote]},
    }], 'error': None}}


def window_fixture():
    """Reported anomaly dates and envelopes, with otherwise synthetic bars."""
    old = bars(2)
    old[0].update(Date='2012-02-10', Open='410.11', High='409.97',
                  Low='400', Close='409')
    old[1].update(Date='2020-09-29', Open='905', High='910',
                  Low='904.69', Close='903.98')
    after = bars(1, first=date(2023, 1, 1))
    after[0]['High'] = '99'
    return old + bars(730, first=date(2021, 1, 1)) + after


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
        self.crops = []
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

        def record_crop(symbol, source, raw_rows, counts, start_date, end_date):
            self.crops.append((symbol, source, copy.deepcopy(raw_rows),
                               dict(counts), start_date, end_date))

        self.namespace = {
            'pd': self.pd, 'select_valid_source': select_valid_source,
            '_iter_ticker_sources': providers, '_symbol_report': symbol_report,
            '_record_rejected_source': record_rejection,
            '_record_window_crop': record_crop,
            'write_json_atomic': record_write,
            'QUALITY_REPORT_PATH': self.report_path,
            'DATA_QUALITY_REPORT': self.report,
        }
        compile_window_helpers(self.namespace)
        self.fetch = compile_function('fetch_ticker_multi_source', self.namespace)


class DownloadWindowTests(unittest.TestCase):
    def setUp(self):
        namespace = compile_window_helpers()
        self.build = namespace['build_download_window']
        self.filter = namespace['_filter_source_window']

    def test_utc_request_bounds_include_both_full_days(self):
        window = self.build('2021-01-01', '2022-12-31')
        utc = datetime_module.timezone.utc
        self.assertEqual(window, {
            'start': '2021-01-01', 'end': '2022-12-31',
            'from_ts': int(datetime(2021, 1, 1, tzinfo=utc).timestamp()),
            'to_ts': int(datetime(2022, 12, 31, 23, 59, 59,
                                   tzinfo=utc).timestamp()),
        })
        one_day = self.build('2024-02-29', '2024-02-29')
        self.assertEqual(one_day['to_ts'] - one_day['from_ts'], 86399)

    def test_only_ordered_canonical_iso_arguments_are_accepted(self):
        invalid = ('2021-1-01', '20210101', ' 2021-01-01', '2021-01-01 ',
                   '2021-01-01T00:00:00', '2021-02-29', '', None, 20210101)
        for value in invalid:
            for start, end in ((value, '2022-12-31'), ('2021-01-01', value)):
                with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                    self.build(start, end)
        with self.assertRaises(ValueError):
            self.build('2022-01-02', '2022-01-01')

    def test_explicit_window_preserves_every_2021_and_2022_bar_unchanged(self):
        raw = window_fixture()
        before = copy.deepcopy(raw)
        kept, counts = self.filter(raw, '2021-01-01', '2022-12-31')
        self.assertEqual(kept, before[2:-1])
        self.assertEqual(raw, before)
        self.assertEqual(counts, {
            'input_rows': 733, 'kept_rows': 730, 'excluded_before_start': 2,
            'excluded_after_end': 1, 'invalid_dates_kept': 0,
        })
        self.assertEqual(kept[0]['Date'], '2021-01-01')
        self.assertEqual(kept[-1]['Date'], '2022-12-31')
        self.assertEqual(sum(row['Date'].startswith('2022-') for row in kept), 365)

    def test_missing_and_malformed_dates_are_kept_for_strict_validation(self):
        malformed = []
        for stamp in ('bad', '20120210', '2012-2-10', '2020-02-30', None):
            row = bars(1)[0]
            row['Date'] = stamp
            malformed.append(row)
        missing = bars(1)[0]
        del missing['Date']
        malformed.append(missing)
        raw = malformed + bars(60, first=date(2021, 1, 1))
        before = copy.deepcopy(raw)
        kept, counts = self.filter(raw, '2021-01-01', '2022-12-31')
        self.assertEqual(kept, before)
        self.assertEqual(raw, before)
        self.assertEqual(counts['invalid_dates_kept'], len(malformed))
        self.assertEqual(counts['kept_rows'], len(raw))
        with self.assertRaises(OHLCVValidationError) as caught:
            validate_rows(kept, 'VNINDEX')
        self.assertEqual(len(caught.exception.issues), len(malformed))

    def test_date_objects_and_datetimes_use_daily_inclusive_boundaries(self):
        raw = bars(4)
        raw[0]['Date'] = date(2020, 12, 31)
        raw[1]['Date'] = datetime(2021, 1, 1, 23, 59, 59)
        raw[2]['Date'] = date(2022, 12, 31)
        raw[3]['Date'] = datetime(2023, 1, 1)
        before = copy.deepcopy(raw)
        kept, counts = self.filter(raw, '2021-01-01', '2022-12-31')
        self.assertEqual(kept, before[1:3])
        self.assertEqual(raw, before)
        self.assertEqual(counts['excluded_before_start'], 1)
        self.assertEqual(counts['excluded_after_end'], 1)

    def test_default_window_keeps_the_legacy_full_candidate(self):
        raw = window_fixture()
        before = copy.deepcopy(raw)
        kept, counts = self.filter(raw)
        self.assertEqual(kept, before)
        self.assertEqual(raw, before)
        self.assertEqual(counts['input_rows'], len(raw))
        self.assertEqual(counts['kept_rows'], len(raw))
        self.assertEqual(counts['excluded_before_start'], 0)
        self.assertEqual(counts['excluded_after_end'], 0)


class DownloadWrapperTests(unittest.TestCase):
    def test_explicit_window_excludes_old_anomalies_before_real_validation(self):
        raw = window_fixture()
        before = copy.deepcopy(raw)
        primary = FrameDouble(raw)

        def candidates():
            yield 'reported old anomalies', primary
            self.fail('old out-of-window anomalies caused an unnecessary fallback')

        harness = WrapperHarness(candidates())
        frame, source = harness.fetch('VNINDEX', history_start='2021-01-01',
                                      history_end='2022-12-31')
        self.assertEqual(source, 'reported old anomalies')
        self.assertEqual(primary.rows, before)
        self.assertEqual(harness.rejected, [])
        self.assertEqual([row['Date'] for row in frame.rows],
                         [datetime.fromisoformat(row['Date']) for row in before[2:-1]])
        self.assertEqual(len(frame.rows), 730)
        self.assertEqual(len(harness.crops), 1)
        symbol, provider, candidate, counts, start, end = harness.crops[0]
        self.assertEqual((symbol, provider, start, end),
                         ('VNINDEX', source, '2021-01-01', '2022-12-31'))
        self.assertEqual(candidate, before)
        self.assertEqual(counts, {
            'input_rows': 733, 'kept_rows': 730, 'excluded_before_start': 2,
            'excluded_after_end': 1, 'invalid_dates_kept': 0,
        })

    def test_bad_bar_inside_window_still_rejects_provider_and_falls_back(self):
        raw = window_fixture()
        raw[400]['High'] = '99'
        before = copy.deepcopy(raw)
        primary = FrameDouble(raw)
        harness = WrapperHarness([('bad in range', primary),
                                  ('complete fallback', FrameDouble(window_fixture()))])
        frame, source = harness.fetch('VNINDEX', history_start='2021-01-01',
                                      history_end='2022-12-31')
        self.assertEqual(source, 'complete fallback')
        self.assertEqual(harness.visited, ['bad in range', 'complete fallback'])
        self.assertEqual(primary.rows, before)
        self.assertEqual(len(frame.rows), 730)
        self.assertEqual(len(harness.crops), 2)
        self.assertEqual(len(harness.rejected), 1)
        rejected = harness.rejected[0]
        self.assertEqual(rejected[2], before[2:-1])
        self.assertEqual(rejected[3].issues[0]['date'], before[400]['Date'])
        self.assertEqual(rejected[3].issues[0]['raw_values'], before[400])
        self.assertIn('High is below max(Open, Close)',
                      rejected[3].issues[0]['reasons'])

    def test_unknown_date_cannot_escape_window_validation_or_fallback(self):
        for stamp in ('malformed', '20120210', None):
            with self.subTest(stamp=stamp):
                raw = bars(60, first=date(2021, 1, 1))
                raw[0]['Date'] = stamp
                before = copy.deepcopy(raw)
                primary = FrameDouble(raw)
                harness = WrapperHarness([
                    ('unknown date', primary),
                    ('complete', FrameDouble(bars(60, first=date(2021, 1, 1)))),
                ])
                frame, source = harness.fetch('VNINDEX', history_start='2021-01-01',
                                              history_end='2022-12-31')
                self.assertEqual(source, 'complete')
                self.assertEqual(primary.rows, before)
                self.assertEqual(len(frame.rows), 60)
                self.assertEqual(harness.rejected[0][2], before)
                self.assertEqual(harness.rejected[0][3].issues[0]['raw_values'], before[0])
                self.assertEqual(harness.crops[0][3]['invalid_dates_kept'], 1)

    def test_window_cannot_make_short_candidate_bypass_minimum_or_merge_sources(self):
        short = bars(60, first=date(2020, 12, 1))
        harness = WrapperHarness([
            ('short after crop', FrameDouble(short)),
            ('complete', FrameDouble(bars(60, first=date(2021, 1, 1)))),
        ])
        frame, source = harness.fetch('VNINDEX', history_start='2021-01-01',
                                      history_end='2022-12-31')
        self.assertEqual(source, 'complete')
        self.assertEqual(len(frame.rows), 60)
        self.assertEqual(harness.rejected[0][3].issues[0]['raw_values'],
                         {'row_count': 29, 'min_rows': 51})

    def test_omitting_window_does_not_ignore_known_old_invalid_bars(self):
        harness = WrapperHarness([
            ('old anomalies', FrameDouble(window_fixture())),
            ('complete', FrameDouble(bars(60))),
        ])
        _, source = harness.fetch('VNINDEX')
        self.assertEqual(source, 'complete')
        self.assertEqual(harness.rejected[0][3].issues[0]['date'], '2012-02-10')
        self.assertEqual(len(harness.rejected[0][2]), 733)

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


class DownloadCliTests(unittest.TestCase):
    class FirstFetchReached(Exception):
        pass

    def harness(self):
        fetches, writes = [], []
        namespace = {
            'argparse': argparse, 'datetime': datetime_module,
            'time': SimpleNamespace(time=lambda: 0),
            'START_DATE_STR': 'unchanged-start', 'NOW_DATE_STR': 'unchanged-end',
            'START_TS': -1, 'NOW_TS': -1,
            'DATA_DIR': 'fixture-data', 'ALL_HISTORICAL_VN30_TICKERS': [],
            'VNSTOCK_OAPI_AVAILABLE': False, 'VNSTOCK_AVAILABLE': False,
            'DATA_QUALITY_REPORT': {'symbols': {}},
            'QUALITY_REPORT_PATH': Path('fixture-quality-report.json'),
            'write_json_atomic': lambda path, payload:
                writes.append((path, copy.deepcopy(payload))),
            'print': lambda *_args, **_kwargs: None,
        }
        compile_window_helpers(namespace)

        def first_fetch(symbol, **kwargs):
            fetches.append((symbol, kwargs, {
                key: namespace[key]
                for key in ('START_DATE_STR', 'NOW_DATE_STR', 'START_TS', 'NOW_TS')
            }))
            raise self.FirstFetchReached()

        namespace['fetch_ticker_multi_source'] = first_fetch
        main = compile_function('main', namespace)
        return main, namespace, fetches, writes

    def test_explicit_dates_configure_api_globals_and_filter_before_first_fetch(self):
        main, namespace, fetches, writes = self.harness()
        with self.assertRaises(self.FirstFetchReached):
            main(['--start-date', '2021-01-01', '--end-date', '2022-12-31'])
        expected = namespace['build_download_window']('2021-01-01', '2022-12-31')
        self.assertEqual(fetches, [('VNINDEX', {
            'history_start': '2021-01-01', 'history_end': '2022-12-31',
        }, {'START_DATE_STR': expected['start'], 'NOW_DATE_STR': expected['end'],
            'START_TS': expected['from_ts'], 'NOW_TS': expected['to_ts']})])
        self.assertEqual(writes[-1][1]['requested_range'],
                         {**expected, 'timestamp_timezone': 'UTC'})

    def test_cli_defaults_preserve_2012_history_and_use_utc_today(self):
        main, namespace, fetches, _writes = self.harness()
        before = datetime_module.datetime.now(datetime_module.timezone.utc).date()
        with self.assertRaises(self.FirstFetchReached):
            main([])
        after = datetime_module.datetime.now(datetime_module.timezone.utc).date()
        self.assertEqual(namespace['START_DATE_STR'], '2012-01-01')
        self.assertIn(namespace['NOW_DATE_STR'], {before.isoformat(), after.isoformat()})
        self.assertEqual(fetches[0][1], {
            'history_start': '2012-01-01', 'history_end': namespace['NOW_DATE_STR'],
        })

    def test_invalid_arguments_fail_before_fetch_or_quality_artifact(self):
        for arguments in (
            ['--start-date', '20210101'],
            ['--end-date', '2022-2-01'],
            ['--start-date', '2022-01-02', '--end-date', '2022-01-01'],
            ['--start-date', '2021-02-29'],
        ):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                main, namespace, fetches, writes = self.harness()
                with self.assertRaises(SystemExit) as caught:
                    main(arguments)
                self.assertEqual(caught.exception.code, 2)
                self.assertEqual(fetches, [])
                self.assertEqual(writes, [])
                self.assertEqual(namespace['START_DATE_STR'], 'unchanged-start')
                self.assertEqual(namespace['NOW_DATE_STR'], 'unchanged-end')
                self.assertEqual((namespace['START_TS'], namespace['NOW_TS']), (-1, -1))


class VnstockHistoryContractTests(unittest.TestCase):
    def setUp(self):
        self.normalize = compile_function('_normalize_vnstock_history',
                                          {'pd': VnstockPandasDouble()})

    def test_lowercase_history_fields_normalize_without_changing_ohlcv_values(self):
        history = VnstockFrameDouble([
            {'time': '2021-01-01 20:00:00', 'open': 100.25,
             'high': 111.0, 'low': 90.0, 'close': 105.75, 'volume': 123},
            {'time': '2021-01-02', 'open': 101.5,
             'high': 112.0, 'low': None, 'close': 106.25, 'volume': 456},
        ])
        before = copy.deepcopy(history.rows)

        frame = self.normalize('FPT', history)

        self.assertEqual(frame.columns, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        self.assertEqual(frame.rows, [
            {'Date': datetime(2021, 1, 1), 'Open': 100.25, 'High': 111.0,
             'Low': 90.0, 'Close': 105.75, 'Volume': 123},
            {'Date': datetime(2021, 1, 2), 'Open': 101.5, 'High': 112.0,
             'Low': None, 'Close': 106.25, 'Volume': 456},
        ])
        self.assertEqual(history.rows, before)

    def test_timezone_conversion_uses_vietnam_local_trading_date(self):
        utc = datetime_module.timezone.utc
        history = VnstockFrameDouble([
            {'time': datetime(2021, 1, 1, 14, tzinfo=utc), 'open': 1,
             'high': 2, 'low': 0, 'close': 1, 'volume': 3},
            {'time': datetime(2021, 1, 1, 18, tzinfo=utc), 'open': 4,
             'high': 5, 'low': 3, 'close': 4, 'volume': 6},
        ])

        frame = self.normalize('FPT', history)

        self.assertEqual([row['Date'] for row in frame.rows], [
            datetime(2021, 1, 1), datetime(2021, 1, 2),
        ])
        self.assertTrue(all(row['Date'].tzinfo is None for row in frame.rows))

    def test_missing_time_or_ohlcv_fields_are_rejected(self):
        full = {'time': datetime(2021, 1, 1), 'open': 1, 'high': 2,
                'low': 0, 'close': 1, 'volume': 3}
        for field in ('time', 'open', 'high', 'low', 'close', 'volume'):
            with self.subTest(missing=field):
                row = dict(full)
                del row[field]
                with self.assertRaisesRegex(ValueError, 'missing fields'):
                    self.normalize('FPT', VnstockFrameDouble([row]))

    def test_datetime_index_is_a_valid_date_source_but_range_index_is_not(self):
        utc = datetime_module.timezone.utc
        index = VnstockIndexDouble(
            [datetime(2021, 1, 1, 18, tzinfo=utc)],
            dtype='datetime64[ns, UTC]', name='time')
        history = VnstockFrameDouble(
            [{'open': 1, 'high': 2, 'low': 0, 'close': 1, 'volume': 3}],
            index=index)

        frame = self.normalize('FPT', history)

        self.assertEqual(frame.rows, [{'Date': datetime(2021, 1, 2), 'Open': 1,
                                       'High': 2, 'Low': 0, 'Close': 1,
                                       'Volume': 3}])
        row = {'open': 1, 'high': 2, 'low': 0, 'close': 1, 'volume': 3}
        with self.assertRaisesRegex(ValueError, 'time/date index'):
            self.normalize('FPT', VnstockFrameDouble([row]))

    def test_empty_history_returns_none(self):
        self.assertIsNone(self.normalize('FPT', VnstockFrameDouble([])))

    def test_fhsc_requests_daily_bars_from_the_index_or_equity_endpoint(self):
        def run_candidate(symbol, *, api_key=None):
            calls = []
            errors = []
            history = VnstockFrameDouble([
                {'time': row['Date'], 'open': row['Open'], 'high': row['High'],
                 'low': row['Low'], 'close': row['Close'], 'volume': row['Volume']}
                for row in bars(first=date(2021, 1, 1))
            ])

            def market(provider):
                calls.append(('market', provider))

                def instrument(kind, requested_symbol):
                    calls.append((kind, requested_symbol))

                    def ohlcv(**kwargs):
                        calls.append(('ohlcv', kwargs))
                        return history

                    return SimpleNamespace(ohlcv=ohlcv)

                return SimpleNamespace(index=lambda value: instrument('index', value),
                                       equity=lambda value: instrument('equity', value))

            def get(_url, **_kwargs):
                return SimpleNamespace(status_code=404, raise_for_status=lambda: None,
                                       json=lambda: {})

            def yahoo_symbol(requested_symbol):
                calls.append(('yahoo', requested_symbol))
                return f'{requested_symbol}.VN'

            namespace = {
                'os': SimpleNamespace(environ={'FHSC_API_KEY': api_key} if api_key else {}),
                'requests': SimpleNamespace(get=get), 'Market': market,
                '_normalize_vnstock_history': self.normalize,
                '_record_fetch_error': lambda *args: errors.append(args),
                'START_TS': 1, 'NOW_TS': 2, 'START_DATE_STR': '2021-01-01',
                'NOW_DATE_STR': '2022-12-31', 'VNSTOCK_OAPI_AVAILABLE': False,
                'VNSTOCK_AVAILABLE': True, 'TV_DATAFEED_AVAILABLE': False,
                '_yahoo_symbol': yahoo_symbol,
                '_normalize_yahoo_chart': lambda *_args: VnstockFrameDouble([]),
            }
            iterator = compile_function('_iter_ticker_sources', namespace)
            if api_key:
                frame, source = next(iterator(symbol))
                self.assertEqual(source, 'Vnstock Open API (FHSC)')
                self.assertEqual(len(frame), 60)
                self.assertEqual(frame.rows[0]['Date'], datetime(2021, 1, 1))
            else:
                self.assertEqual(list(iterator(symbol)), [])
            self.assertEqual(errors, [])
            return calls

        for symbol, endpoint in (('VNINDEX', 'index'), ('FPT', 'equity')):
            with self.subTest(symbol=symbol):
                self.assertEqual(run_candidate(symbol, api_key='fake-key'), [
                    ('market', 'fhsc'), (endpoint, symbol),
                    ('ohlcv', {'start': '2021-01-01', 'end': '2022-12-31',
                               'interval': '1D', 'compat': 'v6', 'floating': None}),
                ])
        self.assertEqual(run_candidate('FPT'), [('yahoo', 'FPT')])

    def test_fhsc_history_request_precedes_yahoo_and_uses_shared_selector(self):
        iterator = function_named('_iter_ticker_sources')
        market_calls = calls_named(iterator, 'Market')
        self.assertEqual(len(market_calls), 1)
        fhsc_yields = [node for node in ast.walk(iterator)
                       if isinstance(node, ast.Yield)
                       and isinstance(node.value, ast.Tuple)
                       and len(node.value.elts) == 2
                       and isinstance(node.value.elts[1], ast.Constant)
                       and node.value.elts[1].value == 'Vnstock Open API (FHSC)']
        self.assertEqual(len(fhsc_yields), 1)
        normalize_calls = calls_named(iterator, '_normalize_vnstock_history')
        self.assertEqual(len(normalize_calls), 1)
        yahoo_calls = calls_named(iterator, '_yahoo_symbol')
        self.assertEqual(len(yahoo_calls), 1)
        self.assertLess(market_calls[0].lineno, normalize_calls[0].lineno)
        self.assertLess(normalize_calls[0].lineno, fhsc_yields[0].lineno)
        self.assertLess(fhsc_yields[0].lineno, yahoo_calls[0].lineno)

        fetch = function_named('fetch_ticker_multi_source')
        selector_calls = [node for node in ast.walk(fetch)
                          if isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Name)
                          and node.func.id == 'select_valid_source']
        self.assertEqual(len(selector_calls), 1)
        self.assertIsInstance(selector_calls[0].args[0], ast.Call)
        self.assertEqual(ast.unparse(selector_calls[0].args[0]), 'candidates()')
        candidates = next(node for node in fetch.body
                          if isinstance(node, ast.FunctionDef)
                          and node.name == 'candidates')
        self.assertTrue(any(isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Name)
                            and node.func.id == '_iter_ticker_sources'
                            for node in ast.walk(candidates)))


class YahooFallbackContractTests(unittest.TestCase):
    def setUp(self):
        namespace = compile_yahoo_helpers()
        self.yahoo_symbol = namespace['_yahoo_symbol']
        self.normalize = namespace['_normalize_yahoo_chart']

    def test_symbol_mapping_uses_index_identifier_and_exchange_suffix_for_stocks(self):
        self.assertEqual(self.yahoo_symbol('VNINDEX'), '^VNINDEX.VN')
        self.assertEqual(self.yahoo_symbol('eib'), 'EIB.VN')
        self.assertEqual(self.yahoo_symbol('vnm'), 'VNM.VN')
        self.assertEqual(self.yahoo_symbol('csm'), 'CSM.VN')

    def test_chart_dates_use_exchange_timezone_and_preserve_each_ohlcv_value(self):
        timestamps = [
            int(datetime(2021, 1, 1, 20, tzinfo=datetime_module.timezone.utc).timestamp()),
            int(datetime(2021, 1, 2, 16, tzinfo=datetime_module.timezone.utc).timestamp()),
        ]
        payload = yahoo_payload(
            timestamps, open=[100.25, 101.5], high=[111.0, 112.0],
            low=[90.0, 91.0], close=[105.75, 106.25], volume=[123, 456])
        before = copy.deepcopy(payload)
        frame = self.normalize('FPT', payload)
        self.assertEqual(payload, before)
        self.assertEqual(frame.columns, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        self.assertEqual(frame.rows, [
            {'Date': '2021-01-02', 'Open': 0.10025, 'High': 0.111,
             'Low': 0.09, 'Close': 0.10575, 'Volume': 123},
            {'Date': '2021-01-02', 'Open': 0.1015, 'High': 0.112,
             'Low': 0.091, 'Close': 0.10625, 'Volume': 456},
        ])

    def test_vnindex_values_are_points_and_are_not_currency_scaled(self):
        payload = yahoo_payload(open=[1712.5], high=[1730.0], low=[1701.0],
                                close=[1725.25], volume=[0])
        before = copy.deepcopy(payload)
        frame = self.normalize('VNINDEX', payload)
        self.assertEqual(payload, before)
        self.assertEqual(frame.rows[0], {
            'Date': '2021-01-01', 'Open': 1712.5, 'High': 1730.0,
            'Low': 1701.0, 'Close': 1725.25, 'Volume': 0,
        })

    def test_yahoo_unit_label_is_preserved_as_validated_source_metadata(self):
        source = 'Yahoo Finance (EIB.VN; VND→thousand VND)'
        harness = WrapperHarness([(source, FrameDouble(bars(60)))])
        _frame, selected_source = harness.fetch('EIB')
        self.assertEqual(selected_source, source)
        metadata = harness.writes[-1][1]['symbols']['EIB']
        self.assertEqual(metadata['source'], source)
        self.assertEqual(metadata['status'], 'validated')

    def test_security_currency_must_be_present_and_vnd_before_scaling(self):
        for currency in (None, 'USD', ''):
            with self.subTest(currency=currency), self.assertRaisesRegex(
                    ValueError, 'currency'):
                self.normalize('FPT', yahoo_payload(currency=currency))

    def test_required_ohlcv_arrays_and_matching_lengths_are_mandatory(self):
        for field in ('open', 'high', 'low', 'close', 'volume'):
            with self.subTest(missing=field):
                payload = yahoo_payload()
                del payload['chart']['result'][0]['indicators']['quote'][0][field]
                with self.assertRaisesRegex(ValueError, 'missing OHLCV arrays'):
                    self.normalize('FPT', payload)
        with self.assertRaisesRegex(ValueError, 'array lengths differ'):
            self.normalize('FPT', yahoo_payload(volume=[1000, 2000]))

    def test_chart_error_is_rejected_and_null_volume_is_preserved_for_strict_validation(self):
        errored = yahoo_payload()
        errored['chart']['error'] = {'code': 'Not Found'}
        with self.assertRaisesRegex(ValueError, 'no result'):
            self.normalize('FPT', errored)

        start = int(datetime(2021, 1, 1, 12,
                             tzinfo=datetime_module.timezone.utc).timestamp())
        timestamps = [start + index * 86400 for index in range(60)]
        volume = [1000] * len(timestamps)
        opens = [100.0] * len(timestamps)
        volume[25] = None
        opens[24] = None
        frame = self.normalize('FPT', yahoo_payload(timestamps, volume=volume,
                                open=opens, high=[110.0] * 60,
                                low=[90.0] * 60, close=[105.0] * 60))
        self.assertIsNone(frame.rows[24]['Open'])
        self.assertIsNone(frame.rows[25]['Volume'])
        rejected = []
        chosen, source = select_valid_source(
            [('Yahoo Finance', frame.rows)], 'FPT', min_rows=51,
            on_rejected=lambda name, rows, error: rejected.append((name, rows, error)))
        self.assertIsNone(chosen)
        self.assertIsNone(source)
        self.assertEqual(rejected[0][0], 'Yahoo Finance')
        self.assertIsNone(rejected[0][1][24]['Open'])
        self.assertIsNone(rejected[0][1][25]['Volume'])

    def test_yahoo_request_is_late_and_flows_through_shared_source_validation(self):
        iterator = function_named('_iter_ticker_sources')
        yahoo_calls = [node for node in ast.walk(iterator)
                       if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Name)
                       and node.func.id == '_yahoo_symbol']
        self.assertEqual(len(yahoo_calls), 1)
        yahoo_line = yahoo_calls[0].lineno
        earlier_yields = [node.lineno for node in ast.walk(iterator)
                          if isinstance(node, ast.Yield) and node.lineno < yahoo_line]
        self.assertTrue(earlier_yields, 'Yahoo must follow the existing providers')
        vnstock_calls = [node for node in ast.walk(iterator)
                         if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name)
                         and node.func.id == 'Market']
        self.assertTrue(vnstock_calls)
        self.assertLess(max(node.lineno for node in vnstock_calls), yahoo_line,
                        'Yahoo must follow the Vnstock Market provider')

        yahoo_requests = [node for node in ast.walk(iterator)
                          if isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Attribute)
                          and isinstance(node.func.value, ast.Name)
                          and node.func.value.id == 'requests'
                          and node.func.attr == 'get' and node.lineno > yahoo_line]
        self.assertEqual(len(yahoo_requests), 1)
        timeout = next((keyword.value for keyword in yahoo_requests[0].keywords
                        if keyword.arg == 'timeout'), None)
        self.assertIsNotNone(timeout)
        self.assertGreater(ast.literal_eval(timeout), 0)
        self.assertTrue(any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == '_record_fetch_error' and node.lineno > yahoo_line
            for node in ast.walk(iterator)))
        yahoo_yields = [node for node in ast.walk(iterator)
                        if isinstance(node, ast.Yield)
                        and node.lineno > yahoo_line
                        and isinstance(node.value, ast.Tuple)
                        and len(node.value.elts) == 2
                        and isinstance(node.value.elts[1], ast.JoinedStr)
                        and any(isinstance(part, ast.Constant)
                                and isinstance(part.value, str)
                                and part.value.startswith('Yahoo Finance (')
                                for part in node.value.elts[1].values)]
        self.assertEqual(len(yahoo_yields), 1)
        tv_calls = [node for node in ast.walk(iterator)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == 'TvDatafeed']
        self.assertTrue(tv_calls)
        self.assertGreater(min(node.lineno for node in tv_calls),
                           yahoo_yields[0].lineno,
                           'TradingView must be the last resort after Yahoo')
        provider_labels = [node.value for node in ast.walk(iterator)
                           if isinstance(node, ast.Constant)
                           and isinstance(node.value, str)]
        self.assertIn('VND→thousand VND', provider_labels)
        self.assertIn('index points', provider_labels)
        self.assertIn('TradingView (', provider_labels)
        tv_yields = [node for node in ast.walk(iterator)
                     if isinstance(node, ast.Yield)
                     and isinstance(node.value, ast.Tuple)
                     and len(node.value.elts) == 2
                     and isinstance(node.value.elts[1], ast.JoinedStr)
                     and any(isinstance(part, ast.Constant)
                             and isinstance(part.value, str)
                             and part.value.startswith('TradingView (')
                             for part in node.value.elts[1].values)]
        self.assertEqual(len(tv_yields), 1)
        self.assertGreater(tv_yields[0].lineno, yahoo_yields[0].lineno)
        self.assertTrue(any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == '_record_fetch_error'
            and node.lineno > tv_yields[0].lineno
            for node in ast.walk(iterator)))

    def test_tradingview_is_optional_and_uses_daily_ohlcv_last(self):
        source = DOWNLOADER.read_text(encoding='utf-8')
        self.assertIn('from tvDatafeed import TvDatafeed, Interval', source)
        self.assertIn('TV_DATAFEED_AVAILABLE = False', source)
        iterator = function_named('_iter_ticker_sources')
        tv_call = next(node for node in ast.walk(iterator)
                       if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute)
                       and node.func.attr == 'get_hist')
        kwargs = {k.arg: ast.unparse(k.value) for k in tv_call.keywords}
        self.assertEqual(kwargs, {'symbol':'symbol', 'exchange':"'HOSE'",
                                  'interval':'Interval.in_daily', 'n_bars':'10000'})

        fetch = function_named('fetch_ticker_multi_source')
        candidate_factory = next(node for node in fetch.body
                                 if isinstance(node, ast.FunctionDef)
                                 and node.name == 'candidates')
        select_calls = [node for node in ast.walk(fetch)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == 'select_valid_source']
        self.assertEqual(len(select_calls), 1)
        self.assertIsInstance(select_calls[0].args[0], ast.Call)
        self.assertIsInstance(select_calls[0].args[0].func, ast.Name)
        self.assertEqual(select_calls[0].args[0].func.id, 'candidates')
        self.assertTrue(any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == '_iter_ticker_sources'
            for node in ast.walk(candidate_factory)))


class DownloadIntegrationContractTests(unittest.TestCase):
    def test_every_provider_branch_yields_without_early_dataset_returns(self):
        iterator = function_named('_iter_ticker_sources')
        yields = [node for node in ast.walk(iterator) if isinstance(node, ast.Yield)]
        self.assertEqual(len(yields), 7)
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

    def test_real_window_report_preserves_original_input_and_declares_crop_counts(self):
        raw = window_fixture()
        before = copy.deepcopy(raw)
        helpers = compile_window_helpers()
        _kept, counts = helpers['_filter_source_window'](
            raw, '2021-01-01', '2022-12-31')
        report = {'symbols': {'VNINDEX': {'attempts': []}}}
        writes = []
        report_path = Path('fixture-quality-report.json')
        namespace = {
            're': re, 'Path': Path, 'DATA_DIR': 'fixture-data',
            'QUALITY_REPORT_PATH': report_path, 'DATA_QUALITY_REPORT': report,
            '_symbol_report': lambda symbol: report['symbols'][symbol],
            'write_json_atomic': lambda path, payload:
                writes.append((Path(path), copy.deepcopy(payload))),
        }
        record = compile_function('_record_window_crop', namespace)
        record('VNINDEX', 'provider / primary', raw, counts, '2021-01-01', '2022-12-31')
        self.assertEqual(len(writes), 2)
        evidence_path, evidence = writes[0]
        self.assertEqual(evidence_path.parent, Path('fixture-data') / '_window_input')
        self.assertEqual(evidence['original_candidate_rows'], before)
        self.assertEqual(evidence['counts'], counts)
        self.assertEqual((evidence['requested_start'], evidence['requested_end']),
                         ('2021-01-01', '2022-12-31'))
        self.assertEqual(writes[1][0], report_path)
        entry = writes[1][1]['symbols']['VNINDEX']['attempts'][0]
        self.assertEqual(entry, {
            'source': 'provider / primary', 'status': 'window_filter',
            'start': '2021-01-01', 'end': '2022-12-31', **counts,
            'original_candidate_evidence': str(evidence_path),
        })
        self.assertEqual(raw, before)


if __name__ == '__main__':
    unittest.main()
