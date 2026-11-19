"""Data-contract regressions; fixtures are synthetic, not historical evidence."""

import copy
from contextlib import contextmanager
import csv
from datetime import date, datetime
import io
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

from ohlcv_quality import (
    FIELDS, OHLCVValidationError, encode_csv, select_valid_source, validate_rows,
    write_json_atomic, write_text_atomic,
)


def bar(stamp='2024-01-02', **changes):
    row = {'Date': stamp, 'Open': 100.0, 'High': 110.0, 'Low': 90.0,
           'Close': 105.0, 'Volume': 1200.0}
    row.update(changes)
    return row


@contextmanager
def temporary_directory():
    # Inherit normal directory ACLs on Windows sandbox hosts. Python 3.14's
    # TemporaryDirectory applies owner-only ACLs that may exclude the sandbox.
    path = Path(__file__).resolve().parent.parent/('ohlcv_quality_test_' + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        for file_path in (path/'nested'/'report.json', path/'report.json', path/'report.txt'):
            file_path.unlink(missing_ok=True)
        try:
            (path/'nested').rmdir()
        except FileNotFoundError:
            pass
        path.rmdir()


class RowValidationTests(unittest.TestCase):
    def rejection(self, rows):
        with self.assertRaises(OHLCVValidationError) as caught:
            validate_rows(rows, 'TEST', min_rows=0)
        return caught.exception

    def test_valid_rows_are_copied_and_normalized_without_sorting_or_extra_fields(self):
        original = [bar('2024-01-03', Open='100', Volume='0', Note='retained in raw only'),
                    bar(date(2024, 1, 2))]
        before = copy.deepcopy(original)
        cleaned = validate_rows(iter(original), 'TEST', min_rows=2)
        self.assertEqual(original, before)
        self.assertEqual([row['Date'] for row in cleaned], ['2024-01-03', '2024-01-02'])
        for source, result in zip(original, cleaned):
            self.assertIsNot(source, result)
            self.assertEqual(tuple(result), FIELDS)
            self.assertTrue(all(isinstance(result[name], float) for name in FIELDS[1:]))

    def test_positive_zero_range_and_zero_volume_are_allowed(self):
        rows = [bar(Open=100, High=100, Low=100, Close=100, Volume=0)]
        self.assertEqual(validate_rows(rows, 'TEST', min_rows=1)[0]['Volume'], 0.0)

    def test_datetime_and_whitespace_dates_normalize_to_iso(self):
        cleaned = validate_rows([bar(datetime(2024, 1, 2)), bar(' 2024-01-03 ')],
                                'TEST', min_rows=0)
        self.assertEqual([row['Date'] for row in cleaned], ['2024-01-02', '2024-01-03'])

    def test_missing_datetime_sentinel_cannot_bypass_iso_validation(self):
        class MissingDatetime(datetime):
            def date(self):
                return self

            def isoformat(self, *args, **kwargs):
                return 'NaT'

        error = self.rejection([bar(MissingDatetime(2024, 1, 2))])
        self.assertTrue(any('valid ISO date' in reason
                            for reason in error.issues[0]['reasons']))

    def test_identical_duplicates_are_rejected_with_both_csv_row_locations(self):
        first = bar()
        error = self.rejection([first, dict(first)])
        self.assertEqual(error.symbol, 'TEST')
        self.assertEqual(len(error.issues), 1)
        issue = error.issues[0]
        self.assertEqual(issue['csv_row'], 3)
        self.assertEqual(issue['first_csv_row'], 2)
        self.assertEqual(issue['date'], '2024-01-02')
        self.assertIn('duplicate Date', issue['reasons'][0])
        self.assertEqual(issue['raw_values'], first)
        self.assertEqual(issue['parsed_values']['Close'], 105.0)

    def test_missing_each_required_field_is_explicit(self):
        for name in FIELDS:
            with self.subTest(name=name):
                row = bar()
                del row[name]
                issue = self.rejection([row]).issues[0]
                self.assertTrue(any(f'missing fields: {name}' in value for value in issue['reasons']))
                self.assertNotIn(name, issue['raw_values'])

    def test_invalid_dates_are_not_silently_truncated_or_guessed(self):
        for value in (None, '', '2024-02-30', '20240102', '02/01/2024',
                      '2024-01-02T00:00:00', '2024-01-02junk', 1704153600):
            with self.subTest(value=value):
                issue = self.rejection([bar(value)]).issues[0]
                self.assertTrue(any('Date must be a valid ISO date' in r for r in issue['reasons']))

    def test_nonnumeric_boolean_nonfinite_and_nonpositive_prices_fail(self):
        values = ('not-a-number', '', None, True, math.nan, math.inf, -math.inf, 0, -1)
        for name in ('Open', 'High', 'Low', 'Close'):
            for value in values:
                with self.subTest(name=name, value=value):
                    issue = self.rejection([bar(**{name: value})]).issues[0]
                    self.assertTrue(any(r.startswith(name + ' must be') for r in issue['reasons']))

    def test_impossible_price_order_is_rejected_without_repair(self):
        for changed, reason in (({'High': 104}, 'High is below max(Open, Close)'),
                                ({'Low': 101}, 'Low exceeds min(Open, Close)'),
                                ({'High': 80, 'Low': 120}, 'Low exceeds High')):
            with self.subTest(changed=changed):
                row = bar(**changed)
                before = dict(row)
                error = self.rejection([row])
                self.assertIn(reason, error.issues[0]['reasons'])
                self.assertEqual(row, before)

    def test_volume_must_be_present_numeric_finite_and_nonnegative(self):
        for value in (-1, math.nan, math.inf, -math.inf, None, '', 'missing', True):
            with self.subTest(value=value):
                error = self.rejection([bar(Volume=value)])
                self.assertTrue(any(r.startswith('Volume must be') for r in error.issues[0]['reasons']))

    def test_reports_every_invalid_row_and_preserves_serializable_raw_evidence(self):
        rows = [bar(), bar('2024-01-03', High=math.inf),
                bar('2024-01-04', Volume=math.nan), bar('bad-date', Close='bad')]
        error = self.rejection(rows)
        self.assertEqual([i['csv_row'] for i in error.issues], [3, 4, 5])
        self.assertEqual(error.issues[0]['raw_values']['High'], 'Infinity')
        self.assertEqual(error.issues[1]['parsed_values']['Volume'], 'NaN')
        self.assertIn('2024-01-03', str(error))
        self.assertIn('CSV row 3', str(error))
        self.assertIn('High must be finite', str(error))
        self.assertIn('Infinity', str(error))
        json.dumps(error.issues, allow_nan=False)

    def test_default_minimum_and_disabled_minimum(self):
        with self.assertRaises(OHLCVValidationError) as caught:
            validate_rows([bar()], 'TEST')
        issue = caught.exception.issues[0]
        self.assertIsNone(issue['csv_row'])
        self.assertIsNone(issue['date'])
        self.assertEqual(issue['raw_values'], {'row_count': 1, 'min_rows': 51})
        self.assertIn('fewer than 51', issue['reasons'][0])
        self.assertEqual(validate_rows([], 'TEST', min_rows=0), [])

    def test_non_mapping_rows_and_noniterable_records_have_structured_errors(self):
        error = self.rejection([None, 'not a row'])
        self.assertEqual([i['csv_row'] for i in error.issues], [2, 3])
        self.assertEqual(error.issues[0]['reasons'], ['row must be a mapping'])
        with self.assertRaises(OHLCVValidationError) as caught:
            validate_rows(None, 'TEST', min_rows=0)
        self.assertIn('iterable', str(caught.exception))


class SourceSelectionTests(unittest.TestCase):
    def test_invalid_primary_falls_back_to_valid_secondary_with_raw_evidence(self):
        invalid = [bar(High=99)]
        valid = [bar()]
        before = copy.deepcopy(invalid + valid)
        rejected = []
        candidates = [('primary', iter(invalid)), ('secondary', iter(valid))]
        cleaned, source = select_valid_source(candidates, 'TEST',
                                              lambda *args: rejected.append(args), min_rows=1)
        self.assertEqual(source, 'secondary')
        self.assertEqual(cleaned, valid)
        self.assertEqual(invalid + valid, before)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0][0:2], ('primary', invalid))
        self.assertIsInstance(rejected[0][2], OHLCVValidationError)

    def test_all_invalid_returns_no_data_and_never_combines_sources(self):
        rejected = []
        result = select_valid_source([('one', [bar(Volume=-1)]), ('two', [bar(High=99)])],
                                     'TEST', lambda *args: rejected.append(args), min_rows=1)
        self.assertEqual(result, (None, None))
        self.assertEqual([entry[0] for entry in rejected], ['one', 'two'])

    def test_does_not_request_more_candidates_after_first_valid_source(self):
        def candidates():
            yield 'good', [bar()]
            self.fail('selection continued after a valid provider')
        self.assertEqual(select_valid_source(candidates(), 'TEST', min_rows=1)[1], 'good')

    def test_rejection_callback_errors_propagate_instead_of_losing_diagnostics(self):
        def fail(*_args):
            raise OSError('diagnostic write failed')
        with self.assertRaisesRegex(OSError, 'diagnostic write failed'):
            select_valid_source([('bad', [bar(High=99)])], 'TEST', fail, min_rows=1)


class SerializationTests(unittest.TestCase):
    def test_float_precision_roundtrips_without_rounding_or_mutation(self):
        rows = [bar(Open=100.00000000000003, High=110.12345678901235,
                    Low=90.00000000000001, Close=105.12345678901235,
                    Volume=123456.123456789)]
        before = copy.deepcopy(rows)
        text = encode_csv(rows)
        self.assertEqual(text.splitlines()[0], ','.join(FIELDS))
        self.assertIn('100.00000000000003', text)
        decoded = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(validate_rows(decoded, 'TEST', min_rows=1), rows)
        self.assertEqual(rows, before)

    def test_atomic_text_and_json_writes_use_safe_json_values(self):
        with temporary_directory() as folder:
            path = Path(folder)/'nested'/'report.json'
            payload = {'positive_inf': math.inf, 'negative_inf': -math.inf, 'nan': math.nan,
                       'time': datetime(2024, 1, 2, 3, 4, 5), 'day': date(2024, 1, 2)}
            write_json_atomic(path, payload)
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(saved, {'positive_inf': 'Infinity', 'negative_inf': '-Infinity',
                                     'nan': 'NaN', 'time': '2024-01-02T03:04:05',
                                     'day': '2024-01-02'})
            write_text_atomic(path, 'replacement\n')
            self.assertEqual(path.read_text(encoding='utf-8'), 'replacement\n')
            self.assertEqual(list(path.parent.glob('*.tmp')), [])

    def test_replace_failure_preserves_previous_file_and_removes_temporary_file(self):
        with temporary_directory() as folder:
            path = Path(folder)/'report.json'
            path.write_text('previous', encoding='utf-8')
            with patch('ohlcv_quality.os.replace', side_effect=OSError('replace failed')):
                with self.assertRaisesRegex(OSError, 'replace failed'):
                    write_json_atomic(path, {'new': 'payload'})
            self.assertEqual(path.read_text(encoding='utf-8'), 'previous')
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_temporary_write_failure_preserves_previous_file(self):
        with temporary_directory() as folder:
            path = Path(folder)/'report.txt'
            path.write_text('previous', encoding='utf-8')
            with patch('ohlcv_quality.os.fsync', side_effect=OSError('flush failed')):
                with self.assertRaisesRegex(OSError, 'flush failed'):
                    write_text_atomic(path, 'replacement')
            self.assertEqual(path.read_text(encoding='utf-8'), 'previous')
            self.assertEqual(list(Path(folder).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
