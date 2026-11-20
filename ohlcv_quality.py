"""Shared, fail-closed OHLCV input checks and diagnostic serialization.

The contract accepts one daily bar per date. It never repairs prices, invents
volume, removes rows, sorts rows, or changes the supplied records in place.
"""

import csv
from collections.abc import Mapping
from datetime import date, datetime
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile


FIELDS = ('Date', 'Open', 'High', 'Low', 'Close', 'Volume')
PRICE_FIELDS = ('Open', 'High', 'Low', 'Close')
_ISO_DATE = re.compile(r'\d{4}-\d{2}-\d{2}\Z')


def _json_safe(value):
    """Keep raw diagnostic values representable in standards-compliant JSON."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return 'NaN'
        if math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


class OHLCVValidationError(ValueError):
    """An invalid dataset, with serializable evidence for every failing row."""

    def __init__(self, symbol, issues):
        self.symbol = symbol
        self.issues = _json_safe(list(issues))
        if self.issues:
            first = self.issues[0]
            stamp = first.get('date')
            location = f' on {stamp}' if stamp is not None else ''
            row = first.get('csv_row')
            if row is not None:
                location += f' (CSV row {row})'
            reasons = '; '.join(str(item) for item in first.get('reasons', []))
            raw = json.dumps(first.get('raw_values', {}), ensure_ascii=True,
                             allow_nan=False, sort_keys=True)
            if len(raw) > 2000:
                raw = raw[:1997] + '...'
            message = f'{symbol}: invalid OHLCV data{location}: {reasons}; values={raw}'
            if len(self.issues) > 1:
                message += f' ({len(self.issues)} issues in total)'
        else:
            message = f'{symbol}: invalid OHLCV data'
        super().__init__(message)


def _issue(csv_row, stamp, reasons, raw_values, parsed_values=None, **extra):
    return {
        'csv_row': csv_row,
        'date': stamp,
        'reasons': list(reasons),
        'raw_values': _json_safe(raw_values),
        'parsed_values': _json_safe(parsed_values or {}),
        **extra,
    }


def _materialize_rows(rows, symbol):
    try:
        return list(rows)
    except TypeError as exc:
        issue = _issue(None, None, ['records must be an iterable of row mappings'],
                       {'records': rows})
        raise OHLCVValidationError(symbol, [issue]) from exc


def _parse_date(value):
    if isinstance(value, datetime):
        stamp = value.date().isoformat()
    elif isinstance(value, date):
        stamp = value.isoformat()
    elif isinstance(value, str):
        stamp = value.strip()
    else:
        raise ValueError('Date must be an ISO date (YYYY-MM-DD)')
    # Date-like missing sentinels can inherit datetime but render as 'NaT'.
    # Apply the same ISO contract to objects and strings before accepting them.
    if not isinstance(stamp, str) or not _ISO_DATE.fullmatch(stamp):
        raise ValueError('Date must be an ISO date (YYYY-MM-DD)')
    return date.fromisoformat(stamp).isoformat()


def validate_rows(rows, symbol, min_rows=51):
    """Return normalized copies, or reject all input with precise row evidence.

    CSV row numbers include the header: the first supplied record is row 2.
    Date strings must be YYYY-MM-DD; date/datetime objects are also accepted.
    Minimum-row checks can be disabled with ``min_rows=0`` for preflight before
    applying a separately declared date cutoff.
    """
    if isinstance(min_rows, bool) or not isinstance(min_rows, int) or min_rows < 0:
        raise ValueError('min_rows must be a nonnegative integer')
    records = _materialize_rows(rows, symbol)
    issues = []
    normalized = []
    first_rows = {}
    for csv_row, row in enumerate(records, start=2):
        if not isinstance(row, Mapping):
            issues.append(_issue(csv_row, None, ['row must be a mapping'],
                                 {'row': row}))
            continue
        reasons = []
        parsed = {}
        raw_stamp = row.get('Date')
        stamp = _json_safe(raw_stamp)
        duplicate_first = None
        missing = [name for name in FIELDS if name not in row]
        if missing:
            reasons.append('missing fields: ' + ', '.join(missing))
        if 'Date' in row:
            try:
                stamp = _parse_date(raw_stamp)
                parsed['Date'] = stamp
                if stamp in first_rows:
                    duplicate_first = first_rows[stamp]
                    reasons.append(f'duplicate Date (first CSV row {duplicate_first})')
                else:
                    first_rows[stamp] = csv_row
            except (TypeError, ValueError, OverflowError):
                reasons.append('Date must be a valid ISO date (YYYY-MM-DD)')
        for name in FIELDS[1:]:
            if name not in row:
                continue
            try:
                if isinstance(row[name], bool):
                    raise ValueError('boolean is not an OHLCV number')
                number = float(row[name])
            except (TypeError, ValueError, OverflowError):
                reasons.append(f'{name} must be numeric')
                continue
            parsed[name] = number
            if not math.isfinite(number):
                reasons.append(f'{name} must be finite')
            elif name in PRICE_FIELDS and number <= 0:
                reasons.append(f'{name} must be positive')
            elif name == 'Volume' and number < 0:
                reasons.append('Volume must be nonnegative')
        if all(name in parsed and math.isfinite(parsed[name]) for name in PRICE_FIELDS):
            if parsed['Low'] > min(parsed['Open'], parsed['Close']):
                reasons.append('Low exceeds min(Open, Close)')
            if parsed['High'] < max(parsed['Open'], parsed['Close']):
                reasons.append('High is below max(Open, Close)')
            if parsed['Low'] > parsed['High']:
                reasons.append('Low exceeds High')
        if reasons:
            extra = {'first_csv_row': duplicate_first} if duplicate_first is not None else {}
            issues.append(_issue(csv_row, stamp, reasons, row, parsed, **extra))
        else:
            normalized.append({name: parsed[name] for name in FIELDS})
    if len(records) < min_rows:
        issues.append(_issue(None, None,
                             [f'fewer than {min_rows} rows (got {len(records)})'],
                             {'row_count': len(records), 'min_rows': min_rows}))
    if issues:
        raise OHLCVValidationError(symbol, issues)
    return normalized


def select_valid_source(candidates, symbol, on_rejected=None, min_rows=51):
    """Choose the first wholly valid source and expose each rejected payload.

    Providers should yield ``(source_name, records)``. No row from a rejected
    candidate is retained or merged with another provider's data.
    """
    for source, records in candidates:
        raw_records = records
        try:
            raw_records = _materialize_rows(records, symbol)
            cleaned = validate_rows(raw_records, symbol, min_rows=min_rows)
        except OHLCVValidationError as error:
            if on_rejected is not None:
                on_rejected(source, raw_records, error)
            continue
        return cleaned, source
    return None, None


def encode_csv(rows):
    """Encode validated bars using Python's round-trip float representation."""
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        encoded = {'Date': row['Date']}
        encoded.update({name: repr(float(row[name])) for name in FIELDS[1:]})
        writer.writerow(encoded)
    return stream.getvalue()


def write_text_atomic(path, text):
    """Replace a text file only after a complete same-directory temporary write."""
    if not isinstance(text, str):
        raise TypeError('text must be a string')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp',
                                              dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def write_json_atomic(path, obj):
    """Atomically preserve diagnostic JSON, including nonfinite raw evidence."""
    text = json.dumps(_json_safe(obj), ensure_ascii=False, allow_nan=False,
                      indent=2) + '\n'
    write_text_atomic(path, text)
