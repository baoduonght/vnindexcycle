"""
=============================================================================
SIÊU CÔNG CỤ TẢI TOÀN BỘ DỮ LIỆU CHỨNG KHOÁN VIỆT NAM (2012 - NAY)
Tích hợp trực tiếp Vnstock Open API (vnstock-oapi v0.1.2) + DNSE + VNDirect + TCBS
Hỗ trợ lấy toàn bộ nến OHLCV chuẩn xác của VNINDEX, VN30 và 66+ mã VN30 lịch sử
=============================================================================
"""

import os
import sys
import time
import datetime
import argparse
import csv
import hashlib
import io
from pathlib import Path
import re
import requests
import pandas as pd
import numpy as np

from ohlcv_quality import (encode_csv, select_valid_source, validate_rows,
                           write_json_atomic, write_text_atomic)

# Nạp biến môi trường từ .env nếu có
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Thử import vnstock_oapi (Vnstock Open API Gen 6)
VNSTOCK_OAPI_AVAILABLE = False
try:
    import vnstock_oapi as oapi
    VNSTOCK_OAPI_AVAILABLE = True
except ImportError:
    VNSTOCK_OAPI_AVAILABLE = False

# Vnstock 6 market client. Sources must be selected explicitly per call.
VNSTOCK_AVAILABLE = False
try:
    from vnstock import Market
    VNSTOCK_AVAILABLE = True
except ImportError:
    VNSTOCK_AVAILABLE = False

# Optional, late fallback independent of the exchange HTTP sources above.
TV_DATAFEED_AVAILABLE = False
try:
    from tvDatafeed import TvDatafeed, Interval
    TV_DATAFEED_AVAILABLE = True
except ImportError:
    TV_DATAFEED_AVAILABLE = False

# Thử import rổ VN30 lịch sử từ vn30_universe
try:
    from vn30_universe import ALL_HISTORICAL_VN30_TICKERS
except ImportError:
    ALL_HISTORICAL_VN30_TICKERS = [
        'ACB', 'BCM', 'BID', 'BVH', 'CII', 'CSM', 'CTD', 'CTG', 'DCM', 'DGC', 
        'DIG', 'DPM', 'DRC', 'DXG', 'EIB', 'FLC', 'FPT', 'GAS', 'GMD', 'GVR', 
        'HAG', 'HCM', 'HDB', 'HHS', 'HNG', 'HPG', 'HSG', 'HVG', 'HVN', 'IJC', 
        'ITA', 'KBC', 'KDC', 'KDH', 'LPB', 'MBB', 'MSB', 'MSN', 'MWG', 'NKG', 
        'NT2', 'NTL', 'NVL', 'OCB', 'OGC', 'PDR', 'PET', 'PGD', 'PLX', 'PNJ', 'POW', 
        'PPC', 'PVD', 'PVF', 'PVT', 'QCG', 'REE', 'ROS', 'SAB', 'SBT', 'SHB', 'SJS', 'SSB', 
        'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VCI', 'VHM', 'VIB', 'VIC', 'VJC', 
        'VND', 'VNM', 'VPB', 'VRE', 'VSH'
    ]

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

START_DATE_STR = "2012-01-01"
START_TS = int(datetime.datetime(2012, 1, 1).timestamp())
NOW_TS = int(datetime.datetime.now().timestamp())
NOW_DATE_STR = datetime.datetime.now().strftime("%Y-%m-%d")

QUALITY_REPORT_PATH = Path('data_quality_report.json')
DATA_QUALITY_REPORT = {
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'requested_range': {'start': START_DATE_STR, 'end': NOW_DATE_STR},
    'strict_ohlcv_validation': True,
    'validation_scope': 'requested_date_window',
    'symbols': {},
}


def build_download_window(start_date, end_date):
    """Return an explicit, inclusive daily window using UTC API timestamps."""
    try:
        start = datetime.date.fromisoformat(start_date)
        end = datetime.date.fromisoformat(end_date)
    except (TypeError, ValueError) as exc:
        raise ValueError('Download dates must be ISO YYYY-MM-DD dates') from exc
    if start.isoformat() != start_date or end.isoformat() != end_date:
        raise ValueError('Download dates must be canonical YYYY-MM-DD dates')
    if start > end:
        raise ValueError('--start-date must not be later than --end-date')
    if end == datetime.date.max:
        raise ValueError('--end-date is outside the supported UTC timestamp range')
    utc = datetime.timezone.utc
    start_time = datetime.datetime.combine(start, datetime.time.min, tzinfo=utc)
    next_day = datetime.datetime.combine(end + datetime.timedelta(days=1),
                                         datetime.time.min, tzinfo=utc)
    return {'start': start.isoformat(), 'end': end.isoformat(),
            'from_ts': int(start_time.timestamp()),
            'to_ts': int(next_day.timestamp()) - 1}


def _filter_source_window(rows, start_date=None, end_date=None):
    """Keep every row in a declared date window, including malformed records.

    Unknown/missing dates remain for strict OHLCV rejection. Only records with
    valid dates known to be outside the requested range are excluded here.
    """
    start = datetime.date.fromisoformat(start_date) if start_date is not None else None
    end = datetime.date.fromisoformat(end_date) if end_date is not None else None
    if start is not None and start.isoformat() != start_date:
        raise ValueError('Window start must be canonical YYYY-MM-DD')
    if end is not None and end.isoformat() != end_date:
        raise ValueError('Window end must be canonical YYYY-MM-DD')
    if start is not None and end is not None and start > end:
        raise ValueError('Window start must not exceed window end')
    records = list(rows)
    selected = []
    counts = {'input_rows': len(records), 'kept_rows': 0,
              'excluded_before_start': 0, 'excluded_after_end': 0,
              'invalid_dates_kept': 0}
    for row in records:
        try:
            raw = row['Date']
            if isinstance(raw, datetime.datetime):
                stamp = raw.date().isoformat()
            elif isinstance(raw, datetime.date):
                stamp = raw.isoformat()
            elif isinstance(raw, str):
                stamp = raw.strip()
            else:
                raise ValueError('Missing or non-ISO date')
            parsed = datetime.date.fromisoformat(stamp)
            if parsed.isoformat() != stamp:
                raise ValueError('Noncanonical date')
        except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
            selected.append(row)
            counts['invalid_dates_kept'] += 1
            continue
        if start is not None and parsed < start:
            counts['excluded_before_start'] += 1
        elif end is not None and parsed > end:
            counts['excluded_after_end'] += 1
        else:
            selected.append(row)
    counts['kept_rows'] = len(selected)
    return selected, counts


def _yahoo_symbol(symbol):
    """Yahoo chart identifiers for the index and listed Vietnamese equities."""
    symbol = str(symbol).strip().upper()
    if not re.fullmatch(r'[A-Z0-9^.-]{1,24}', symbol):
        raise ValueError(f'Invalid ticker for Yahoo chart endpoint: {symbol!r}')
    if symbol == 'VNINDEX':
        return '^VNINDEX.VN'
    if symbol in {'VN30', 'VN100', 'VNXALL', 'HNX', 'HNX30', 'UPCOM'}:
        return f'{symbol}.VN'
    return f'{symbol}.VN'


def _normalize_yahoo_chart(symbol, payload):
    """Convert a Yahoo chart payload into unmodified daily OHLCV records."""
    chart = payload.get('chart', {}) if isinstance(payload, dict) else {}
    error = chart.get('error')
    results = chart.get('result') or []
    if error or not results or not isinstance(results[0], dict):
        raise ValueError(f'Yahoo chart returned no result for {symbol}: {error!r}')
    result = results[0]
    metadata = result.get('meta') or {}
    timestamps = result.get('timestamp')
    indicators = result.get('indicators') or {}
    quotes = indicators.get('quote') or []
    if not isinstance(timestamps, list) or not quotes or not isinstance(quotes[0], dict):
        raise ValueError(f'Yahoo chart missing timestamp or quote arrays for {symbol}')
    quote = quotes[0]
    is_index = str(symbol).strip().upper() in {
        'VNINDEX', 'VN30', 'VN100', 'VNXALL', 'HNX', 'HNX30', 'UPCOM'}
    currency = metadata.get('currency')
    if not is_index and currency != 'VND':
        raise ValueError(f'Yahoo chart currency for {symbol} must be VND; got {currency!r}')
    # The trading engine and historical CSV contract store Vietnamese share
    # prices in thousand VND. Yahoo reports equity OHLC in full VND; index
    # levels remain index points and are never scaled.
    price_scale = 1.0 if is_index else 0.001
    required = ('open', 'high', 'low', 'close', 'volume')
    missing = [name for name in required if not isinstance(quote.get(name), list)]
    if missing:
        raise ValueError(f'Yahoo chart missing OHLCV arrays for {symbol}: {", ".join(missing)}')
    lengths = {len(timestamps), *(len(quote[name]) for name in required)}
    if len(lengths) != 1:
        raise ValueError(f'Yahoo chart OHLCV array lengths differ for {symbol}')
    timezone_name = ((result.get('meta') or {}).get('exchangeTimezoneName') or 'UTC')
    rows = []
    for index, stamp in enumerate(timestamps):
        try:
            utc_stamp = pd.to_datetime(stamp, unit='s', utc=True)
            local_stamp = utc_stamp.tz_convert(timezone_name).date().isoformat()
        except Exception as exc:
            raise ValueError(f'Yahoo chart invalid timestamp at row {index} for {symbol}') from exc
        rows.append({
            'Date': local_stamp,
            'Open': None if quote['open'][index] is None else float(quote['open'][index]) * price_scale,
            'High': None if quote['high'][index] is None else float(quote['high'][index]) * price_scale,
            'Low': None if quote['low'][index] is None else float(quote['low'][index]) * price_scale,
            'Close': None if quote['close'][index] is None else float(quote['close'][index]) * price_scale,
            'Volume': quote['volume'][index],
        })
    return pd.DataFrame(rows, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])


def _normalize_vnstock_history(symbol, history):
    """Normalize Vnstock 6 OHLCV output before strict provider validation."""
    if history is None or history.empty:
        return None
    frame = history.copy()
    columns = {str(column).strip().lower(): column for column in frame.columns}
    date_column = next((columns[name] for name in ('time', 'date', 'datetime')
                        if name in columns), None)
    if date_column is None:
        # Some releases return a named datetime index; never mistake a default
        # integer RangeIndex for a date field.
        index = frame.index
        index_name = str(getattr(index, 'name', '') or '').strip().lower()
        is_datetime_index = pd.api.types.is_datetime64_any_dtype(index.dtype)
        if index_name in ('time', 'date', 'datetime') or is_datetime_index:
            frame = frame.reset_index()
            columns = {str(column).strip().lower(): column for column in frame.columns}
            index_aliases = ('time', 'date', 'datetime', 'index') if is_datetime_index else (
                'time', 'date', 'datetime')
            date_column = next((columns[name] for name in index_aliases
                                if name in columns), None)
    required = {'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'}
    missing = []
    if date_column is None:
        missing.append('time/date index')
    missing.extend(output for output, key in required.items() if key not in columns)
    if missing:
        raise ValueError(f'Vnstock FHSC OHLCV missing fields for {symbol}: {", ".join(missing)}')
    dates = pd.to_datetime(frame[date_column], errors='raise')
    timezone = getattr(getattr(dates, 'dt', None), 'tz', None)
    if timezone is not None:
        dates = dates.dt.tz_convert('Asia/Ho_Chi_Minh').dt.tz_localize(None)
    normalized = {'Date': dates.dt.normalize()}
    for output, key in required.items():
        normalized[output] = frame[columns[key]]
    return pd.DataFrame(normalized, index=frame.index)


def _symbol_report(symbol):
    return DATA_QUALITY_REPORT['symbols'].setdefault(symbol, {'attempts': []})


def _record_fetch_error(symbol, source, error):
    # Record exception type only: SDK errors can contain private request details.
    attempt = {
        'source': source, 'status': 'fetch_error', 'error_type': type(error).__name__,
    }
    response = getattr(error, 'response', None)
    status_code = getattr(response, 'status_code', None)
    if isinstance(status_code, int):
        attempt['http_status'] = status_code
    # Provider shape/unsupported-ticker ValueErrors are safe and actionable;
    # do not serialize arbitrary SDK/network exception strings or full URLs.
    if isinstance(error, ValueError):
        attempt['reason'] = str(error)[:240]
    _symbol_report(symbol)['attempts'].append(attempt)
    write_json_atomic(QUALITY_REPORT_PATH, DATA_QUALITY_REPORT)


def _record_rejected_source(symbol, source, rows, error):
    key = re.sub(r'[^A-Za-z0-9_-]+', '_', source).strip('_')
    raw_path = Path(DATA_DIR)/'_rejected'/f'{symbol}_{key}.json'
    write_json_atomic(raw_path, {'symbol': symbol, 'source': source,
                                'candidate_rows': rows, 'issues': error.issues})
    _symbol_report(symbol)['attempts'].append({
        'source': source, 'status': 'rejected', 'rows': len(rows),
        'issues': error.issues, 'candidate_evidence': str(raw_path),
    })
    write_json_atomic(QUALITY_REPORT_PATH, DATA_QUALITY_REPORT)
    print(f'\n   [REJECT {source}] {error}', flush=True)


def _record_window_crop(symbol, source, rows, counts, start_date, end_date):
    entry = {'source': source, 'status': 'window_filter',
             'start': start_date, 'end': end_date, **counts}
    if counts['excluded_before_start'] or counts['excluded_after_end']:
        key = re.sub(r'[^A-Za-z0-9_-]+', '_', source).strip('_')
        evidence = Path(DATA_DIR)/'_window_input'/f'{symbol}_{key}.json'
        write_json_atomic(evidence, {'symbol': symbol, 'source': source,
                                    'requested_start': start_date,
                                    'requested_end': end_date,
                                    'reason': 'Outside explicitly requested download window',
                                    'original_candidate_rows': rows, 'counts': counts})
        entry['original_candidate_evidence'] = str(evidence)
    _symbol_report(symbol)['attempts'].append(entry)
    write_json_atomic(QUALITY_REPORT_PATH, DATA_QUALITY_REPORT)

def _iter_ticker_sources(symbol):
    """
    Tải nến OHLCV thật từ hệ thống màng lưới đa API:
    1. vnstock-oapi (Nếu đã cài đặt và cấu hình khoá DNSE/SSI/TCBS)
    2. DNSE Entrade API Trực tiếp (Siêu tốc độ, không cần key)
    3. VNDirect DChart API
    4. TCBS Public API
    5. Vnstock Market API
    """
    is_index = symbol.upper() in ['VNINDEX', 'VN30', 'VN100', 'VNXALL', 'HNX', 'HNX30', 'UPCOM']
    entrade_endpoint = 'index' if is_index else 'stock'

    # --- NGUỒN 1: DNSE ENTRADE API TRỰC TIẾP (Siêu tốc độ, 100% nến thật, KHÔNG CẦN KHOÁ) ---
    try:
        url = f"https://services.entrade.com.vn/chart-api/v2/ohlcs/{entrade_endpoint}?from={START_TS}&to={NOW_TS}&symbol={symbol}&resolution=1D"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}, timeout=8)
        if res.status_code == 200:
            data = res.json()
            if 't' in data and len(data['t']) > 50:
                df = pd.DataFrame({
                    'Date': pd.to_datetime(data['t'], unit='s').tz_localize(None).normalize(),
                    'Open': data['o'],
                    'High': data['h'],
                    'Low': data['l'],
                    'Close': data['c'],
                    'Volume': data['v']
                })
                df = df.sort_values('Date').reset_index(drop=True)
                yield df, "DNSE Entrade (Không cần key)"
    except Exception as exc:
        _record_fetch_error(symbol, 'DNSE Entrade', exc)

    # --- NGUỒN 2: VNDIRECT DCHART API (Dự phòng 1) ---
    try:
        url = f"https://dchart-api.vndirect.com.vn/dchart/history?resolution=D&symbol={symbol}&from={START_TS}&to={NOW_TS}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}, timeout=8)
        if res.status_code == 200:
            data = res.json()
            if data.get('s') == 'ok' and len(data.get('t', [])) > 50:
                df = pd.DataFrame({
                    'Date': pd.to_datetime(data['t'], unit='s').tz_localize(None).normalize(),
                    'Open': data['o'],
                    'High': data['h'],
                    'Low': data['l'],
                    'Close': data['c'],
                    'Volume': data['v']
                })
                df = df.sort_values('Date').reset_index(drop=True)
                yield df, "VNDirect"
    except Exception as exc:
        _record_fetch_error(symbol, 'VNDirect', exc)

    # --- NGUỒN 3: TCBS PUBLIC API (Dự phòng 2) ---
    try:
        api_type = 'index' if is_index else 'stock'
        url = f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker={symbol}&type={api_type}&resolution=D&from={START_TS}&to={NOW_TS}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}, timeout=8)
        if res.status_code == 200:
            data = res.json().get('data', [])
            if len(data) > 50:
                df = pd.DataFrame(data)
                df.rename(columns={'tradingDate': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].sort_values('Date').reset_index(drop=True)
                yield df, "TCBS"
    except Exception as exc:
        _record_fetch_error(symbol, 'TCBS', exc)

    # --- NGUỒN 4: VNSTOCK-OAPI (Dự phòng nếu có khoá trong .env) ---
    if VNSTOCK_OAPI_AVAILABLE:
        for src_name in ['dnse', 'ssi', 'tcbs']:
            try:
                src = oapi.source(src_name)
                envelopes = src.ohlc(symbol, start=START_DATE_STR, end=NOW_DATE_STR, interval="1D")
                if envelopes and len(envelopes) > 0:
                    records = []
                    for env in envelopes:
                        body = getattr(env, 'body', env)
                        if isinstance(body, list):
                            records.extend(body)
                        elif isinstance(body, dict):
                            data_items = body.get('data') or body.get('bars') or body.get('t') or [body]
                            if isinstance(data_items, list):
                                records.extend(data_items)
                    
                    if records and len(records) > 50:
                        df_o = pd.DataFrame(records)
                        col_remap = {
                            'tradingDate': 'Date', 'time': 'Date', 'date': 'Date', 't': 'Date',
                            'open': 'Open', 'o': 'Open', 'high': 'High', 'h': 'High',
                            'low': 'Low', 'l': 'Low', 'close': 'Close', 'c': 'Close',
                            'volume': 'Volume', 'v': 'Volume'
                        }
                        for k, v in col_remap.items():
                            if k in df_o.columns and v not in df_o.columns:
                                df_o.rename(columns={k: v}, inplace=True)
                        
                        if 'Date' in df_o.columns and 'Close' in df_o.columns:
                            df_o['Date'] = pd.to_datetime(df_o['Date']).dt.tz_localize(None).dt.normalize()
                            df_o = df_o[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].sort_values('Date').reset_index(drop=True)
                            yield df_o, f"vnstock-oapi ({src_name.upper()})"
            except Exception as exc:
                _record_fetch_error(symbol, f'vnstock-oapi ({src_name.upper()})', exc)

    # --- NGUỒN 5: Vnstock 6 + FHSC Open API (cần FHSC_API_KEY) ---
    if VNSTOCK_AVAILABLE and os.environ.get('FHSC_API_KEY', '').strip():
        try:
            market = Market('fhsc')
            if symbol.upper() in ('VNINDEX', 'VN30', 'VN100', 'VNXALL', 'HNX', 'HNX30', 'UPCOM'):
                history = market.index(symbol).ohlcv(
                    start=START_DATE_STR, end=NOW_DATE_STR, interval='1D',
                    compat='v6', floating=None)
            else:
                history = market.equity(symbol).ohlcv(
                    start=START_DATE_STR, end=NOW_DATE_STR, interval='1D',
                    compat='v6', floating=None)
            # Vnstock compat=v6 already returns equity candles in thousand VND,
            # index values in points, and Vietnamese market timestamps.
            frame = _normalize_vnstock_history(symbol, history)
            if frame is not None and len(frame) > 50:
                yield frame.sort_values('Date').reset_index(drop=True), 'Vnstock Open API (FHSC)'
        except Exception as exc:
            _record_fetch_error(symbol, 'Vnstock Open API (FHSC)', exc)

    # --- NGUỒN 6: Yahoo Finance chart API (độc lập với các nguồn VN ở trên) ---
    # Chỉ thử sau khi các nguồn khác không cho ra tập OHLCV đạt chuẩn. Mọi bar
    # vẫn qua cùng bộ strict validator; Yahoo không được phép bỏ volume/null.
    try:
        yahoo_ticker = _yahoo_symbol(symbol)
        yahoo_url = f'https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_ticker}'
        params = {'period1': START_TS, 'period2': NOW_TS + 1, 'interval': '1d',
                  'events': 'div,splits', 'includeAdjustedClose': 'false'}
        response = requests.get(
            yahoo_url, params=params,
            headers={'User-Agent': 'Mozilla/5.0'}, timeout=12)
        response.raise_for_status()
        yahoo_frame = _normalize_yahoo_chart(symbol, response.json())
        if not yahoo_frame.empty:
            unit = 'index points' if symbol.upper() in {'VNINDEX','VN30','VN100','VNXALL','HNX','HNX30','UPCOM'} else 'VND→thousand VND'
            yield yahoo_frame.sort_values('Date').reset_index(drop=True), f'Yahoo Finance ({yahoo_ticker}; {unit})'
    except Exception as exc:
        _record_fetch_error(symbol, 'Yahoo Finance chart', exc)

    # --- NGUỒN 7: TradingView guest chart (last resort; same strict validator) ---
    if TV_DATAFEED_AVAILABLE:
        try:
            tv = TvDatafeed()
            frame = tv.get_hist(symbol=symbol, exchange='HOSE',
                                interval=Interval.in_daily, n_bars=10000)
            if frame is not None and not frame.empty:
                frame = frame.reset_index()
                frame.rename(columns={'datetime':'Date','open':'Open','high':'High',
                                      'low':'Low','close':'Close','volume':'Volume'},
                             inplace=True)
                frame['Date'] = pd.to_datetime(frame['Date']).dt.tz_localize(None).dt.normalize()
                frame = frame[['Date','Open','High','Low','Close','Volume']].sort_values('Date').reset_index(drop=True)
                yield frame, f'TradingView ({symbol}; exchange prices)'
        except Exception as exc:
            _record_fetch_error(symbol, 'TradingView guest chart', exc)


def fetch_ticker_multi_source(symbol, history_start=None, history_end=None):
    """Choose the first complete provider dataset passing the replay contract."""
    def candidates():
        for frame, source in _iter_ticker_sources(symbol):
            rows = frame.to_dict('records')
            if history_start is not None or history_end is not None:
                selected, counts = _filter_source_window(rows, history_start, history_end)
                _record_window_crop(symbol, source, rows, counts, history_start, history_end)
                rows = selected
            yield source, rows
    rows, source = select_valid_source(
        candidates(), symbol, min_rows=51,
        on_rejected=lambda source, rows, error:
            _record_rejected_source(symbol, source, rows, error))
    report = _symbol_report(symbol)
    if rows is None:
        report['status'] = 'failed_no_valid_source'
        write_json_atomic(QUALITY_REPORT_PATH, DATA_QUALITY_REPORT)
        return None, None
    report.update(status='validated', source=source, rows=len(rows))
    write_json_atomic(QUALITY_REPORT_PATH, DATA_QUALITY_REPORT)
    frame = pd.DataFrame(rows)
    frame['Date'] = pd.to_datetime(frame['Date'])
    return frame, source

def main(argv=None):
    global START_DATE_STR, START_TS, NOW_DATE_STR, NOW_TS
    parser = argparse.ArgumentParser(description='Download validated daily OHLCV for an explicit UTC date window.')
    parser.add_argument('--start-date', default='2012-01-01',
                        help='Inclusive history start; comparison uses 2021 warmup for a 2022 replay.')
    parser.add_argument('--end-date', default=datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
                        help='Inclusive history end, YYYY-MM-DD (default: current UTC date).')
    args = parser.parse_args(argv)
    try:
        window = build_download_window(args.start_date, args.end_date)
    except (ValueError, OverflowError, OSError) as exc:
        parser.error(str(exc))
    START_DATE_STR, NOW_DATE_STR = window['start'], window['end']
    START_TS, NOW_TS = window['from_ts'], window['to_ts']
    DATA_QUALITY_REPORT['requested_range'] = {**window, 'timestamp_timezone': 'UTC'}
    write_json_atomic(QUALITY_REPORT_PATH, DATA_QUALITY_REPORT)
    print("=" * 75)
    print("🚀 BỘ TẢI DỮ LIỆU ĐỊNH LƯỢNG CHỨNG KHOÁN VIỆT NAM (2012 - NAY)")
    print(f"📅 Khung thời gian: Từ {START_DATE_STR} đến {NOW_DATE_STR}")
    print(f"📂 Thư mục lưu trữ: ./{DATA_DIR}/")
    print(f"🔑 vnstock-oapi: {'✅ Đã kết nối' if VNSTOCK_OAPI_AVAILABLE else 'ℹ️ Chưa cài (Tự động dùng DNSE Direct, VNDirect, TCBS)'}")
    print(f"📈 vnstock Gen 6: {'✅ Sẵn sàng' if VNSTOCK_AVAILABLE else 'ℹ️ Dùng Multi-API Trực tiếp'}")
    print("=" * 75 + "\n")

    target_symbols = ['VNINDEX', 'VN30'] + sorted(list(set(ALL_HISTORICAL_VN30_TICKERS)))
    total_targets = len(target_symbols)

    success_count = 0
    failed_symbols = []
    all_master_rows = []

    start_time = time.time()

    for idx, sym in enumerate(target_symbols, 1):
        print(f"[{idx:02d}/{total_targets:02d}] Đang tải {sym:<8} ... ", end="", flush=True)
        
        df, source = fetch_ticker_multi_source(
            sym, history_start=START_DATE_STR, history_end=NOW_DATE_STR)
        if df is not None and not df.empty:
            file_path = os.path.join(DATA_DIR, f"{sym}.csv")
            rows = validate_rows(df.to_dict('records'), sym, min_rows=51)
            serialized = encode_csv(rows)
            # Validate the exact CSV bytes' numeric round trip before replacing
            # any previous dataset. Do not round prices or manufacture candles.
            validate_rows(csv.DictReader(io.StringIO(serialized)), sym, min_rows=51)
            write_text_atomic(file_path, serialized)
            _symbol_report(sym).update(
                status='saved', csv_file=file_path,
                csv_sha256=hashlib.sha256(serialized.encode('utf-8')).hexdigest())
            write_json_atomic(QUALITY_REPORT_PATH, DATA_QUALITY_REPORT)

            df_master = df.copy()
            df_master['Ticker'] = sym
            all_master_rows.append(df_master[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']])

            first_date = df['Date'].iloc[0].strftime('%Y-%m-%d')
            last_date = df['Date'].iloc[-1].strftime('%Y-%m-%d')
            n_bars = len(df)
            print(f"✅ {n_bars:>5} phiên ({first_date} -> {last_date}) [{source}]")
            success_count += 1
        else:
            print("❌ Thất bại (Không có dữ liệu hoặc mã đã hủy niêm yết từ lâu)")
            failed_symbols.append(sym)
            if sym == 'VNINDEX':
                DATA_QUALITY_REPORT['status'] = 'failed'
                write_json_atomic(QUALITY_REPORT_PATH, DATA_QUALITY_REPORT)
                raise SystemExit('VNINDEX: no valid provider dataset; see data_quality_report.json '
                                 'and data/_rejected/ for evidence. No replay can run.')

        time.sleep(0.08)

    print("\n" + "=" * 75)
    print("📊 ĐANG TẠO FILE TỔNG HỢP TOÀN BỘ THỊ TRƯỜNG (MASTER CSV)...")
    if all_master_rows:
        master_df = pd.concat(all_master_rows, ignore_index=True)
        master_df = master_df.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
        master_file = os.path.join(DATA_DIR, f"vn_database_all_tickers_{START_DATE_STR[:4]}_{NOW_DATE_STR[:4]}.csv")
        write_text_atomic(master_file, master_df.to_csv(index=False, float_format='%.17g'))
        print(f"✅ Đã xuất File Master: {master_file} ({len(master_df):,} dòng dữ liệu)")

    elapsed = time.time() - start_time
    print("=" * 75)
    print(f"🎉 HOÀN THÀNH TRONG {elapsed:.1f} GIÂY!")
    print(f"   • Tải thành công: {success_count}/{total_targets} mã")
    if failed_symbols:
        print(f"   • Mã không tải được: {', '.join(failed_symbols)}")
    print(f"   • Thư mục chứa file CSV từng mã: {os.path.abspath(DATA_DIR)}")
    DATA_QUALITY_REPORT.update(status='download_complete', failed_symbols=failed_symbols)
    write_json_atomic(QUALITY_REPORT_PATH, DATA_QUALITY_REPORT)
    print("=" * 75)

if __name__ == "__main__":
    main()
