"""Compare entry modes and profit stress scenarios on the same OHLCV snapshot.

No model fitting, Telegram, GitHub writes, network fetch or automatic tuning.
"""
import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys

from ohlcv_quality import OHLCVValidationError, validate_rows, write_json_atomic

ROOT = Path(__file__).resolve().parent
BACKTEST_START = '2022-01-01'
MODES = ('baseline', 'balanced', 'profit_guard')
STRESS_CASES = {
    'higher_slippage': {'additional_slippage': 0.005},
    'exclude_vic_pdr': {'excluded_entries': ['VIC', 'PDR']},
}


def validate_market_coverage(rows, symbol='VNINDEX', start_date=BACKTEST_START):
    """Require enough market history for the replay's fixed 200-session warmup."""
    if symbol != 'VNINDEX':
        return
    before = sum(row['Date'] < start_date for row in rows)
    in_period = sum(row['Date'] >= start_date for row in rows)
    reasons = []
    if before < 200:
        reasons.append(f'at least 200 market rows strictly before {start_date} are required; found {before}')
    if not in_period:
        reasons.append(f'at least one market row on or after {start_date} and before the cutoff is required')
    if reasons:
        raise OHLCVValidationError(symbol, [{
            'csv_row': None, 'date': None, 'reasons': reasons, 'raw_values': {},
            'parsed_values': {'warmup_rows': before, 'in_period_rows': in_period},
        }])


def validate_input_file(path, symbol, end_date, reportdir):
    """Validate the untouched source before cutoff selection or replay output.

    A failure writes diagnostics only. Every source row is checked first so the
    report retains actual CSV row numbers and cannot hide malformed dated rows
    behind the cutoff. This function never repairs or rewrites source data.
    """
    path = Path(path)
    rows = []
    selected = None
    digest = None
    try:
        try:
            source = path.read_bytes()
            digest = hashlib.sha256(source).hexdigest()
            with io.StringIO(source.decode('utf-8-sig'), newline='') as stream:
                rows = list(csv.DictReader(stream, strict=True))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise OHLCVValidationError(symbol, [{
                'csv_row': None, 'date': None,
                'reasons': [f'cannot read OHLCV CSV: {exc}'],
                'raw_values': {}, 'parsed_values': {},
            }]) from exc
        normalized = validate_rows(rows, symbol, min_rows=0)
        selected = [row for row in normalized if row['Date'] <= end_date]
        selected = validate_rows(selected, symbol, min_rows=51)
        validate_market_coverage(selected, symbol)
        return selected
    except OHLCVValidationError as exc:
        # Counts do not filter validation: malformed dates are still rejected.
        candidates = 0
        unknown_dates = 0
        for row in rows:
            stamp = row.get('Date')
            try:
                if not isinstance(stamp, str) or len(stamp.strip()) != 10:
                    raise ValueError('non-ISO date')
                stamp = date.fromisoformat(stamp.strip()).isoformat()
            except (TypeError, ValueError):
                unknown_dates += 1
                continue
            candidates += stamp <= end_date
        issues = exc.issues
        report = {
            'phase': 'preflight', 'status': 'failed', 'symbol': symbol,
            'source_file': str(path.resolve()), 'sha256': digest,
            'end_date': end_date, 'backtest_start': BACKTEST_START,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'input_row_count': len(rows), 'candidate_row_count': candidates,
            'unparseable_date_count': unknown_dates,
            'error_count': len(issues),
            'invalid_row_count': len({issue['csv_row'] for issue in issues
                                      if issue.get('csv_row') is not None}),
            'reason_count': sum(len(issue.get('reasons', [])) for issue in issues),
            'issues': issues, 'error': str(exc), 'replay_started': False,
        }
        if selected is not None and symbol == 'VNINDEX':
            report['market_coverage'] = {
                'warmup_rows': sum(row['Date'] < BACKTEST_START for row in selected),
                'in_period_rows': sum(row['Date'] >= BACKTEST_START for row in selected),
                'required_warmup_rows': 200,
            }
        write_json_atomic(Path(reportdir)/'preflight_report.json', report)
        raise


def summarize(nav_rows, trades, start=None, end=None, initial=100_000_000):
    nav_rows = sorted(nav_rows, key=lambda r: r['Date'])
    selected = [r for r in nav_rows if (start is None or r['Date'] >= start)
                and (end is None or r['Date'] <= end)]
    if not selected:
        return None
    prior = [r for r in nav_rows if r['Date'] < selected[0]['Date']]
    base = float(prior[-1]['NAV']) if prior else initial
    base_date = prior[-1]['Date'] if prior else selected[0]['Date']
    final = float(selected[-1]['NAV'])
    cash_return = ((1 + 0.04/252)**len(selected) - 1)*100
    portfolio_return = 100*(final/base-1)
    peak = base
    mdd = 0.0
    for row in selected:
        peak = max(peak, float(row['NAV']))
        mdd = min(mdd, float(row['NAV']) / peak - 1)
    closed = [t for t in trades if t['Exit_Date'] >= selected[0]['Date']
              and t['Exit_Date'] <= selected[-1]['Date']]
    pnls = [float(t['Total_PnL_VND']) for t in closed]
    wins = sum(p > 0 for p in pnls)
    n = len(pnls)
    gain = sum(p for p in pnls if p > 0)
    loss = -sum(p for p in pnls if p < 0)
    interval = None
    if n:
        # Wilson interval; descriptive only, no independence assumption proven.
        z = 1.96
        p = wins / n
        denom = 1 + z*z/n
        center = (p + z*z/(2*n))/denom
        half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
        interval = [round(100*(center-half), 2), round(100*(center+half), 2)]
    days = (date.fromisoformat(selected[-1]['Date']) - date.fromisoformat(base_date)).days
    by_ticker = {}
    for t in closed:
        by_ticker[t['Ticker']] = by_ticker.get(t['Ticker'], 0) + float(t['Total_PnL_VND'])
    top = sorted(by_ticker.items(), key=lambda item: item[1], reverse=True)
    return {
        'start': selected[0]['Date'], 'end': selected[-1]['Date'],
        'final_nav': round(final, 2), 'return_pct': round(portfolio_return, 4),
        'cash_benchmark_return_pct': round(cash_return, 4),
        'excess_over_cash_pp': round(portfolio_return - cash_return, 4),
        'cagr_pct': round(100*((final/base)**(365.25/days)-1), 4) if days else None,
        'max_drawdown_pct': round(100*mdd, 4), 'closed_positions': n, 'wins': wins,
        'win_rate_pct': round(100*wins/n, 4) if n else None,
        'win_rate_95pct_descriptive_interval': interval,
        'profit_factor': round(gain/loss, 4) if loss else None,
        'trade_pnl': round(sum(pnls), 2),
        'cash_only_pct': round(100*sum(float(r['NAV']) == float(r['Cash']) for r in selected)/len(selected), 2),
        'top_tickers': top[:3],
        'pnl_excluding_VIC_PDR_attribution_only': round(sum(v for k,v in by_ticker.items() if k not in ('VIC','PDR')), 2),
        'warning': 'Observed historical comparison; neither a forecast nor untouched holdout data.',
    }


def nonnegative_finite(value):
    """Argparse type for a declared minimum gain in percentage points."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('must be a finite, nonnegative number')
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('must be a finite, nonnegative number')
    return number


def evaluate_candidate(candidate, baseline, min_win_rate_gain_pp=0.0):
    """Apply predeclared historical gates. A pass does not promote the mode."""
    minimum_gain = nonnegative_finite(min_win_rate_gain_pp)
    c = candidate['full_period']
    b = baseline['full_period']
    both_pf = c['profit_factor'] is not None and b['profit_factor'] is not None
    both_win = c['win_rate_pct'] is not None and b['win_rate_pct'] is not None
    # Summaries have four decimal places; avoid losing an exact 5 pp gain to
    # binary floating-point subtraction (for example 30.53 - 25.53).
    win_rate_gain = round(c['win_rate_pct'] - b['win_rate_pct'], 10) if both_win else None
    years = [y for y, result in candidate['annual'].items()
             if result and result['start'] <= f'{y}-01-10'
             and result['end'] >= f'{y}-12-25' and baseline['annual'].get(y)]
    cash_beating_years = sum(candidate['annual'][y]['excess_over_cash_pp'] > 0 for y in years)
    baseline_years = sum(baseline['annual'][y]['excess_over_cash_pp'] > 0 for y in years)
    checks = {
        'higher_win_rate': (both_win and win_rate_gain > 0
                            and win_rate_gain >= minimum_gain),
        'at_least_30_closes_and_75pct_of_baseline':
            c['closed_positions'] >= max(30, math.ceil(0.75 * b['closed_positions'])),
        'profit_factor_at_least_baseline_and_above_one':
            both_pf and c['profit_factor'] > 1 and c['profit_factor'] >= b['profit_factor'],
        'net_return_at_least_baseline_and_above_cash':
            c['return_pct'] >= b['return_pct'] and c['excess_over_cash_pp'] > 0,
        'drawdown_no_worse_than_baseline':
            c['max_drawdown_pct'] >= b['max_drawdown_pct'],
        'beat_cash_in_at_least_half_the_full_years_and_match_baseline':
            len(years) >= 3 and cash_beating_years >= max(math.ceil(len(years)/2), baseline_years),
    }
    for case in STRESS_CASES:
        stressed = candidate['stress'][case]
        base_stress = baseline['stress'][case]
        checks[f'{case}_above_cash_and_at_least_baseline'] = (
            stressed['excess_over_cash_pp'] > 0
            and stressed['return_pct'] >= base_stress['return_pct']
        )
    return {'historical_pass': all(checks.values()), 'checks': checks,
            'min_win_rate_gain_pp': minimum_gain, 'win_rate_gain_pp': win_rate_gain,
            'full_years_evaluated': years, 'cash_beating_years': cash_beating_years,
            'baseline_cash_beating_years': baseline_years}


def assess_modes(summaries, reference_mode='balanced', min_win_rate_gain_pp=5.0):
    """Compare every other mode with one incumbent on the same snapshot.

    Existing check names retain ``baseline`` for report/test compatibility;
    their comparator is always the explicitly recorded reference mode.
    """
    minimum_gain = nonnegative_finite(min_win_rate_gain_pp)
    reference = summaries[reference_mode]
    for mode, result in summaries.items():
        result['reference_mode'] = reference_mode
        result['min_win_rate_gain_pp'] = minimum_gain
        result.pop('assessment', None)
        if mode != reference_mode:
            result['assessment'] = evaluate_candidate(result, reference, minimum_gain)
    return summaries


def comparison_summary(summaries, reference_mode, min_win_rate_gain_pp):
    """A compact, reviewable report for stdout and GitHub's step summary."""
    def number(value, places=2):
        return 'n/a' if value is None else f'{value:.{places}f}'

    lines = [
        f'## Fixed-snapshot comparison against {reference_mode}',
        f'Minimum win rate gain: {min_win_rate_gain_pp:g} percentage points; '
        'all existing profit and risk criteria must also pass.',
        '',
        '| Mode | Win rate % | Gain pp | Return % | Profit factor | Max drawdown % | Closes | Criteria |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
    ]
    for mode, result in summaries.items():
        m = result['full_period']
        assessment = result.get('assessment')
        status = ('PASS' if assessment['historical_pass'] else 'FAIL') if assessment else 'REFERENCE'
        gain = assessment['win_rate_gain_pp'] if assessment else 0.0
        lines.append(f'| {mode} | {number(m["win_rate_pct"])} | {number(gain)} | '
                     f'{number(m["return_pct"])} | {number(m["profit_factor"])} | '
                     f'{number(m["max_drawdown_pct"])} | {m["closed_positions"]} | {status} |')
    assessed = [(mode, result['assessment']) for mode, result in summaries.items()
                if 'assessment' in result]
    if assessed:
        labels = {
            'higher_win_rate': 'Win rate gain meets minimum and is positive',
            'at_least_30_closes_and_75pct_of_baseline': 'At least 30 closes and 75% of reference',
            'profit_factor_at_least_baseline_and_above_one': 'Profit factor >= reference and > 1',
            'net_return_at_least_baseline_and_above_cash': 'Net return >= reference and above cash',
            'drawdown_no_worse_than_baseline': 'Drawdown no worse than reference',
            'beat_cash_in_at_least_half_the_full_years_and_match_baseline': 'Full-year cash consistency >= reference',
            'higher_slippage_above_cash_and_at_least_baseline': 'Higher slippage: above cash and >= reference',
            'exclude_vic_pdr_above_cash_and_at_least_baseline': 'Exclude VIC/PDR: above cash and >= reference',
        }
        lines.extend(['', '| Check | ' + ' | '.join(mode for mode, _ in assessed) + ' |',
                      '| --- | ' + ' | '.join('---' for _ in assessed) + ' |'])
        for key in assessed[0][1]['checks']:
            lines.append('| ' + labels.get(key, key) + ' | ' + ' | '.join(
                'PASS' if assessment['checks'][key] else 'FAIL'
                for _, assessment in assessed) + ' |')
        target = next(((mode, assessment) for mode, assessment in assessed
                       if mode == 'profit_guard'), None)
        conclusions = [target] if target else assessed
        for mode, assessment in conclusions:
            status = 'PASS' if assessment['historical_pass'] else 'FAIL'
            lines.extend(['', f'Conclusion: {mode} historical criteria {status} against {reference_mode}.'])
    lines.extend(['', 'Historical comparison only; not a forecast or untouched holdout. '
                  'No automatic promotion: review the artifacts before choosing a mode manually.'])
    return '\n'.join(lines) + '\n'


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--output', type=Path, default=ROOT/'validation')
    parser.add_argument('--end-date', default='2026-11-19')
    parser.add_argument('--allow-missing', default='', help='Explicit comma-separated unavailable historical symbols')
    parser.add_argument('--reference-mode', choices=MODES, default='balanced',
                        help='Incumbent for all profit and risk gates (default: balanced)')
    parser.add_argument('--min-win-rate-gain-pp', type=nonnegative_finite, default=5.0,
                        help='Required positive win rate gain in percentage points (default: 5.0)')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if date.fromisoformat(args.end_date).isoformat() != args.end_date:
            raise ValueError('not canonical ISO date')
    except ValueError:
        parser.error('--end-date must be an ISO date (YYYY-MM-DD)')
    if args.end_date < BACKTEST_START:
        parser.error(f'--end-date must be on or after the fixed backtest start {BACKTEST_START}')
    missing_deps = [p for p in ('pandas','numpy','requests','xgboost','sklearn')
                    if importlib.util.find_spec(p) is None]
    if missing_deps:
        parser.exit(2, 'Missing dependencies: '+', '.join(missing_deps)+
                    '. Install requirements-winrate.txt; no replay results were generated.\n')
    # Freeze every strategy setting used by main(), independent of the shell.
    protocol = {
        'QUANT_OFFLINE':'1', 'VN30_PREFETCH':'0', 'PYTHON_DOTENV_DISABLED':'1',
        'ENTRY_MODE':'baseline', 'RUN_EXPECTED_ENTRY_MODE':'baseline',
        'BACKTEST_START_DATE':BACKTEST_START,
        'ENABLE_DYNAMIC_PERCENTILE_AI':'True', 'ENABLE_ASYMMETRIC_ATR_EXIT':'True',
        'ENABLE_MA20_SLOPE_FILTER':'False', 'ENABLE_ANTI_CHASE_FILTER':'False',
        'MAX_PER_SECTOR':'2', 'MAX_PER_ECOSYSTEM':'2',
        'HARD_STOPLOSS_FLOOR':'-0.085', 'ATR_STOPLOSS_MULTIPLIER':'2.2',
    }
    os.environ.update(protocol)
    from vn30_universe import ALL_HISTORICAL_VN30_TICKERS
    allowed = set(filter(None, (s.strip() for s in args.allow_missing.split(','))))
    expected = ['VNINDEX'] + list(ALL_HISTORICAL_VN30_TICKERS)
    missing = [s for s in expected if not (args.data_dir/f'{s}.csv').is_file()]
    undeclared = set(missing)-allowed
    if undeclared or 'VNINDEX' in missing:
        parser.exit(2, 'Missing OHLCV files: '+', '.join(sorted(undeclared or {'VNINDEX'}))+
                    '. Run download_all_data.py where network is available.\n')
    if allowed-set(expected):
        parser.error('Unknown allowed-missing symbols: '+', '.join(sorted(allowed-set(expected))))
    out = args.output.resolve()
    if out.exists():
        parser.error(f'Output already exists: {out}; choose another --output')
    # Validate before creating any replay artifacts. Failures retain diagnostics.
    inputs = {}
    for symbol in expected:
        if symbol in missing:
            continue
        try:
            inputs[symbol] = validate_input_file(args.data_dir/f'{symbol}.csv', symbol,
                                                args.end_date, out)
        except OHLCVValidationError as exc:
            parser.exit(2, f'OHLCV preflight failed: {exc}\n'
                        f'Diagnostics: {out / "preflight_report.json"}\n'
                        'Source data was not changed; no replay results were generated.\n')
    snapshot = out/'snapshot'
    (snapshot/'data').mkdir(parents=True)
    for symbol, rows in inputs.items():
        with (snapshot/'data'/f'{symbol}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(sorted(rows,key=lambda r:r['Date']))
    (snapshot/'models').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT/'data'/'markets_database10years.csv', snapshot/'data'/'markets_database10years.csv')
    for name in ('quant_bot.py','execution_policy.py','run_audit.py',
                 'vn30_universe.py','validate_winrate.py','ohlcv_quality.py'):
        shutil.copyfile(ROOT/name, snapshot/name)
    for path in (ROOT/'models').glob('xgboost_*.json'):
        shutil.copyfile(path, snapshot/'models'/path.name)
    hashes = {str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in snapshot.rglob('*') if p.is_file()}
    manifest = {'created_utc':datetime.now(timezone.utc).isoformat(), 'end_date':args.end_date,
                'missing_symbols':missing, 'sha256':hashes, 'effective_environment':protocol,
                'python':sys.version,
                'packages':{p:importlib.metadata.version(p) for p in ('pandas','numpy','requests','xgboost','scikit-learn')},
                'modes':MODES, 'stress_cases':STRESS_CASES,
                'reference_mode':args.reference_mode,
                'min_win_rate_gain_pp':args.min_win_rate_gain_pp,
                'status':'preparing'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    # Capture main()'s prepared inputs before it writes reports or sends Telegram.
    original_cwd = Path.cwd()
    sys.path.insert(0,str(snapshot))
    captured = {}
    class InputsReady(BaseException):
        pass
    try:
        os.chdir(snapshot)
        spec = importlib.util.spec_from_file_location('quant_validation',snapshot/'quant_bot.py')
        bot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bot)
        runner = bot.run_ai_quant_system
        def capture(**kwargs):
            captured.update(kwargs)
            raise InputsReady()
        bot.run_ai_quant_system = capture
        try:
            bot.main()
        except InputsReady:
            pass
        if not captured:
            raise RuntimeError('Could not capture replay inputs')
        if set(captured['loaded_tickers']) != set(inputs)-{'VNINDEX'}:
            raise RuntimeError('Loaded stock coverage differs from validated snapshot')
        import pandas as pd
        summaries = {}
        for mode in MODES:
            decisions = []
            kw = dict(captured, entry_mode=mode, decision_log=decisions,
                      enable_ma20_slope_filter=False, enable_anti_chase_filter=False)
            log, nav, final, interest, held, buys, sells, completed = runner(**kw)
            folder = out/mode
            folder.mkdir()
            log.to_csv(folder/'trading_log.csv',index=False)
            nav.to_csv(folder/'daily_nav.csv',index=False)
            pd.DataFrame(completed,columns=['Trade_ID','Ticker','Buy_Date','Exit_Date','Buy_Price',
                         'Exit_Price','Initial_Qty','Total_PnL_VND','Profit(%)','Hold_Days','Reason']).to_csv(folder/'completed_trades.csv',index=False)
            pd.DataFrame(decisions).to_csv(folder/'signal_diagnostics.csv',index=False)
            nav_records = nav.to_dict('records')
            stress_results = {}
            for case, settings in STRESS_CASES.items():
                stress_kw = dict(kw, decision_log=None, **settings)
                stress_log, stress_nav, _, _, stress_held, _, _, stress_trades = runner(**stress_kw)
                case_folder = folder/case
                case_folder.mkdir()
                stress_nav.to_csv(case_folder/'daily_nav.csv', index=False)
                stress_log.to_csv(case_folder/'trading_log.csv', index=False)
                pd.DataFrame(stress_trades, columns=['Trade_ID','Ticker','Buy_Date','Exit_Date',
                    'Buy_Price','Exit_Price','Initial_Qty','Total_PnL_VND','Profit(%)','Hold_Days','Reason']).to_csv(
                        case_folder/'completed_trades.csv', index=False)
                stress_results[case] = summarize(stress_nav.to_dict('records'), stress_trades)
                stress_results[case]['open_positions'] = list(stress_held)
            summaries[mode] = {
                'full_period':summarize(nav_records,completed),
                'annual':{str(y):summarize(nav_records,completed,f'{y}-01-01',f'{y}-12-31') for y in range(2022,2027)},
                'open_positions':list(held), 'final_nav':final, 'bank_interest':interest,
                'stress':stress_results,
            }
        assess_modes(summaries, args.reference_mode, args.min_win_rate_gain_pp)
        comparison = {'reference_mode':args.reference_mode,
                      'min_win_rate_gain_pp':args.min_win_rate_gain_pp,
                      'modes':summaries}
        (out/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding='utf-8')
        report = comparison_summary(summaries, args.reference_mode, args.min_win_rate_gain_pp)
        (out/'comparison.md').write_text(report, encoding='utf-8')
        manifest['status'] = 'complete'
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(report)
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a', encoding='utf-8') as stream:
                stream.write(report)
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = str(exc)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        raise
    finally:
        os.chdir(original_cwd)
        sys.path.pop(0)


if __name__ == '__main__':
    main()
