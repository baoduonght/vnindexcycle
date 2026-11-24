"""Causal entry and execution rules. No model or numerical dependencies."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EntryPolicy:
    name: str
    candle_filter: bool = False
    index_slopes: bool = False
    min_close_location: float = 0.60
    max_open_gap_atr: float = 0.75
    max_open_gap_pct: float = 0.04
    risk_per_trade: float | None = None
    profit_guard: bool = False
    weekly_regime: bool = False
    crash_guard: bool = False
    max_positions: int = 5
    max_per_sector: int = 2
    fast_trail_trigger: float = 0.25
    fast_trail_confirm_days: int = 2
    pyramid_trigger: float = 0.08
    allow_reentry: bool = True


POLICIES = {
    "baseline": EntryPolicy("baseline"),
    "quality": EntryPolicy("quality", candle_filter=True),
    "balanced": EntryPolicy("balanced", candle_filter=True, risk_per_trade=0.015),
    "profit_guard": EntryPolicy("profit_guard", candle_filter=True, risk_per_trade=0.015,
                                profit_guard=True),
    "index_trend": EntryPolicy("index_trend", index_slopes=True),
    "uptrend_regime": EntryPolicy(
        "uptrend_regime",
        candle_filter=True,
        index_slopes=True,
        risk_per_trade=0.015,
        profit_guard=True,
        weekly_regime=True,
        crash_guard=True,
        max_positions=5,
        max_per_sector=2,
        fast_trail_trigger=0.25,
        fast_trail_confirm_days=2,
        pyramid_trigger=0.08,
        allow_reentry=False,
    ),
}


def get_policy(name):
    if name not in POLICIES:
        raise ValueError(f"Unknown ENTRY_MODE {name!r}; choose {', '.join(POLICIES)}")
    return POLICIES[name]


def _is_finite(*values):
    try:
        return all(v is not None and math.isfinite(float(v)) for v in values)
    except (ValueError, TypeError):
        return False


def valid_prices(*values):
    try:
        return all(v is not None and math.isfinite(float(v)) and float(v) > 0 for v in values)
    except (ValueError, TypeError):
        return False


def valid_bar(open_price, high, low, close):
    return (valid_prices(open_price, high, low, close)
            and low <= min(open_price, close) <= max(open_price, close) <= high)


def signal_rejection(policy, open_price, high, low, close):
    if not policy.candle_filter:
        return None
    if not valid_bar(open_price, high, low, close):
        return "invalid_signal_ohlc"
    if high <= low:
        return "zero_range_signal"
    if close < open_price:
        return "red_breakout_candle"
    if (close - low) / (high - low) < policy.min_close_location:
        return "weak_breakout_close"
    return None


def open_rejection(policy, open_price, signal_close, signal_atr, breakout_level):
    if not valid_prices(open_price):
        return "missing_open"
    if not policy.candle_filter:
        return None
    if not valid_prices(signal_close, signal_atr, breakout_level):
        return "missing_signal_reference"
    if open_price < breakout_level:
        return "open_below_breakout"
    ceiling = min(signal_close + policy.max_open_gap_atr * signal_atr,
                  signal_close * (1.0 + policy.max_open_gap_pct))
    if open_price > ceiling:
        return "open_gap_too_large"
    return None


def stop_fraction(atr, reference_close, floor=-0.085, multiple=2.2):
    if not (-1.0 < floor < 0) or not math.isfinite(multiple) or multiple <= 0:
        raise ValueError("Stop floor must be between -1 and 0; multiplier must be positive")
    if not valid_prices(atr, reference_close):
        return max(floor, -0.065)
    return max(floor, -multiple * atr / reference_close)


def profit_floor_price(buy_price, known_high, previous_floor=None,
                       trigger=0.10, floor=0.02,
                       super_trigger=0.20, super_floor=0.05):
    """Latch existing gross profit floors using only a prior-session peak.

    This is an activation level, not a guaranteed fill or net return. The
    caller must freeze it before observing today's OHLC and respect settlement,
    gaps and unavailable liquidity. No fixed take-profit caps an upward trend.
    """
    try:
        valid_configuration = (
            all(math.isfinite(float(v)) for v in (trigger, floor, super_trigger, super_floor))
            and 0 <= floor < trigger < super_trigger
            and floor <= super_floor < super_trigger
        )
    except (ValueError, TypeError):
        valid_configuration = False
    if not valid_configuration:
        raise ValueError('Profit guard requires finite 0 <= floor < trigger < super_trigger '
                         'and floor <= super_floor < super_trigger')
    armed_floor = float(previous_floor) if valid_prices(previous_floor) else None
    if not valid_prices(buy_price, known_high):
        return armed_floor
    buy_price, known_high = float(buy_price), float(known_high)

    def reached(level):
        return known_high >= level or math.isclose(known_high, level, rel_tol=1e-12, abs_tol=1e-12)

    if reached(buy_price * (1 + super_trigger)):
        candidate = buy_price * (1 + super_floor)
    elif reached(buy_price * (1 + trigger)):
        candidate = buy_price * (1 + floor)
    else:
        return armed_floor
    return max(candidate, armed_floor) if armed_floor is not None else candidate


def position_budget(policy, nav, slot, stop, buy_friction, sell_friction):
    """Limit modeled loss at the trigger, including both sides' costs."""
    if policy.risk_per_trade is None:
        return slot
    loss_fraction = 1 - (1 + stop) * (1 - sell_friction) / (1 + buy_friction)
    return min(slot, nav * policy.risk_per_trade / loss_fraction)


def opening_sell_price(open_price, reference_close):
    """Defer an opening order near a daily limit down; no future bar consulted.

    6.8% is a conservative proxy, not an exchange tick-price calculation.
    Missing/adjusted references need an actual reference-price data source.
    """
    if not valid_prices(open_price):
        return None
    if valid_prices(reference_close) and open_price <= reference_close * 0.932:
        return None
    return open_price


def stop_fill_price(open_price, high, low, close, reference_close, stop_price,
                    settlement_day=False):
    """Use a pre-session stop. T+2 has only a conservative close-time exit.

    A stop touched only in the morning of settlement day cannot be inferred
    from daily OHLC. Locked limit-down bars remain held; never force a fill.
    """
    if not valid_bar(open_price, high, low, close) or not valid_prices(stop_price):
        return None
    if valid_prices(reference_close) and high <= reference_close * 0.932:
        return None
    if settlement_day:
        return close if close <= stop_price else None
    if low > stop_price:
        return None
    if open_price <= stop_price:
        if opening_sell_price(open_price, reference_close) is None:
            # Opening sale was unavailable; conservative liquidation at close
            # only if the session traded away from its limit-down region.
            return close
        return open_price
    return stop_price


# --- CAUSAL UPTREND REGIME & PORTFOLIO GUARD RULES ---

def vnindex_regime_ok(vni_close, prev_week_close, prev_ma20_w, prev_ma20_w_slope):
    """Allow new entries only when VNINDEX and closed week sit above rising MA20_W."""
    if not valid_prices(vni_close, prev_week_close, prev_ma20_w):
        return False
    if not _is_finite(prev_ma20_w_slope):
        return False
    return (
        float(vni_close) > float(prev_ma20_w)
        and float(prev_week_close) > float(prev_ma20_w)
        and float(prev_ma20_w_slope) > 0.0
    )


def vnindex_crash_guard_exit(vni_close, vni_prev_week_close, vni_prev_ma20_w,
                             stock_close, stock_ma20_d):
    """Protect capital when VNINDEX breaks below weekly MA20 and stock loses daily MA20."""
    if not valid_prices(vni_close, vni_prev_week_close, vni_prev_ma20_w,
                        stock_close, stock_ma20_d):
        return False
    vni_broken = (
        float(vni_prev_week_close) < float(vni_prev_ma20_w)
        and float(vni_close) < float(vni_prev_ma20_w)
    )
    return vni_broken and float(stock_close) < float(stock_ma20_d)


def stock_weekly_regime_ok(stock_close, prev_week_close, prev_ma20_w, prev_ma20_w_slope,
                           stock_ret_63d=None, vni_ret_63d=None):
    """Confirm stock weekly uptrend and optional 63-day relative strength vs VNINDEX."""
    if not vnindex_regime_ok(stock_close, prev_week_close, prev_ma20_w, prev_ma20_w_slope):
        return False
    if stock_ret_63d is not None and vni_ret_63d is not None:
        if not _is_finite(stock_ret_63d, vni_ret_63d):
            return False
        if float(stock_ret_63d) <= float(vni_ret_63d):
            return False
    return True


def sector_cap_rejection(ticker, sector_map, open_tickers,
                         max_positions=5, max_per_sector=2):
    """Enforce portfolio slot limit and maximum exposure per sector."""
    held = list(open_tickers)
    if ticker in held:
        return "already_held"
    if len(held) >= max_positions:
        return "max_positions_reached"
    target_sector = sector_map.get(ticker, "OTHER")
    same_sector_count = sum(1 for t in held if sector_map.get(t, "OTHER") == target_sector)
    if same_sector_count >= max_per_sector:
        return f"sector_cap_{target_sector}"
    return None


def fast_trail_ma20_exit(close, ma20_d, max_pnl_pct, below_ma20_days,
                         trigger=0.25, confirm_days=2):
    """Trail strong winners (>=25% peak gain) using consecutive closes below MA20_D."""
    if not valid_prices(close, ma20_d) or not _is_finite(max_pnl_pct):
        return False, int(below_ma20_days)
    if float(max_pnl_pct) < float(trigger):
        return False, 0
    updated_days = int(below_ma20_days) + 1 if float(close) < float(ma20_d) else 0
    return updated_days >= int(confirm_days), updated_days


def should_pyramid_tranche(tranches, is_market_uptrend, pnl_pct, close, high_10d,
                           trigger=0.08):
    """Allow second 50% tranche only when tranche 1 is profitable and breaks 10D high."""
    if int(tranches) != 1 or not is_market_uptrend:
        return False
    if not _is_finite(pnl_pct) or not valid_prices(close, high_10d):
        return False
    return float(pnl_pct) >= float(trigger) and float(close) > float(high_10d)