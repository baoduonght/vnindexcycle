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


POLICIES = {
    "baseline": EntryPolicy("baseline"),
    "quality": EntryPolicy("quality", candle_filter=True),
    "balanced": EntryPolicy("balanced", candle_filter=True, risk_per_trade=0.015),
    "index_trend": EntryPolicy("index_trend", index_slopes=True),
}


def get_policy(name):
    if name not in POLICIES:
        raise ValueError(f"Unknown ENTRY_MODE {name!r}; choose {', '.join(POLICIES)}")
    return POLICIES[name]


def valid_prices(*values):
    try:
        return all(math.isfinite(float(v)) and float(v) > 0 for v in values)
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
