"""Behavioral checks for causal signals, settlement and unavailable fills."""

import math
import unittest

from execution_policy import (
    get_policy,
    open_rejection,
    opening_sell_price,
    signal_rejection,
    stop_fill_price,
    stop_fraction,
)


class EntryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.quality = get_policy("quality")

    def test_valid_bullish_breakout_and_close_location_boundary(self):
        for close in (106.0, 107.0, 110.0):
            with self.subTest(close=close):
                self.assertIsNone(
                    signal_rejection(self.quality, 101.0, 110.0, 100.0, close)
                )

    def test_wicky_positive_candle_is_rejected(self):
        self.assertEqual(
            signal_rejection(self.quality, 102.0, 110.0, 100.0, 105.0),
            "weak_breakout_close",
        )

    def test_red_candle_is_rejected_despite_strong_close_location(self):
        self.assertEqual(
            signal_rejection(self.quality, 109.0, 110.0, 100.0, 108.0),
            "red_breakout_candle",
        )

    def test_zero_range_candle_has_a_distinct_rejection(self):
        self.assertEqual(
            signal_rejection(self.quality, 100.0, 100.0, 100.0, 100.0),
            "zero_range_signal",
        )

    def test_impossible_or_nonfinite_signal_prices_are_rejected(self):
        bars = (
            (105.0, 104.0, 100.0, 103.0),
            (101.0, 110.0, 102.0, 107.0),
            (101.0, math.inf, 100.0, 107.0),
            (101.0, 110.0, 100.0, math.nan),
            (0.0, 110.0, 100.0, 107.0),
        )
        for bar in bars:
            with self.subTest(bar=bar):
                self.assertEqual(
                    signal_rejection(self.quality, *bar), "invalid_signal_ohlc"
                )

    def test_baseline_and_index_mode_do_not_enable_candle_filter(self):
        for mode in ("baseline", "index_trend"):
            with self.subTest(mode=mode):
                self.assertIsNone(
                    signal_rejection(get_policy(mode), 109.0, 110.0, 100.0, 108.0)
                )
        self.assertTrue(get_policy("index_trend").index_slopes)
        self.assertFalse(get_policy("quality").index_slopes)

    def test_unknown_mode_fails_instead_of_silently_falling_back(self):
        with self.assertRaises(ValueError):
            get_policy("quailty")

    def test_atr_gap_ceiling_accepts_boundary_and_rejects_above_it(self):
        self.assertIsNone(open_rejection(self.quality, 101.5, 100.0, 2.0, 99.0))
        self.assertEqual(
            open_rejection(self.quality, 101.5001, 100.0, 2.0, 99.0),
            "open_gap_too_large",
        )

    def test_percent_gap_cap_applies_when_atr_cap_would_be_looser(self):
        self.assertIsNone(open_rejection(self.quality, 104.0, 100.0, 10.0, 99.0))
        self.assertEqual(
            open_rejection(self.quality, 104.0001, 100.0, 10.0, 99.0),
            "open_gap_too_large",
        )

    def test_open_below_breakout_is_rejected_but_boundary_is_accepted(self):
        self.assertIsNone(open_rejection(self.quality, 99.0, 100.0, 2.0, 99.0))
        self.assertEqual(
            open_rejection(self.quality, 98.99, 100.0, 2.0, 99.0),
            "open_below_breakout",
        )

    def test_missing_open_is_rejected_even_in_baseline(self):
        for value in (None, math.nan, math.inf, 0.0, -1.0):
            for mode in ("baseline", "quality"):
                with self.subTest(value=value, mode=mode):
                    self.assertEqual(
                        open_rejection(get_policy(mode), value, 100.0, 2.0, 99.0),
                        "missing_open",
                    )

    def test_quality_mode_requires_known_signal_references(self):
        for close, atr, breakout in (
            (None, 2.0, 99.0),
            (100.0, math.nan, 99.0),
            (100.0, 0.0, 99.0),
            (100.0, 2.0, math.inf),
        ):
            with self.subTest(close=close, atr=atr, breakout=breakout):
                self.assertEqual(
                    open_rejection(self.quality, 100.0, close, atr, breakout),
                    "missing_signal_reference",
                )
        self.assertIsNone(
            open_rejection(get_policy("baseline"), 120.0, None, None, None)
        )


class StopFractionTests(unittest.TestCase):
    def test_stop_scales_with_atr_and_caps_trigger_distance(self):
        self.assertAlmostEqual(stop_fraction(2.0, 100.0), -0.044)
        self.assertAlmostEqual(stop_fraction(3.0, 100.0), -0.066)
        self.assertAlmostEqual(stop_fraction(10.0, 100.0), -0.085)

    def test_nonfinite_missing_or_nonpositive_inputs_use_fallback(self):
        for atr, close in (
            (None, 100.0),
            (math.nan, 100.0),
            (math.inf, 100.0),
            (-1.0, 100.0),
            (0.0, 100.0),
            (2.0, math.nan),
            (2.0, math.inf),
            (2.0, 0.0),
        ):
            with self.subTest(atr=atr, close=close):
                self.assertAlmostEqual(stop_fraction(atr, close), -0.065)
        self.assertAlmostEqual(stop_fraction(None, 100.0, floor=-0.05), -0.05)

    def test_invalid_risk_configuration_fails(self):
        for floor in (-1.0, 0.0, 0.01, math.nan, math.inf):
            with self.subTest(floor=floor):
                with self.assertRaises(ValueError):
                    stop_fraction(2.0, 100.0, floor=floor)
        for multiple in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(multiple=multiple):
                with self.assertRaises(ValueError):
                    stop_fraction(2.0, 100.0, multiple=multiple)

    def test_frozen_prior_session_stop_is_not_repriced_by_todays_close(self):
        # The caller freezes this threshold from yesterday's ATR/reference.
        stop_price = 100.0 * (1.0 + stop_fraction(2.0, 100.0))
        for todays_close in (95.0, 100.0, 104.0):
            with self.subTest(todays_close=todays_close):
                self.assertAlmostEqual(
                    stop_fill_price(100.0, 105.0, 94.0, todays_close, 100.0, stop_price),
                    95.6,
                )


class ExecutionPolicyTests(unittest.TestCase):
    def test_untouched_stop_does_not_sell(self):
        self.assertIsNone(stop_fill_price(100.0, 103.0, 98.0, 101.0, 100.0, 95.0))

    def test_exact_intraday_touch_sells_at_pre_session_stop(self):
        self.assertEqual(stop_fill_price(100.0, 103.0, 95.0, 101.0, 100.0, 95.0), 95.0)

    def test_gap_down_can_realize_a_loss_below_the_trigger(self):
        # A 95 trigger is not a guaranteed 95 fill when the opening price is 94.
        self.assertEqual(stop_fill_price(94.0, 96.0, 93.0, 94.0, 100.0, 95.0), 94.0)

    def test_settlement_day_recovery_does_not_infer_an_afternoon_stop(self):
        # Daily Low may have happened before the stock was available to sell.
        self.assertIsNone(
            stop_fill_price(100.0, 103.0, 93.0, 100.0, 100.0, 95.0, settlement_day=True)
        )

    def test_settlement_day_close_below_stop_uses_close_not_morning_open(self):
        self.assertEqual(
            stop_fill_price(100.0, 103.0, 93.0, 94.0, 100.0, 95.0, settlement_day=True),
            94.0,
        )
        self.assertEqual(
            stop_fill_price(100.0, 103.0, 93.0, 95.0, 100.0, 95.0, settlement_day=True),
            95.0,
        )

    def test_locked_limit_down_bar_has_no_guaranteed_fill(self):
        for settlement_day in (False, True):
            with self.subTest(settlement_day=settlement_day):
                self.assertIsNone(
                    stop_fill_price(
                        93.0, 93.2, 92.9, 93.0, 100.0, 95.0,
                        settlement_day=settlement_day,
                    )
                )

    def test_opening_order_defers_near_limit_down_without_future_bar(self):
        self.assertIsNone(opening_sell_price(93.2, 100.0))
        self.assertIsNone(opening_sell_price(93.0, 100.0))
        self.assertEqual(opening_sell_price(93.21, 100.0), 93.21)
        self.assertEqual(opening_sell_price(100.0, 100.0), 100.0)

    def test_unlocked_session_after_unavailable_open_uses_conservative_close(self):
        self.assertEqual(stop_fill_price(93.0, 97.0, 92.9, 96.0, 100.0, 95.0), 96.0)

    def test_missing_or_inconsistent_execution_bar_is_not_filled(self):
        bars = (
            (None, 103.0, 94.0, 100.0),
            (100.0, math.nan, 94.0, 100.0),
            (100.0, 103.0, math.inf, 100.0),
            (104.0, 103.0, 94.0, 100.0),
        )
        for bar in bars:
            with self.subTest(bar=bar):
                self.assertIsNone(stop_fill_price(*bar, 100.0, 95.0))
        self.assertIsNone(stop_fill_price(100.0, 103.0, 94.0, 100.0, 100.0, math.nan))

    def test_open_requires_valid_positive_price(self):
        for price in (None, math.nan, math.inf, 0.0, -1.0):
            with self.subTest(price=price):
                self.assertIsNone(opening_sell_price(price, 100.0))


if __name__ == "__main__":
    unittest.main()
