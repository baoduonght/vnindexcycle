"""Dependency-free checks for causal, ratcheted profit-protection stops.

A profit floor is a pre-session trigger, not a guaranteed execution price.
The replay wiring is inspected without importing its optional data/model stack.
"""

import ast
from dataclasses import fields
import math
from pathlib import Path
import unittest

from execution_policy import (
    EntryPolicy,
    POLICIES,
    get_policy,
    profit_floor_price,
    stop_fill_price,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "tests" else HERE
INVALID_PRICES = (None, 0.0, -1.0, math.nan, math.inf, -math.inf, "missing")


class ProfitGuardPolicyTests(unittest.TestCase):
    def test_profit_guard_is_opt_in_for_existing_and_custom_policies(self):
        self.assertFalse(EntryPolicy("custom").profit_guard)
        for name, policy in POLICIES.items():
            with self.subTest(mode=name):
                self.assertEqual(policy.profit_guard, name == "profit_guard")

    def test_profit_guard_keeps_balanced_entry_and_risk_settings(self):
        balanced = get_policy("balanced")
        guarded = get_policy("profit_guard")
        self.assertTrue(guarded.profit_guard)
        self.assertEqual(guarded.name, "profit_guard")
        for field in fields(EntryPolicy):
            if field.name not in {"name", "profit_guard"}:
                with self.subTest(setting=field.name):
                    self.assertEqual(getattr(guarded, field.name),
                                     getattr(balanced, field.name))


class ProfitFloorPriceTests(unittest.TestCase):
    def test_below_first_trigger_remains_unarmed(self):
        for known_high in (50.0, 100.0, 109.0, 109.9999):
            with self.subTest(known_high=known_high):
                self.assertIsNone(profit_floor_price(100.0, known_high))

    def test_both_trigger_boundaries_are_inclusive_despite_float_roundoff(self):
        # 100 * 1.10 is slightly greater than the literal 110 in binary floats.
        for high, expected in ((110.0, 102.0), (120.0, 105.0)):
            for adjacent in (math.nextafter(high, -math.inf), high,
                             math.nextafter(high, math.inf)):
                with self.subTest(known_high=adjacent, expected=expected):
                    self.assertAlmostEqual(profit_floor_price(100.0, adjacent),
                                           expected)
        for buy in (1.0, 10.0, 125.45):
            with self.subTest(buy=buy):
                self.assertAlmostEqual(profit_floor_price(buy, buy * 1.10),
                                       buy * 1.02)
                self.assertAlmostEqual(profit_floor_price(buy, buy * 1.20),
                                       buy * 1.05)

    def test_floors_are_buy_price_tiers_and_do_not_trail_the_peak(self):
        for known_high in (110.0, 115.0, 119.9999):
            with self.subTest(known_high=known_high):
                self.assertAlmostEqual(profit_floor_price(100.0, known_high), 102.0)
        for known_high in (120.0, 130.0, 500.0):
            with self.subTest(known_high=known_high):
                self.assertAlmostEqual(profit_floor_price(100.0, known_high), 105.0)

    def test_custom_trigger_and_floor_configuration(self):
        settings = dict(trigger=0.15, floor=0.03,
                        super_trigger=0.30, super_floor=0.07)
        self.assertIsNone(profit_floor_price(200.0, 229.99, **settings))
        self.assertAlmostEqual(profit_floor_price(200.0, 230.0, **settings), 206.0)
        self.assertAlmostEqual(profit_floor_price(200.0, 260.0, **settings), 214.0)

    def test_zero_floor_equal_tier_floors_and_large_triggers_are_valid(self):
        settings = dict(floor=0.0, super_floor=0.0)
        self.assertAlmostEqual(profit_floor_price(100.0, 110.0, **settings), 100.0)
        self.assertAlmostEqual(profit_floor_price(100.0, 120.0, **settings), 100.0)
        self.assertAlmostEqual(profit_floor_price(
            100.0, 300.0, trigger=2.0, floor=0.5,
            super_trigger=3.0, super_floor=1.0), 150.0)

    def test_previous_floor_ratchets_across_regressing_observations(self):
        previous = None
        observations = ((109.0, None), (110.0, 102.0), (108.0, 102.0),
                        (120.0, 105.0), (112.0, 105.0), (100.0, 105.0))
        for known_high, expected in observations:
            with self.subTest(known_high=known_high, previous=previous):
                previous = profit_floor_price(100.0, known_high, previous)
                if expected is None:
                    self.assertIsNone(previous)
                else:
                    self.assertAlmostEqual(previous, expected)

    def test_a_valid_previous_floor_is_never_lowered_by_a_tier(self):
        for known_high in (109.0, 110.0, 120.0, 200.0):
            with self.subTest(known_high=known_high):
                self.assertAlmostEqual(profit_floor_price(100.0, known_high, 108.0),
                                       108.0)
        self.assertAlmostEqual(profit_floor_price(100.0, 110.0, 101.0), 102.0)
        self.assertAlmostEqual(profit_floor_price(100.0, 120.0, 103.0), 105.0)

    def test_invalid_buy_price_preserves_only_a_valid_previous_floor(self):
        for buy in INVALID_PRICES:
            with self.subTest(buy=buy):
                self.assertIsNone(profit_floor_price(buy, 120.0))
                self.assertAlmostEqual(profit_floor_price(buy, 120.0, 105.0), 105.0)

    def test_invalid_high_preserves_only_a_valid_previous_floor(self):
        for high in INVALID_PRICES:
            with self.subTest(known_high=high):
                self.assertIsNone(profit_floor_price(100.0, high))
                self.assertAlmostEqual(profit_floor_price(100.0, high, 105.0), 105.0)

    def test_invalid_previous_floor_is_ignored_without_inventing_protection(self):
        for previous in INVALID_PRICES:
            with self.subTest(previous=previous):
                self.assertIsNone(profit_floor_price(100.0, 109.0, previous))
                self.assertAlmostEqual(profit_floor_price(100.0, 110.0, previous),
                                       102.0)
                self.assertIsNone(profit_floor_price(None, None, previous))

    def test_invalid_configuration_ordering_fails(self):
        bad_settings = (
            {"floor": -0.01}, {"floor": 0.10}, {"floor": 0.11},
            {"trigger": 0.0}, {"trigger": 0.02},
            {"super_trigger": 0.10}, {"super_trigger": 0.09},
            {"super_floor": 0.01}, {"super_floor": 0.20},
            {"super_floor": 0.21},
        )
        for settings in bad_settings:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    profit_floor_price(100.0, 120.0, **settings)

    def test_nonfinite_or_nonnumeric_configuration_fails(self):
        for parameter in ("trigger", "floor", "super_trigger", "super_floor"):
            for value in (math.nan, math.inf, -math.inf, None, "missing"):
                with self.subTest(parameter=parameter, value=value):
                    with self.assertRaises(ValueError):
                        profit_floor_price(100.0, 120.0, **{parameter: value})

    def test_bad_prices_do_not_mask_invalid_configuration(self):
        with self.assertRaises(ValueError):
            profit_floor_price(None, None, 105.0, floor=-0.01)


class ProfitFloorExecutionTests(unittest.TestCase):
    def test_today_crossing_a_trigger_cannot_arm_todays_stop(self):
        floor = profit_floor_price(100.0, 109.0)
        self.assertIsNone(floor)
        stop = max(93.0, floor) if floor is not None else 93.0
        # Today's high may precede or follow its low; daily bars cannot tell us.
        self.assertIsNone(stop_fill_price(100.0, 125.0, 95.0, 110.0, 100.0, stop))
        self.assertAlmostEqual(profit_floor_price(100.0, 125.0), 105.0)

    def test_today_high_cannot_upgrade_the_frozen_first_tier_floor(self):
        floor = profit_floor_price(100.0, 119.0)
        self.assertAlmostEqual(floor, 102.0)
        self.assertAlmostEqual(
            stop_fill_price(110.0, 125.0, 101.0, 108.0, 110.0, max(93.0, floor)),
            102.0,
        )

    def test_a_gap_below_an_armed_floor_can_still_realize_a_loss(self):
        floor = profit_floor_price(100.0, 120.0)
        fill = stop_fill_price(99.0, 104.0, 98.0, 103.0, 105.0, floor)
        self.assertEqual(fill, 99.0)
        self.assertLess(fill, 100.0)

    def test_settlement_day_keeps_the_existing_close_only_rule(self):
        floor = profit_floor_price(100.0, 120.0)
        self.assertIsNone(stop_fill_price(
            110.0, 113.0, 103.0, 110.0, 110.0, floor, settlement_day=True))
        self.assertEqual(stop_fill_price(
            110.0, 113.0, 103.0, 104.0, 110.0, floor, settlement_day=True), 104.0)

    def test_locked_limit_down_never_becomes_a_guaranteed_profit_fill(self):
        floor = profit_floor_price(100.0, 120.0)
        for settlement_day in (False, True):
            with self.subTest(settlement_day=settlement_day):
                self.assertIsNone(stop_fill_price(
                    102.0, 102.5, 101.8, 102.0, 110.0, floor,
                    settlement_day=settlement_day))

    def test_guard_cannot_loosen_an_existing_tighter_stop(self):
        floor = profit_floor_price(100.0, 110.0)
        threshold = max(104.0, floor)
        self.assertEqual(threshold, 104.0)
        self.assertEqual(stop_fill_price(110.0, 111.0, 103.0, 106.0, 110.0,
                                         threshold), 104.0)


def calls_named(tree, name):
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == name]


def held_field(node, field):
    """Recognize a per-position field, read by subscript or dict.get."""
    if isinstance(node, ast.Subscript):
        return (isinstance(node.slice, ast.Constant) and node.slice.value == field
                and isinstance(node.value, ast.Subscript)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "held")
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get" and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == field):
        position = node.func.value
        return (isinstance(position, ast.Subscript)
                and isinstance(position.value, ast.Name)
                and position.value.id == "held")
    return False


class ProfitGuardIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse((ROOT / "quant_bot.py").read_text(encoding="utf-8"))
        cls.runner = next(node for node in ast.walk(cls.tree)
                          if isinstance(node, ast.FunctionDef)
                          and node.name == "run_ai_quant_system")
        cls.parents = {child: node for node in ast.walk(cls.runner)
                       for child in ast.iter_child_nodes(node)}

    def ancestors(self, node):
        while node in self.parents:
            node = self.parents[node]
            yield node

    def guard_call(self):
        calls = calls_named(self.runner, "profit_floor_price")
        self.assertEqual(len(calls), 1)
        return calls[0]

    def resolve_name(self, expression, before_line):
        """Follow a local reference back to its latest preceding assignment."""
        seen = set()
        while isinstance(expression, ast.Name) and expression.id not in seen:
            seen.add(expression.id)
            assignments = [node for node in ast.walk(self.runner)
                           if isinstance(node, ast.Assign) and node.lineno < before_line
                           and any(isinstance(target, ast.Name)
                                   and target.id == expression.id
                                   for target in node.targets)]
            if not assignments:
                break
            assignment = max(assignments, key=lambda node: node.lineno)
            expression, before_line = assignment.value, assignment.lineno
        return expression

    def test_guard_uses_prior_position_peak_without_current_bar_inputs(self):
        call = self.guard_call()
        high = (call.args[1] if len(call.args) > 1 else
                next(keyword.value for keyword in call.keywords
                     if keyword.arg == "known_high"))
        self.assertTrue(any(held_field(node, "highest") for node in ast.walk(high)))
        names = {node.id for argument in call.args for node in ast.walk(argument)
                 if isinstance(node, ast.Name)}
        names.update(node.id for keyword in call.keywords
                     for node in ast.walk(keyword.value) if isinstance(node, ast.Name))
        self.assertTrue(names.isdisjoint({"stocks_high", "curr_h", "curr_l", "curr_c"}))

    def test_guard_respects_existing_hold_days_and_policy_gates(self):
        ancestors = list(self.ancestors(self.guard_call()))
        gates = [node.test for node in ancestors if isinstance(node, ast.If)]
        self.assertTrue(any(isinstance(node, ast.Attribute)
                            and node.attr == "profit_guard"
                            for test in gates for node in ast.walk(test)))
        self.assertTrue(any(isinstance(node, ast.Name)
                            and node.id == "enable_asymmetric_atr_exit"
                            for test in gates for node in ast.walk(test)))
        self.assertTrue(any(
            isinstance(node, ast.Compare) and len(node.ops) == 1
            and isinstance(node.ops[0], ast.GtE)
            and held_field(node.left, "hold_days")
            and isinstance(node.comparators[0], ast.Name)
            and node.comparators[0].id == "MIN_HOLD_DAYS"
            for test in gates for node in ast.walk(test)))
        minimums = [node.value.value for node in ast.walk(self.runner)
                    if isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Constant)
                    and any(isinstance(target, ast.Name)
                            and target.id == "MIN_HOLD_DAYS" for target in node.targets)]
        self.assertEqual(minimums, [3])

    def test_guard_reuses_and_persists_the_position_floor(self):
        call = self.guard_call()
        previous = (call.args[2] if len(call.args) > 2 else
                    next(keyword.value for keyword in call.keywords
                         if keyword.arg == "previous_floor"))
        previous = self.resolve_name(previous, call.lineno)
        self.assertTrue(any(held_field(node, "profit_floor")
                            for node in ast.walk(previous)))
        helper_assignments = [node for node in ast.walk(self.runner)
                              if isinstance(node, ast.Assign) and node.value is call]
        self.assertEqual(len(helper_assignments), 1)
        result_names = {target.id for target in helper_assignments[0].targets
                        if isinstance(target, ast.Name)}
        self.assertTrue(any(
            isinstance(node, ast.Assign)
            and any(held_field(target, "profit_floor") for target in node.targets)
            and (node.value is call or (isinstance(node.value, ast.Name)
                                       and node.value.id in result_names))
            for node in ast.walk(self.runner)))

    def test_guard_raises_the_existing_threshold_before_existing_fill_logic(self):
        guard = self.guard_call()
        stops = calls_named(self.runner, "stop_fill_price")
        self.assertEqual(len(stops), 1)
        stop = stops[0]
        self.assertIsInstance(stop.args[5], ast.Name)
        threshold = stop.args[5].id
        result_names = {target.id for node in ast.walk(self.runner)
                        if isinstance(node, ast.Assign) and node.value is guard
                        for target in node.targets if isinstance(target, ast.Name)}
        updates = [node for node in ast.walk(self.runner)
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == threshold
                           for target in node.targets)
                   and guard.lineno < node.lineno < stop.lineno]
        raises_threshold = False
        for update in updates:
            value = update.value
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id == "max"):
                names = {arg.id for arg in value.args if isinstance(arg, ast.Name)}
                raises_threshold |= threshold in names and bool(names & result_names)
            elif isinstance(value, ast.Name) and value.id in result_names:
                # An explicit conditional assignment is equivalent to max().
                raises_threshold |= any(
                    isinstance(parent, ast.If)
                    and isinstance(parent.test, ast.Compare)
                    and len(parent.test.ops) == 1
                    and isinstance(parent.test.ops[0], ast.Gt)
                    and isinstance(parent.test.left, ast.Name)
                    and parent.test.left.id == value.id
                    and isinstance(parent.test.comparators[0], ast.Name)
                    and parent.test.comparators[0].id == threshold
                    for parent in self.ancestors(update))
        self.assertTrue(raises_threshold)
        self.assertTrue(any(isinstance(node, ast.For)
                            and any(child is stop for child in ast.walk(node))
                            for node in self.ancestors(guard)))
        settlement = next(keyword.value for keyword in stop.keywords
                          if keyword.arg == "settlement_day")
        self.assertIsInstance(settlement, ast.Compare)
        self.assertIsInstance(settlement.ops[0], ast.Eq)
        self.assertTrue(held_field(settlement.left, "hold_days"))
        self.assertEqual(settlement.comparators[0].id, "MIN_HOLD_DAYS")

    def test_current_high_updates_the_peak_only_after_intraday_execution(self):
        stop = calls_named(self.runner, "stop_fill_price")
        self.assertEqual(len(stop), 1)
        updates = [node for node in ast.walk(self.runner)
                   if isinstance(node, ast.Assign)
                   and any(held_field(target, "highest") for target in node.targets)]
        self.assertTrue(updates)
        self.assertTrue(all(node.lineno > stop[0].lineno for node in updates))


if __name__ == "__main__":
    unittest.main()
