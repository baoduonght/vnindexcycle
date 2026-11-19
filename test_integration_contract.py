"""Dependency-free guards for the replay wiring and comparison accounting.

These checks do not simulate a strategy or establish a win-rate improvement.
"""

import ast
from contextlib import redirect_stderr
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

import validate_winrate


ROOT = Path(__file__).resolve().parents[1]


def function_named(tree, name):
    return next(node for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name)


def calls_named(tree, name):
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == name]


class BotIntegrationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse((ROOT / "quant_bot.py").read_text(encoding="utf-8"))
        cls.runner = function_named(cls.tree, "run_ai_quant_system")

    def test_known_session_selection_excludes_current_bar(self):
        assignments = [node for node in ast.walk(self.runner)
                       if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "known_dates"
                               for t in node.targets)]
        self.assertEqual(len(assignments), 1)
        comparisons = [node for node in ast.walk(assignments[0].value)
                       if isinstance(node, ast.Compare)]
        self.assertTrue(any(len(node.ops) == 1 and isinstance(node.ops[0], ast.Lt)
                            and isinstance(node.comparators[0], ast.Name)
                            and node.comparators[0].id == "date"
                            for node in comparisons))

    def test_intraday_threshold_uses_known_session_not_current_close(self):
        stop_calls = calls_named(self.runner, "stop_fill_price")
        self.assertEqual(len(stop_calls), 1)
        intraday_assignments = [node for node in ast.walk(self.runner)
                                if isinstance(node, ast.Assign)
                                and any(isinstance(t, ast.Name) and t.id == "dynamic_sl"
                                        for t in node.targets)]
        self.assertEqual(len(intraday_assignments), 1)
        value = intraday_assignments[0].value
        self.assertIsInstance(value, ast.Call)
        self.assertEqual(value.func.id, "calc_dynamic_stoploss")
        self.assertEqual(value.args[1].id, "known_date")
        settlement = next(k.value for k in stop_calls[0].keywords
                          if k.arg == "settlement_day")
        self.assertIsInstance(settlement, ast.Compare)
        self.assertIsInstance(settlement.ops[0], ast.Eq)
        self.assertEqual(settlement.comparators[0].id, "MIN_HOLD_DAYS")

    def test_next_open_filter_receives_frozen_signal_references(self):
        calls = calls_named(self.runner, "open_rejection")
        self.assertEqual(len(calls), 1)
        reference_names = [arg.args[0].value for arg in calls[0].args[2:]
                           if isinstance(arg, ast.Call) and arg.args
                           and isinstance(arg.args[0], ast.Constant)]
        self.assertEqual(reference_names,
                         ["signal_close", "signal_atr", "breakout_level"])

    def test_buy_market_value_does_not_capitalize_execution_costs(self):
        updates = [node for node in ast.walk(self.runner)
                   if isinstance(node, ast.AugAssign)
                   and isinstance(node.target, ast.Name)
                   and node.target.id == "stock_val_open"]
        self.assertEqual(len(updates), 1)
        names = {node.id for node in ast.walk(updates[0].value)
                 if isinstance(node, ast.Name)}
        self.assertEqual(names, {"shares", "price_vnd"})
        self.assertIsInstance(updates[0].value.op, ast.Mult)

    def test_offline_return_precedes_network_fallback(self):
        fetch = function_named(self.tree, "fetch_ticker_data")
        gates = [node for node in fetch.body
                 if isinstance(node, ast.If)
                 and isinstance(node.test, ast.Name)
                 and node.test.id == "QUANT_OFFLINE"]
        self.assertEqual(len(gates), 1)
        self.assertTrue(any(isinstance(node, ast.Return) for node in gates[0].body))
        network_calls = [node for node in ast.walk(fetch)
                         if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Attribute)
                         and isinstance(node.func.value, ast.Name)
                         and node.func.value.id == "requests"]
        self.assertTrue(network_calls)
        self.assertLess(gates[0].lineno, min(c.lineno for c in network_calls))

    def test_missing_model_feature_fails_instead_of_silent_nan(self):
        predictor = function_named(self.tree, "predict_with_model")
        self.assertTrue(any(isinstance(node, ast.Raise) for node in ast.walk(predictor)))
        silent_assignments = [node for node in ast.walk(predictor)
                              if isinstance(node, ast.Assign)
                              and isinstance(node.value, ast.Attribute)
                              and node.value.attr == "nan"]
        self.assertEqual(silent_assignments, [])


class ComparisonAccountingTests(unittest.TestCase):
    def setUp(self):
        self.nav = [
            {"Date": "2023-01-04", "NAV": 121.0, "Cash": 121.0},
            {"Date": "2022-12-30", "NAV": 110.0, "Cash": 110.0},
            {"Date": "2023-01-03", "NAV": 108.0, "Cash": 50.0},
        ]
        self.trades = [
            {"Exit_Date": "2022-12-30", "Ticker": "A", "Total_PnL_VND": 10.0},
            {"Exit_Date": "2023-01-03", "Ticker": "B", "Total_PnL_VND": -2.0},
            {"Exit_Date": "2023-01-04", "Ticker": "C", "Total_PnL_VND": 5.0},
            {"Exit_Date": "2023-01-05", "Ticker": "D", "Total_PnL_VND": 999.0},
        ]

    def test_annual_return_uses_last_prior_nav_and_closed_positions_only(self):
        result = validate_winrate.summarize(
            self.nav, self.trades, "2023-01-01", "2023-12-31", initial=100.0
        )
        self.assertEqual(result["return_pct"], 10.0)
        self.assertEqual(result["closed_positions"], 2)
        self.assertEqual(result["wins"], 1)
        self.assertEqual(result["win_rate_pct"], 50.0)
        self.assertEqual(result["trade_pnl"], 3.0)
        self.assertEqual(result["profit_factor"], 2.5)
        self.assertEqual(result["cash_only_pct"], 50.0)
        self.assertAlmostEqual(result["max_drawdown_pct"], -1.8182, places=4)

    def test_full_period_return_starts_from_initial_capital(self):
        result = validate_winrate.summarize(self.nav, self.trades, initial=100.0)
        self.assertEqual(result["return_pct"], 21.0)
        self.assertEqual(result["closed_positions"], 3)
        self.assertEqual(result["trade_pnl"], 13.0)

    def test_empty_interval_has_no_invented_metrics(self):
        self.assertIsNone(validate_winrate.summarize(
            self.nav, self.trades, "2024-01-01", "2024-12-31", initial=100.0
        ))

    def test_no_completed_positions_is_not_a_zero_percent_win_rate(self):
        result = validate_winrate.summarize(self.nav, [], initial=100.0)
        self.assertEqual(result["closed_positions"], 0)
        self.assertIsNone(result["win_rate_pct"])
        self.assertIsNone(result["profit_factor"])
        self.assertIsNone(result["win_rate_95pct_descriptive_interval"])

    def test_missing_dependencies_stop_before_any_output_is_created(self):
        # The failure path must not create anything, including a parent directory.
        output = ROOT / "tests" / ("never-generated-" + uuid.uuid4().hex)
        self.assertFalse(output.exists())
        errors = io.StringIO()
        with patch.object(validate_winrate.importlib.util, "find_spec", return_value=None), \
                patch.object(Path, "mkdir") as mkdir, \
                patch.object(Path, "write_text") as write_text:
            with redirect_stderr(errors):
                with self.assertRaises(SystemExit) as caught:
                    validate_winrate.main(["--output", str(output)])
            mkdir.assert_not_called()
            write_text.assert_not_called()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("no replay results were generated", errors.getvalue())
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
