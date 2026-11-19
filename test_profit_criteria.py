import copy
import unittest

from execution_policy import get_policy, position_budget
from validate_winrate import evaluate_candidate, summarize


class ProfitCriteriaTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            'full_period': {
                'win_rate_pct': 23.4, 'closed_positions': 47,
                'profit_factor': 1.38, 'return_pct': 32.24,
                'excess_over_cash_pp': 11.03, 'max_drawdown_pct': -6.89,
            },
            'annual': {
                str(year): {'start': f'{year}-01-03', 'end': f'{year}-12-30',
                            'excess_over_cash_pp': 4 if year == 2025 else -1}
                for year in range(2022, 2026)
            },
            'stress': {
                'higher_slippage': {'excess_over_cash_pp': 5, 'return_pct': 26},
                'exclude_vic_pdr': {'excess_over_cash_pp': -7, 'return_pct': 14},
            },
        }
        self.candidate = copy.deepcopy(self.baseline)
        self.candidate['full_period'].update(win_rate_pct=30, closed_positions=37, return_pct=33)
        self.candidate['annual']['2023']['excess_over_cash_pp'] = 1
        self.candidate['stress']['exclude_vic_pdr'].update(excess_over_cash_pp=1, return_pct=22)

    def test_improved_results_with_acceptable_sample_pass(self):
        report = evaluate_candidate(self.candidate, self.baseline)
        self.assertTrue(report['historical_pass'])
        self.assertEqual(report['cash_beating_years'], 2)

    def test_higher_win_rate_cannot_offset_lower_net_return(self):
        self.candidate['full_period']['return_pct'] = 30
        report = evaluate_candidate(self.candidate, self.baseline)
        self.assertFalse(report['checks']['net_return_at_least_baseline_and_above_cash'])
        self.assertFalse(report['historical_pass'])

    def test_higher_win_rate_cannot_offset_thin_sample(self):
        self.candidate['full_period']['closed_positions'] = 35
        report = evaluate_candidate(self.candidate, self.baseline)
        self.assertFalse(report['checks']['at_least_30_closes_and_75pct_of_baseline'])

    def test_stress_below_cash_or_baseline_rejects_candidate(self):
        self.candidate['stress']['exclude_vic_pdr'].update(excess_over_cash_pp=-1, return_pct=20)
        self.candidate['stress']['higher_slippage'].update(excess_over_cash_pp=3, return_pct=24)
        report = evaluate_candidate(self.candidate, self.baseline)
        self.assertFalse(report['checks']['exclude_vic_pdr_above_cash_and_at_least_baseline'])
        self.assertFalse(report['checks']['higher_slippage_above_cash_and_at_least_baseline'])
        self.assertFalse(report['historical_pass'])

    def test_partial_year_does_not_rescue_inconsistent_performance(self):
        self.candidate['annual']['2023']['excess_over_cash_pp'] = -1
        partial_year = {'start': '2026-01-05', 'end': '2026-11-18', 'excess_over_cash_pp': 5}
        self.candidate['annual']['2026'] = partial_year
        self.baseline['annual']['2026'] = partial_year
        report = evaluate_candidate(self.candidate, self.baseline)
        self.assertNotIn('2026', report['full_years_evaluated'])
        self.assertFalse(report['checks']['beat_cash_in_at_least_half_the_full_years_and_match_baseline'])

    def test_cash_benchmark_counts_same_sessions_as_nav(self):
        initial = 100_000_000
        final = initial * (1 + 0.04/252)**2
        rows = [{'Date': '2025-03-03', 'NAV': initial * (1 + 0.04/252), 'Cash': initial},
                {'Date': '2025-03-04', 'NAV': final, 'Cash': final}]
        self.assertEqual(summarize(rows, [])['excess_over_cash_pp'], 0.0)


class RiskBudgetTests(unittest.TestCase):
    def test_wide_stop_and_costs_derive_smaller_budget(self):
        nav, slot = 100_000_000, 20_000_000
        stop, buy_cost, sell_cost = -0.085, 0.0065, 0.0075
        mode = get_policy('balanced')
        budget = position_budget(mode, nav, slot, stop, buy_cost, sell_cost)
        proceeds_at_stop = budget / (1 + buy_cost) * (1 + stop) * (1 - sell_cost)
        self.assertLess(budget, slot)
        self.assertAlmostEqual(budget - proceeds_at_stop, nav * 0.015)
        self.assertLess(position_budget(mode, nav, slot, stop, buy_cost + .005, sell_cost + .005), budget)

    def test_risk_budget_cannot_increase_slot(self):
        self.assertEqual(position_budget(get_policy('balanced'), 100_000_000,
                                         20_000_000, -.04, .0065, .0075), 20_000_000)
        self.assertEqual(position_budget(get_policy('baseline'), 100_000_000,
                                         20_000_000, -.085, .0065, .0075), 20_000_000)
