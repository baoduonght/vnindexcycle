"""Guard that the selected mode reaches the trading process, not training."""
import ast
from pathlib import Path
import re
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == 'tests' else HERE


def workflow_steps(text):
    return re.split(r"(?m)^      - name:", text)[1:]


class WorkflowModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = (ROOT / '.github/workflows/quant_bot.yml').read_text(encoding='utf-8')
        cls.steps = workflow_steps(cls.workflow)

    def step_running(self, script):
        matches = [step for step in self.steps
                   if re.search(r'(?m)^        run:.*\bpython\s+' + re.escape(script) + r'\s*$', step)]
        self.assertEqual(len(matches), 1, f'Expected one execution step for {script}')
        return matches[0]

    def test_training_command_has_no_entry_mode_prefix(self):
        step = self.step_running('train_bot.py')
        self.assertRegex(step, r'(?m)^        run: python train_bot\.py\s*$')
        self.assertNotIn('ENTRY_MODE:', step)
        self.assertNotIn('ENTRY_MODE=', step)

    def test_quant_step_receives_selected_mode_and_assertion(self):
        step = self.step_running('quant_bot.py')
        value = "${{ inputs.entry_mode || 'balanced' }}"
        for name in ('ENTRY_MODE', 'RUN_EXPECTED_ENTRY_MODE'):
            self.assertRegex(step, r'(?m)^          ' + name + ': ' + re.escape(value) + r'\s*$')

    def test_dispatch_selects_balanced_and_exposes_only_known_modes(self):
        mode_block = self.workflow.split('      entry_mode:', 1)[1].split('      retrain:', 1)[0]
        self.assertRegex(mode_block, r'(?m)^        type: choice\s*$')
        self.assertRegex(mode_block, r'(?m)^        default: balanced\s*$')
        options = re.findall(r'(?m)^          - (\w+)\s*$', mode_block)
        self.assertEqual(set(options), {'baseline', 'quality', 'balanced', 'index_trend', 'profit_guard'})
        self.assertEqual(len(options), 5)

    def test_manual_retraining_is_opt_in_for_all_mutating_training_steps(self):
        retrain_block = self.workflow.split('      retrain:', 1)[1].split('\npermissions:', 1)[0]
        self.assertRegex(retrain_block, r'(?m)^        type: boolean\s*$')
        self.assertRegex(retrain_block, r'(?m)^        default: false\s*$')
        training_steps = [step for step in self.steps
                          if 'run: python macro_database.py' in step
                          or 'run: python train_bot.py' in step
                          or 'git add xgboost_*.json' in step]
        self.assertEqual(len(training_steps), 3)
        for step in training_steps:
            condition = next(line.strip() for line in step.splitlines() if line.strip().startswith('if:'))
            self.assertIn("github.event.schedule == '0 2 * * 6'", condition)
            self.assertIn("(github.event_name == 'workflow_dispatch' && inputs.retrain)", condition)
            self.assertNotRegex(condition, r"\|\|\s*github.event_name == 'workflow_dispatch'\s*$")

    def test_report_commit_includes_diagnostics_and_manifest(self):
        commit_step = next(step for step in self.steps if 'git add trading_log.csv' in step)
        self.assertIn('signal_diagnostics.csv', commit_step)
        self.assertIn('run_manifest.json', commit_step)

    def test_tests_run_before_trading_process(self):
        self.assertLess(self.workflow.index('run: python run_tests.py'),
                        self.workflow.index('run: python quant_bot.py'))

    def test_bot_resolves_environment_mode_with_shared_checker(self):
        tree = ast.parse((ROOT/'quant_bot.py').read_text(encoding='utf-8'))
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'ENTRY_MODE' for t in node.targets)]
        self.assertEqual(len(assignments), 1)
        value = assignments[0].value
        self.assertIsInstance(value, ast.Call)
        self.assertEqual(value.func.id, 'resolve_entry_mode')
        self.assertIsInstance(value.args[0], ast.Attribute)
        self.assertEqual(value.args[0].value.id, 'os')
        self.assertEqual(value.args[0].attr, 'environ')


if __name__ == '__main__':
    unittest.main()
