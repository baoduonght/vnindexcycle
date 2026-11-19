"""Run local tests from this file's directory; fail clearly on partial uploads."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
REQUIRED_TESTS = (
    '__init__.py',
    'test_execution_policy.py',
    'test_integration_contract.py',
    'test_profit_criteria.py',
    'test_run_audit.py',
    'test_test_runner.py',
    'test_workflow_config.py',
)


def build_suite(root=None):
    root = Path(root if root is not None else ROOT).resolve()
    tests = root/'tests'
    if not tests.is_dir():
        raise RuntimeError(
            f'Missing local test directory: {tests}. Upload tests/ next to quant_bot.py '
            'and run_tests.py, not inside another vnindexcycle-main/ directory.')
    missing = [name for name in REQUIRED_TESTS if not (tests/name).is_file()]
    if missing:
        raise RuntimeError('Incomplete tests/ upload. Missing: '+', '.join(missing))
    suite = unittest.TestLoader().discover(
        start_dir=str(tests), pattern='test_*.py', top_level_dir=str(root))
    if suite.countTestCases() == 0:
        raise RuntimeError(f'No tests discovered in {tests}; refusing a false successful run')
    return suite


def main():
    # Make project modules importable regardless of the shell working directory.
    sys.path.insert(0, str(ROOT))
    try:
        print(f'Test project root: {ROOT}', flush=True)
        suite = build_suite()
    except (RuntimeError, ImportError, TypeError) as exc:
        print(f'TEST SETUP ERROR: {exc}', file=sys.stderr)
        return 2
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
