"""Run local regression tests from either supported repository layout."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
REQUIRED_TESTS = (
    'test_execution_policy.py',
    'test_integration_contract.py',
    'test_profit_criteria.py',
    'test_run_audit.py',
    'test_test_runner.py',
    'test_workflow_config.py',
)
REQUIRED_PROJECT_FILES = (
    'quant_bot.py',
    'execution_policy.py',
    'run_audit.py',
    'validate_winrate.py',
    '.github/workflows/quant_bot.yml',
)


def build_suite(root=None):
    root = Path(root if root is not None else ROOT).resolve()
    missing_project = [name for name in REQUIRED_PROJECT_FILES
                       if not (root/name).is_file()]
    if missing_project:
        raise RuntimeError('Incomplete project upload. Missing: '+', '.join(missing_project))
    tests = root/'tests'
    if tests.is_dir():
        required = ('__init__.py',) + REQUIRED_TESTS
    elif any((root/name).is_file() for name in REQUIRED_TESTS):
        tests = root
        required = REQUIRED_TESTS
    else:
        raise RuntimeError(
            f'No local regression tests in {root} or {tests}. '
            'Upload all test_*.py files next to run_tests.py, or inside tests/.')
    missing = [name for name in required if not (tests/name).is_file()]
    if missing:
        raise RuntimeError(f'Incomplete test upload in {tests}. Missing: '+', '.join(missing))
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
