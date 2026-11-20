"""Regression tests for flat/nested uploads and safe local discovery."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import run_tests


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == 'tests' else HERE


@contextmanager
def foreign_cwd():
    previous = Path.cwd()
    os.chdir(ROOT.parent)
    try:
        yield
    finally:
        os.chdir(previous)


def project_files(root=ROOT):
    return {root/name for name in run_tests.REQUIRED_PROJECT_FILES}


def test_files(directory):
    return {directory/name for name in run_tests.REQUIRED_TESTS}


@contextmanager
def fake_filesystem(files, directories):
    files = {path.resolve() for path in files}
    directories = {path.resolve() for path in directories}
    with patch.object(Path, 'is_file', lambda path: path.resolve() in files), \
            patch.object(Path, 'is_dir', lambda path: path.resolve() in directories):
        yield


def discovered_suite():
    suite = Mock()
    suite.countTestCases.return_value = 1
    loader = Mock()
    loader.discover.return_value = suite
    return loader, suite


class RunnerDiscoveryTests(unittest.TestCase):
    def test_flat_layout_is_anchored_to_script_from_foreign_cwd(self):
        loader, suite = discovered_suite()
        files = project_files() | test_files(ROOT)
        with foreign_cwd(), fake_filesystem(files, {ROOT}), \
                patch.object(run_tests, 'ROOT', ROOT), \
                patch.object(run_tests.unittest, 'TestLoader', return_value=loader):
            self.assertNotEqual(Path.cwd(), ROOT)
            self.assertIs(run_tests.build_suite(), suite)
        loader.discover.assert_called_once_with(
            start_dir=str(ROOT), pattern='test_*.py', top_level_dir=str(ROOT))

    def test_nested_layout_uses_absolute_paths_and_project_top_level(self):
        directory = ROOT/'tests'
        files = project_files() | test_files(directory) | {directory/'__init__.py'}
        loader, suite = discovered_suite()
        with foreign_cwd(), fake_filesystem(files, {ROOT, directory}), \
                patch.object(run_tests.unittest, 'TestLoader', return_value=loader):
            self.assertIs(run_tests.build_suite(ROOT), suite)
        loader.discover.assert_called_once_with(
            start_dir=str(directory), pattern='test_*.py', top_level_dir=str(ROOT))

    def test_explicit_root_overrides_script_root(self):
        loader, suite = discovered_suite()
        files = project_files() | test_files(ROOT)
        with fake_filesystem(files, {ROOT}), \
                patch.object(run_tests, 'ROOT', ROOT.parent/'not-the-project'), \
                patch.object(run_tests.unittest, 'TestLoader', return_value=loader):
            self.assertIs(run_tests.build_suite(ROOT), suite)
        loader.discover.assert_called_once_with(
            start_dir=str(ROOT), pattern='test_*.py', top_level_dir=str(ROOT))

    def test_absent_project_root_fails_before_discovery(self):
        with fake_filesystem(set(), set()), \
                patch.object(run_tests.unittest, 'TestLoader') as loader:
            with self.assertRaises(RuntimeError):
                run_tests.build_suite(ROOT.parent/'missing-project')
            loader.assert_not_called()

    def test_no_local_tests_cannot_import_unrelated_namespace(self):
        namespace = types.ModuleType('tests')
        namespace.__file__ = None
        namespace.__path__ = ['an-unrelated-namespace']
        with fake_filesystem(project_files(), {ROOT}), \
                patch.dict(sys.modules, {'tests': namespace}), \
                patch.object(run_tests.unittest, 'TestLoader') as loader:
            with self.assertRaises(RuntimeError):
                run_tests.build_suite(ROOT)
            loader.assert_not_called()

    def test_missing_project_dependency_fails_before_discovery(self):
        for name in run_tests.REQUIRED_PROJECT_FILES:
            files = (project_files() | test_files(ROOT)) - {ROOT/name}
            with self.subTest(file=name), fake_filesystem(files, {ROOT}), \
                    patch.object(run_tests.unittest, 'TestLoader') as loader:
                with self.assertRaises(RuntimeError) as caught:
                    run_tests.build_suite(ROOT)
                self.assertIn(name, str(caught.exception))
                loader.assert_not_called()

    def test_incomplete_flat_upload_names_missing_test(self):
        for name in run_tests.REQUIRED_TESTS:
            files = (project_files() | test_files(ROOT)) - {ROOT/name}
            with self.subTest(file=name), fake_filesystem(files, {ROOT}), \
                    patch.object(run_tests.unittest, 'TestLoader') as loader:
                with self.assertRaises(RuntimeError) as caught:
                    run_tests.build_suite(ROOT)
                self.assertIn(name, str(caught.exception))
                loader.assert_not_called()

    def test_nested_package_requires_init_file(self):
        directory = ROOT/'tests'
        files = project_files() | test_files(directory)
        with fake_filesystem(files, {ROOT, directory}), \
                patch.object(run_tests.unittest, 'TestLoader') as loader:
            with self.assertRaises(RuntimeError) as caught:
                run_tests.build_suite(ROOT)
            self.assertIn('__init__.py', str(caught.exception))
            loader.assert_not_called()

    def test_incomplete_nested_layout_does_not_fall_through_to_complete_flat_copy(self):
        directory = ROOT/'tests'
        missing = run_tests.REQUIRED_TESTS[0]
        files = (project_files() | test_files(ROOT) | test_files(directory)
                 | {directory/'__init__.py'}) - {directory/missing}
        with fake_filesystem(files, {ROOT, directory}), \
                patch.object(run_tests.unittest, 'TestLoader') as loader:
            with self.assertRaises(RuntimeError) as caught:
                run_tests.build_suite(ROOT)
            self.assertIn(missing, str(caught.exception))
            loader.assert_not_called()

    def test_empty_discovered_suite_fails(self):
        loader = Mock()
        loader.discover.return_value = unittest.TestSuite()
        with fake_filesystem(project_files() | test_files(ROOT), {ROOT}), \
                patch.object(run_tests.unittest, 'TestLoader', return_value=loader):
            with self.assertRaises(RuntimeError):
                run_tests.build_suite(ROOT)

    def test_discovery_import_error_propagates_to_main(self):
        loader = Mock()
        loader.discover.side_effect = ImportError('invalid local package')
        with fake_filesystem(project_files() | test_files(ROOT), {ROOT}), \
                patch.object(run_tests.unittest, 'TestLoader', return_value=loader):
            with self.assertRaisesRegex(ImportError, 'invalid local package'):
                run_tests.build_suite(ROOT)

    def test_required_files_exist_in_delivered_layout(self):
        directory = ROOT/'tests' if (ROOT/'tests').is_dir() else ROOT
        for name in run_tests.REQUIRED_PROJECT_FILES:
            with self.subTest(project_file=name):
                self.assertTrue((ROOT/name).is_file(), name)
        for name in run_tests.REQUIRED_TESTS:
            with self.subTest(test_file=name):
                self.assertTrue((directory/name).is_file(), name)
        if directory != ROOT:
            self.assertTrue((directory/'__init__.py').is_file())
        self.assertNotIn('__init__.py', run_tests.REQUIRED_TESTS)
        self.assertIn(Path(__file__).name, run_tests.REQUIRED_TESTS)


class RunnerExitStatusTests(unittest.TestCase):
    def test_success_from_foreign_cwd_returns_zero_and_sets_import_path(self):
        suite = Mock()
        runner = Mock()
        runner.run.return_value.wasSuccessful.return_value = True
        output = io.StringIO()
        with foreign_cwd(), patch.object(run_tests.sys, 'path', list(sys.path)), \
                patch.object(run_tests, 'build_suite', return_value=suite), \
                patch.object(run_tests.unittest, 'TextTestRunner', return_value=runner), \
                redirect_stdout(output):
            self.assertEqual(run_tests.main(), 0)
            self.assertEqual(run_tests.sys.path[0], str(ROOT))
        runner.run.assert_called_once_with(suite)
        self.assertIn(str(ROOT), output.getvalue())

    def test_failing_tests_return_one(self):
        runner = Mock()
        runner.run.return_value.wasSuccessful.return_value = False
        with patch.object(run_tests.sys, 'path', list(sys.path)), \
                patch.object(run_tests, 'build_suite', return_value=Mock()), \
                patch.object(run_tests.unittest, 'TextTestRunner', return_value=runner), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(run_tests.main(), 1)

    def test_setup_errors_return_two_without_running_tests(self):
        for error in (RuntimeError('missing file'), ImportError('bad import'),
                      TypeError('invalid namespace package')):
            errors = io.StringIO()
            with self.subTest(error=error), \
                    patch.object(run_tests.sys, 'path', list(sys.path)), \
                    patch.object(run_tests, 'build_suite', side_effect=error), \
                    patch.object(run_tests.unittest, 'TextTestRunner') as runner, \
                    redirect_stdout(io.StringIO()), redirect_stderr(errors):
                self.assertEqual(run_tests.main(), 2)
                runner.assert_not_called()
            self.assertIn('TEST SETUP ERROR:', errors.getvalue())
            self.assertIn(str(error), errors.getvalue())


if __name__ == '__main__':
    unittest.main()
