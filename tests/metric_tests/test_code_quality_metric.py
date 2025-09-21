import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from loguru import logger
from src.metrics.CodeQualityMetric import CodeQualityMetric
from tests.metric_tests.base_metric_test import BaseMetricTest


class TestCodeQualityMetric(BaseMetricTest):
    @pytest.fixture
    def metric(self):
        return CodeQualityMetric()

    @pytest.fixture
    def model_no_github(self, base_model):
        model = base_model
        model._github_metadata = None
        return model

    @pytest.fixture
    def model_popularity_only(self, base_model):
        model = base_model
        model._github_metadata = {
            "stargazers_count": 150,  # 3 * 50 = 3 * 0.01 = 0.03
            "forks_count": 30,        # 3 * 10 = 3 * 0.01 = 0.03
            "avg_daily_commits_30d": 2.0  # 2 * 0.05 = 0.1
        }
        return model

    @pytest.fixture
    def model_with_clone_url(self, base_model):
        model = base_model
        model._github_metadata = {
            "stargazers_count": 100,
            "forks_count": 20,
            "avg_daily_commits_30d": 1.0,
            "clone_url": "https://github.com/test/repo.git"
        }
        return model

    @pytest.fixture
    def model_max_popularity(self, base_model):
        model = base_model
        model._github_metadata = {
            "stargazers_count": 1000,  # Max popularity: 0.2
            "forks_count": 200,        # Max popularity: 0.2
            "avg_daily_commits_30d": 10,  # Max commits: 0.3
            "clone_url": "https://github.com/test/repo.git"
        }
        return model

    # --- Tests ---

    def test_no_github_metadata(self, metric, model_no_github):
        logger.info("Testing model with no GitHub metadata...")
        score = metric.evaluate(model_no_github)
        assert score == 0.0

    def test_popularity_only_no_clone(self, metric, model_popularity_only):
        logger.info("Testing model with popularity data but no clone URL...")
        score = metric.evaluate(model_popularity_only)
        # 0.03 (stars) + 0.03 (forks) + 0.1 (commits) = 0.16
        assert score == pytest.approx(0.16, abs=0.01)

    @patch('src.metrics.CodeQualityMetric.subprocess.run')
    def test_clone_repository_success(self, mock_run, metric):
        logger.info("Testing successful repository cloning...")
        mock_run.return_value = MagicMock()

        result = metric._clone_repository("https://github.com/test/repo.git", "/tmp/test")

        assert result is True
        mock_run.assert_called_once()

    @patch('src.metrics.CodeQualityMetric.subprocess.run')
    def test_clone_repository_failure(self, mock_run, metric):
        logger.info("Testing repository cloning failure...")
        from subprocess import CalledProcessError
        mock_run.side_effect = CalledProcessError(1, 'git clone')

        result = metric._clone_repository("https://github.com/test/repo.git", "/tmp/test")

        assert result is False

    @patch('src.metrics.CodeQualityMetric.subprocess.run')
    def test_clone_repository_timeout(self, mock_run, metric):
        logger.info("Testing repository cloning timeout...")
        from subprocess import TimeoutExpired
        mock_run.side_effect = TimeoutExpired('git clone', 30)

        result = metric._clone_repository("https://github.com/test/repo.git", "/tmp/test")

        assert result is False

    @patch('src.metrics.CodeQualityMetric.Path')
    def test_count_test_files(self, mock_path, metric):
        logger.info("Testing test file counting...")
        mock_repo = MagicMock()
        mock_path.return_value = mock_repo

        mock_repo.glob.side_effect = lambda pattern: {
            'tests/**/*.py': ['test1.py', 'test2.py', 'test3.py'],
            'test/**/*.py': ['test4.py']
        }.get(pattern, [])

        result = metric._count_test_files("/fake/path")

        assert result == 4  # 3 + 1

    @patch('src.metrics.CodeQualityMetric.Path')
    def test_count_source_files(self, mock_path, metric):
        logger.info("Testing source file counting...")
        mock_repo = MagicMock()
        mock_path.return_value = mock_repo

        mock_files = [
            MagicMock(parts=['src', 'main.py']),           # Include
            MagicMock(parts=['src', 'utils.py']),          # Include
            MagicMock(parts=['tests', 'test_main.py']),    # Exclude
            MagicMock(parts=['docs', 'example.py']),       # Exclude
            MagicMock(parts=['lib', 'helper.py'])          # Include
        ]
        mock_repo.rglob.return_value = mock_files

        result = metric._count_source_files("/fake/path")

        assert result == 3  # main.py, utils.py, helper.py

    @patch('src.metrics.CodeQualityMetric.Path')
    def test_documentation_all_files_present(self, mock_path, metric):
        logger.info("Testing documentation evaluation with all files present...")
        mock_repo = MagicMock()
        mock_path.return_value = mock_repo

        mock_repo.glob.side_effect = lambda pattern: {
            'LICENSE*': ['LICENSE'],
            'license*': [],
            'README*': ['README.md'],
            'readme*': [],
            'CONTRIBUTING*': ['CONTRIBUTING.md'],
            'contributing*': []
        }.get(pattern, [])

        result = metric._evaluate_documentation("/fake/path")

        assert result == 0.2  # 0.05 + 0.05 + 0.10

    @patch('src.metrics.CodeQualityMetric.Path')
    def test_documentation_readme_only(self, mock_path, metric):
        logger.info("Testing documentation evaluation with README only...")
        mock_repo = MagicMock()
        mock_path.return_value = mock_repo
        
        mock_repo.glob.side_effect = lambda pattern: {
            'LICENSE*': [],
            'license*': [],
            'README*': ['README.md'],
            'readme*': [],
            'CONTRIBUTING*': [],
            'contributing*': []
        }.get(pattern, [])
        
        result = metric._evaluate_documentation("/fake/path")
        
        assert result == 0.05  # Only README

    @patch('src.metrics.CodeQualityMetric.Path')
    def test_documentation_no_files(self, mock_path, metric):
        logger.info("Testing documentation evaluation with no files...")
        mock_repo = MagicMock()
        mock_path.return_value = mock_repo
        mock_repo.glob.return_value = []
        
        result = metric._evaluate_documentation("/fake/path")
        
        assert result == 0.0

    @pytest.mark.parametrize("stars,forks,expected", [
        (0, 0, 0.0),
        (49, 9, 0.0),     # Below thresholds
        (50, 10, 0.02),   # Exactly at thresholds
        (100, 20, 0.04),  # 2*0.01 + 2*0.01
        (500, 100, 0.2),  # Capped at 0.1 each
    ])
    def test_popularity_score_calculation(self, metric, base_model, stars, forks, expected):
        logger.info(f"Testing popularity calculation: {stars} stars, {forks} forks...")
        model = base_model
        model._github_metadata = {
            "stargazers_count": stars,
            "forks_count": forks,
            "avg_daily_commits_30d": 0
        }
        
        result = metric.evaluate(model)
        assert result == pytest.approx(expected, abs=0.01)

    @patch.object(CodeQualityMetric, '_clone_and_analyze')
    def test_total_score_capped_at_one(self, mock_clone_analyze, metric, model_max_popularity):
        logger.info("Testing that total score is capped at 1.0...")
        mock_clone_analyze.return_value = (0.3, 0.2)  # test_score, doc_score
        
        result = metric.evaluate(model_max_popularity)
        
        assert result == 1.0

    @patch('src.metrics.CodeQualityMetric.tempfile.TemporaryDirectory')
    @patch('src.metrics.CodeQualityMetric.subprocess.run')
    @patch('src.metrics.CodeQualityMetric.Path')
    def test_successful_clone_and_analyze(self, mock_path, mock_subprocess, mock_tempdir, metric, model_with_clone_url):
        logger.info("Testing successful clone and analysis...")
        
        # Setup mocks
        mock_temp_context = MagicMock()
        mock_temp_context.__enter__.return_value = "/tmp/test_repo"
        mock_temp_context.__exit__.return_value = None
        mock_tempdir.return_value = mock_temp_context
        
        mock_subprocess.return_value = MagicMock()
        
        mock_repo = MagicMock()
        mock_path.return_value = mock_repo
        
        # Mock test files and source files
        mock_repo.glob.side_effect = lambda pattern: {
            'tests/**/*.py': ['test1.py', 'test2.py'],
            'test/**/*.py': [],
            'README*': ['README.md'],
            'LICENSE*': ['LICENSE'],
            'CONTRIBUTING*': [],
            'readme*': [],
            'license*': [],
            'contributing*': []
        }.get(pattern, [])
        
        mock_repo.rglob.return_value = [
            MagicMock(parts=['src', 'main.py']),
            MagicMock(parts=['src', 'utils.py'])
        ]
        
        result = metric.evaluate(model_with_clone_url)
        
        # Should include popularity + commits + test score + doc score
        assert result > 0.08  # At least popularity + commits
