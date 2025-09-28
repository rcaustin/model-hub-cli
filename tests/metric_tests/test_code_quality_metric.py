import pytest
from unittest.mock import patch, MagicMock
from typing import Any, Dict, List
from loguru import logger
from src.metrics.CodeQualityMetric import CodeQualityMetric
from tests.metric_tests.base_metric_test import BaseMetricTest


class TestCodeQualityMetric(BaseMetricTest):
    @pytest.fixture
    def metric(self) -> CodeQualityMetric:
        return CodeQualityMetric()

    @pytest.fixture
    def model_no_github(self, base_model: Any) -> Any:
        model = base_model
        model._github_metadata = None
        return model

    @pytest.fixture
    def model_popularity_only(self, base_model: Any) -> Any:
        model = base_model
        model._github_metadata = {
            "stargazers_count": 150,  # 3 * 50 = 3 * 0.01 = 0.03
            "forks_count": 30,  # 3 * 10 = 3 * 0.01 = 0.03
            "avg_daily_commits_30d": 2.0,  # 2 * 0.05 = 0.1
        }
        return model

    @pytest.fixture
    def model_with_clone_url(self, base_model: Any) -> Any:
        model = base_model
        model._github_metadata = {
            "stargazers_count": 100,
            "forks_count": 20,
            "avg_daily_commits_30d": 1.0,
            "clone_url": "https://github.com/test/repo.git",
        }
        return model

    def test_no_github_metadata(
        self, metric: CodeQualityMetric, model_no_github: Any
    ) -> None:
        logger.info("Testing CodeQualityMetric with no GitHub metadata...")
        score: float = metric.evaluate(model_no_github)
        assert score == 0.0

    def test_popularity_only_score(
        self, metric: CodeQualityMetric, model_popularity_only: Any
    ) -> None:
        logger.info("Testing CodeQualityMetric with popularity only...")
        score: float = metric.evaluate(model_popularity_only)
        # 0.03 (stars) + 0.03 (forks) + 0.1 (commits) = 0.16
        assert score == pytest.approx(0.16, abs=0.01)

    @patch('src.metrics.CodeQualityMetric.Repo.clone_from')
    def test_clone_repository_success(
        self, mock_clone: MagicMock, metric: CodeQualityMetric
    ) -> None:
        logger.info("Testing successful repository cloning...")
        mock_clone.return_value = MagicMock()

        result: bool = metric._clone_repository(
            "https://github.com/test/repo.git", "/tmp/test"
        )

        assert result is True
        mock_clone.assert_called_once_with(
            "https://github.com/test/repo.git",
            "/tmp/test",
            depth=1
        )

    @patch('src.metrics.CodeQualityMetric.Repo.clone_from')
    def test_clone_repository_failure(
        self, mock_clone: MagicMock, metric: CodeQualityMetric
    ) -> None:
        logger.info("Testing repository cloning failure...")
        from git.exc import GitCommandError
        mock_clone.side_effect = GitCommandError("clone", "git clone failed")

        result: bool = metric._clone_repository(
            "https://github.com/test/repo.git", "/tmp/test"
        )

        assert result is False

    @patch('src.metrics.CodeQualityMetric.Repo.clone_from')
    def test_clone_repository_git_error(
        self, mock_clone: MagicMock, metric: CodeQualityMetric
    ) -> None:
        logger.info("Testing repository cloning git error...")
        from git.exc import GitError
        mock_clone.side_effect = GitError("git error")

        result: bool = metric._clone_repository(
            "https://github.com/test/repo.git", "/tmp/test"
        )

        assert result is False

    @patch('src.metrics.CodeQualityMetric.Repo.clone_from')
    def test_clone_repository_general_exception(
        self, mock_clone: MagicMock, metric: CodeQualityMetric
    ) -> None:
        logger.info("Testing repository cloning general exception...")
        mock_clone.side_effect = Exception("Unexpected error")

        result: bool = metric._clone_repository(
            "https://github.com/test/repo.git",
            "/tmp/test"
        )

        assert result is False

    def test_calculate_popularity_score(self, metric: CodeQualityMetric) -> None:
        logger.info("Testing popularity score calculation...")
        
        # Test with high stars and forks
        gh_meta = {
            "stargazers_count": 500,  # 500/50 * 0.01 = 0.1 (capped)
            "forks_count": 100        # 100/10 * 0.01 = 0.1 (capped)
        }
        score = metric._calculate_popularity_score(gh_meta)
        assert score == 0.2  # 0.1 + 0.1 = 0.2 (max)

        # Test with low stars and forks
        gh_meta = {
            "stargazers_count": 25,   # 25/50 * 0.01 = 0.005
            "forks_count": 5          # 5/10 * 0.01 = 0.005
        }
        score = metric._calculate_popularity_score(gh_meta)
        assert score == 0.01  # 0.005 + 0.005 = 0.01

    def test_calculate_commit_score(self, metric: CodeQualityMetric) -> None:
        logger.info("Testing commit score calculation...")
        
        # Test with high commit activity
        gh_meta = {"avg_daily_commits_30d": 10.0}  # 10 * 0.05 = 0.5 (capped at 0.3)
        score = metric._calculate_commit_score(gh_meta)
        assert score == 0.3  # Capped at 0.3

        # Test with low commit activity
        gh_meta = {"avg_daily_commits_30d": 2.0}   # 2 * 0.05 = 0.1
        score = metric._calculate_commit_score(gh_meta)
        assert score == 0.1

    @patch('src.metrics.CodeQualityMetric.Repo.clone_from')
    def test_full_evaluation_with_clone(
        self, mock_clone: MagicMock, metric: CodeQualityMetric, model_with_clone_url: Any
    ) -> None:
        logger.info("Testing full evaluation with repository cloning...")
        
        # Mock successful clone
        mock_clone.return_value = MagicMock()
        
        # Mock the analysis methods to return known values
        with patch.object(metric, '_evaluate_testing_quality', return_value=0.2), \
             patch.object(metric, '_evaluate_documentation', return_value=0.15):
            
            score = metric.evaluate(model_with_clone_url)
            
            # Expected: popularity (0.02 + 0.02) + commits (0.05) + test (0.2) + doc (0.15) = 0.44
            expected_score = 0.02 + 0.02 + 0.05 + 0.2 + 0.15
            assert score == pytest.approx(expected_score, abs=0.01)

    def test_evaluate_testing_quality(self, metric: CodeQualityMetric) -> None:
        logger.info("Testing testing quality evaluation...")
        
        # Create a temporary directory structure for testing
        import tempfile
        import os
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            test_dir = Path(temp_dir) / "tests"
            test_dir.mkdir()
            (test_dir / "test_file1.py").touch()
            (test_dir / "test_file2.py").touch()
            
            # Create some source files
            src_dir = Path(temp_dir) / "src"
            src_dir.mkdir()
            (src_dir / "source1.py").touch()
            (src_dir / "source2.py").touch()
            (src_dir / "source3.py").touch()
            
            # Test ratio: 2 test files / 3 source files = 0.67
            # Score: 0.67 * 0.3 = 0.2
            score = metric._evaluate_testing_quality(temp_dir)
            assert score == pytest.approx(0.2, abs=0.01)

    def test_evaluate_documentation(self, metric: CodeQualityMetric) -> None:
        logger.info("Testing documentation evaluation...")
        
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create documentation files
            (Path(temp_dir) / "README.md").touch()
            (Path(temp_dir) / "LICENSE").touch()
            (Path(temp_dir) / "CONTRIBUTING.md").touch()
            
            # Expected: README (0.05) + LICENSE (0.05) + CONTRIBUTING (0.10) = 0.20
            score = metric._evaluate_documentation(temp_dir)
            assert score == 0.20

    def test_count_test_files(self, metric: CodeQualityMetric) -> None:
        logger.info("Testing test file counting...")
        
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test directory structure
            tests_dir = Path(temp_dir) / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_file1.py").touch()
            (tests_dir / "test_file2.py").touch()
            
            test_dir = Path(temp_dir) / "test"
            test_dir.mkdir()
            (test_dir / "test_file3.py").touch()
            
            count = metric._count_test_files(temp_dir)
            assert count == 3

    def test_count_source_files(self, metric: CodeQualityMetric) -> None:
        logger.info("Testing source file counting...")
        
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create various source files
            (Path(temp_dir) / "main.py").touch()
            (Path(temp_dir) / "utils.js").touch()
            (Path(temp_dir) / "config.ts").touch()
            (Path(temp_dir) / "README.md").touch()  # Not a source file
            (Path(temp_dir) / "data.txt").touch()   # Not a source file
            
            count = metric._count_source_files(temp_dir)
            assert count == 3  # Only .py, .js, .ts files

    def test_evaluation_with_clone_failure(
        self, metric: CodeQualityMetric, model_with_clone_url: Any
    ) -> None:
        logger.info("Testing evaluation when clone fails...")
        
        # Mock clone failure
        with patch.object(metric, '_clone_repository', return_value=False):
            score = metric.evaluate(model_with_clone_url)
            
            # Should return only popularity + commit scores
            # popularity: 0.02 + 0.02 = 0.04, commits: 0.05, total: 0.09
            expected_score = 0.02 + 0.02 + 0.05
            assert score == pytest.approx(expected_score, abs=0.01)

    def test_score_capping_at_one(self, metric: CodeQualityMetric) -> None:
        logger.info("Testing score capping at 1.0...")
        
        # Create a model with very high scores that would exceed 1.0
        model = MagicMock()
        model._github_metadata = {
            "stargazers_count": 10000,  # Would give 0.1
            "forks_count": 1000,        # Would give 0.1
            "avg_daily_commits_30d": 20.0,  # Would give 1.0
            "clone_url": "https://github.com/test/repo.git"
        }
        
        # Mock successful clone and high analysis scores
        with patch.object(metric, '_clone_repository', return_value=True), \
             patch.object(metric, '_evaluate_testing_quality', return_value=0.5), \
             patch.object(metric, '_evaluate_documentation', return_value=0.5):
            
            score = metric.evaluate(model)
            assert score == 1.0  # Should be capped at 1.0
