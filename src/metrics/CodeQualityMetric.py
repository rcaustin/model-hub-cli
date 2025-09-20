import os
import subprocess
import tempfile
from pathlib import Path
from src.Interfaces import ModelData
from src.Metric import Metric


class CodeQualityMetric(Metric):
    def evaluate(self, model: ModelData) -> float:
        gh_meta = model.github_metadata
        if gh_meta:
            # Code Popularity (+0.2)
            stars = gh_meta.get("stargazers_count", 0)
            forks = gh_meta.get("forks_count", 0)
            popularity_score = min((stars // 50) * 0.01, 0.1) + min((forks // 10) * 0.01, 0.1)

            # Robust Test Suite (+0.3) - NOW WITH CLONING
            test_score = self._evaluate_test_suite_by_cloning(gh_meta)

            # Commit Frequency (+0.3)
            daily_commits = gh_meta.get("avg_daily_commits_30d", 0)
            commit_score = min(daily_commits * 0.05, 0.3)

            # Documentation (+0.2) - NOW WITH CLONING
            doc_score = self._evaluate_documentation_by_cloning(gh_meta)

            # Total score capped at 1.0
            total_score = popularity_score + test_score + commit_score + doc_score
            return min(total_score, 1.0)

    def _evaluate_test_suite_by_cloning(self, gh_meta: dict) -> float:
        """Clone repo and count test files vs source files."""
        clone_url = gh_meta.get("clone_url")
        if not clone_url:
            return 0.0
            
        with tempfile.TemporaryDirectory() as temp_dir:
            if self._clone_repository(clone_url, temp_dir):
                test_files = self._count_test_files(temp_dir)
                source_files = self._count_source_files(temp_dir)
                
                if source_files == 0:
                    return 0.0
                    
                test_ratio = min(test_files / source_files, 1.0)
                return test_ratio * 0.3
            else:
                return 0.0

    def _evaluate_documentation_by_cloning(self, gh_meta: dict) -> float:
        """Clone repo and check for documentation files."""
        clone_url = gh_meta.get("clone_url")
        if not clone_url:
            return 0.0
            
        with tempfile.TemporaryDirectory() as temp_dir:
            if self._clone_repository(clone_url, temp_dir):
                repo = Path(temp_dir)
                
                # Check for license file
                has_license = any(repo.glob("LICENSE*")) or any(repo.glob("license*"))
                
                # Check for README
                has_readme = any(repo.glob("README*")) or any(repo.glob("readme*"))
                
                # Check for CONTRIBUTING
                has_contributing = any(repo.glob("CONTRIBUTING*")) or any(repo.glob("contributing*"))
                
                doc_score = (0.05 if has_license else 0.0) + \
                            (0.05 if has_readme else 0.0) + \
                            (0.10 if has_contributing else 0.0)
                
                return doc_score
            else:
                return 0.0

    def _clone_repository(self, clone_url: str, temp_dir: str) -> bool:
        """Clone repository to temp directory. Returns True if successful."""
        try:
            subprocess.run(
                [
                    'git', 'clone',
                    '--depth', '1',
                    clone_url,
                    temp_dir
                ],
                check=True,
                capture_output=True,
                timeout=30
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def _count_test_files(self, repo_path: str) -> int:
        """Count test files using common patterns."""
        test_patterns = [
            '**/test_*.py',
            '**/*_test.py', 
            '**/tests/**/*.py',
            '**/test/**/*.py'
        ]

        count = 0
        repo = Path(repo_path)
        for pattern in test_patterns:
            count += len(list(repo.glob(pattern)))
        return count

    def _count_source_files(self, repo_path: str) -> int:
        """Count source files excluding test/doc directories."""
        repo = Path(repo_path)
        exclude_dirs = {'tests', 'test', 'docs', 'examples', '.git', '__pycache__'}

        count = 0
        for py_file in repo.rglob('*.py'):
            if any(excl in py_file.parts for excl in exclude_dirs):
                continue
            count += 1
        return count
