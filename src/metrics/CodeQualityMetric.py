"""
CodeQualityMetric.py
====================

Evaluates code quality based on GitHub repository analysis.

Score Breakdown (Total: 1.0)
----------------------------
- Code Popularity (stars, forks): up to 0.2
- Test Suite Coverage (test files vs source files): up to 0.3
- Commit Frequency (avg daily commits): up to 0.3
- Documentation presence (LICENSE, README, CONTRIBUTING): up to 0.2

Requirements
------------
- Access to GitHub metadata with repository stats and clone URL
- Git installed and accessible via command line for cloning
- Python environment with pathlib and subprocess modules

Limitations
-----------
- Cloning repositories can be slow or fail due to network or access issues
- Test file detection relies on naming conventions and folder structure
- Documentation score is basic and based on file presence only
- Git commands executed via subprocess, which may raise security concerns
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from git import Repo
from git.exc import GitCommandError, GitError
from loguru import logger

from src.ModelData import ModelData
from src.Metric import Metric


class CodeQualityMetric(Metric):
    """
    Evaluates code quality based on:
    1. Documentation coverage (README, docstrings, comments)
    2. Code structure and organization
    3. Repository popularity (stars, forks)
    4. Code complexity and maintainability
    """

    def __init__(self):
        self.temp_dirs: List[str] = []

    def evaluate(self, model: ModelData) -> float:
        """
        Evaluate code quality for the given model.

        Args:
            model: ModelData object containing URLs and metadata

        Returns:
            float: Code quality score from 0.0 to 1.0
        """
        logger.info("Evaluating CodeQualityMetric...")

        # Check if we have GitHub metadata
        if not hasattr(model, '_github_metadata') or not model._github_metadata:
            logger.info("CodeQualityMetric: No GitHub metadata found -> 0.0")
            return 0.0

        gh_meta = model._github_metadata
        clone_url = gh_meta.get('clone_url')

        # Calculate popularity score (always available)
        popularity_score = self._calculate_popularity_score(gh_meta)
        
        # Add commit activity score
        commit_score = self._calculate_commit_score(gh_meta)
        
        # If no clone URL, return only popularity + commit scores
        if not clone_url:
            total_score = popularity_score + commit_score
            logger.info(f"CodeQualityMetric: No clone URL, popularity + commits -> {total_score:.3f}")
            return min(total_score, 1.0)

        # Full analysis with repository cloning
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                if self._clone_repository(clone_url, temp_dir):
                    test_score, doc_score = self._clone_and_analyze(temp_dir)
                    total_score = popularity_score + commit_score + test_score + doc_score
                    logger.info(f"CodeQualityMetric: Full analysis -> {total_score:.3f}")
                    return min(total_score, 1.0)
                else:
                    logger.warning("CodeQualityMetric: Clone failed, popularity + commits only")
                    total_score = popularity_score + commit_score
                    return min(total_score, 1.0)
        except Exception as e:
            logger.error(f"CodeQualityMetric: Error during evaluation: {e}")
            total_score = popularity_score + commit_score
            return min(total_score, 1.0)

    def _calculate_popularity_score(self, gh_meta: Dict[str, Any]) -> float:
        """Calculate popularity score from GitHub metadata."""
        stars = gh_meta.get('stargazers_count', 0)
        forks = gh_meta.get('forks_count', 0)
        
        # 0.01 per 50 stars, max 0.1
        star_score = min(stars / 50 * 0.01, 0.1)
        # 0.01 per 10 forks, max 0.1  
        fork_score = min(forks / 10 * 0.01, 0.1)
        
        return star_score + fork_score

    def _calculate_commit_score(self, gh_meta: Dict[str, Any]) -> float:
        """Calculate commit activity score."""
        avg_commits = gh_meta.get('avg_daily_commits_30d', 0)
        # 0.05 per daily commit, max 0.3
        return min(avg_commits * 0.05, 0.3)

    def _clone_repository(self, clone_url: str, temp_dir: str) -> bool:
        """Clone repository to temp directory using GitPython. Returns True if successful."""
        logger.debug(f"Cloning repository: {clone_url} → {temp_dir}")

        try:
            # Use GitPython to clone the repository
            # Note: GitPython doesn't have a direct equivalent to --depth 1,
            # but we can use shallow clone by setting depth=1
            Repo.clone_from(
                clone_url, 
                temp_dir,
                depth=1  # Shallow clone equivalent to --depth 1
            )
            logger.debug("Git clone completed successfully")
            return True
        except GitCommandError as e:
            logger.error(f"Git clone failed for {clone_url}: {e}")
            return False
        except GitError as e:
            logger.error(f"Git error during clone for {clone_url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during git clone: {e}")
            return False

    def _clone_and_analyze(self, repo_path: str) -> tuple[float, float]:
        """Analyze cloned repository and return (test_score, doc_score)."""
        test_score = self._evaluate_testing_quality(repo_path)
        doc_score = self._evaluate_documentation(repo_path)
        return test_score, doc_score

    def _evaluate_testing_quality(self, repo_path: str) -> float:
        """Evaluate testing quality based on test-to-source file ratio."""
        test_files = self._count_test_files(repo_path)
        source_files = self._count_source_files(repo_path)
        
        if source_files == 0:
            return 0.0
            
        ratio = test_files / source_files
        # Full score (0.3) when test files >= source files
        return min(ratio * 0.3, 0.3)

    def _count_test_files(self, repo_path: str) -> int:
        """Count test files in the repository."""
        test_patterns = ['tests/**/*.py', 'test/**/*.py']
        count = 0
        
        for pattern in test_patterns:
            count += len(list(Path(repo_path).glob(pattern)))
            
        return count

    def _count_source_files(self, repo_path: str) -> int:
        """Count source files in the repository."""
        source_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala'}
        
        count = 0
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories and common non-code directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'node_modules', 'venv', '__pycache__', 'build', 'dist', 'tests', 'test'}]
            
            for file in files:
                if Path(file).suffix.lower() in source_extensions:
                    count += 1
        
        return count


def _evaluate_documentation(self, repo_path: str) -> float:
    """Evaluate documentation quality."""
    logger.debug("Evaluating documentation in: {}", repo_path)
    score = 0.0
    found_docs = []

    # Check for LICENSE file
    if any(Path(repo_path).glob('LICENSE*')) or any(Path(repo_path).glob('license*')):
        score += 0.05
        found_docs.append("LICENSE")

    # Check for README file
    if any(Path(repo_path).glob('README*')) or any(Path(repo_path).glob('readme*')):
        score += 0.05
        found_docs.append("README")

    # Check for CONTRIBUTING file
    if any(Path(repo_path).glob('CONTRIBUTING*')) or any(Path(repo_path).glob('contributing*')):
        score += 0.10
        found_docs.append("CONTRIBUTING")

    logger.debug("Found docs: {} → doc score: {}", found_docs, score)
    return score

