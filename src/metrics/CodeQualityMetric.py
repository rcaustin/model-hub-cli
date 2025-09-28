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

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from src.ModelData import ModelData
from src.Metric import Metric


class CodeQualityMetric(Metric):
    def evaluate(self, model: ModelData) -> float:
        logger.info("Evaluating CodeQualityMetric...")

        gh_meta: Optional[Dict[str, Any]] = model.github_metadata
        if not gh_meta:
            logger.warning("No GitHub metadata available for code quality evaluation")
            return 0.0

        # Code Popularity (+0.2)
        stars: int = gh_meta.get("stargazers_count", 0)
        forks: int = gh_meta.get("forks_count", 0)
        popularity_score: float = (
            min((stars // 50) * 0.01, 0.1) + min((forks // 10) * 0.01, 0.1)
        )
        logger.debug(
            f"Code popularity: {stars} stars, {forks} forks → score: {popularity_score}"
        )

        # Clone once and get both testing (+0.3) and documentation (+0.2) scores
        test_score: float
        doc_score: float
        test_score, doc_score = self._clone_and_analyze(gh_meta)

        # Commit Frequency (+0.3)
        daily_commits: float = gh_meta.get("avg_daily_commits_30d", 0)
        commit_score: float = min(daily_commits * 0.05, 0.3)
        logger.debug(
            f"Commit frequency: {daily_commits} daily commits → score: {commit_score}"
        )

        # Total score capped at 1.0
        total_score: float = popularity_score + test_score + commit_score + doc_score
        final_score: float = min(total_score, 1.0)

        logger.info(
            f"CodeQualityMetric final score: {final_score} "
            f"(popularity: {popularity_score}, test: {test_score}, "
            f"commit: {commit_score}, doc: {doc_score})"
        )

        return final_score

    def _clone_and_analyze(self, gh_meta: Dict[str, Any]) -> Tuple[float, float]:
        """Clone repo once and return (test_score, doc_score)."""
        clone_url: Optional[str] = gh_meta.get("clone_url")
        if not clone_url:
            logger.warning("No clone URL available in GitHub metadata")
            return 0.0, 0.0

        logger.info(f"Starting repository analysis for: {clone_url}")

        with tempfile.TemporaryDirectory() as temp_dir:
            logger.debug(f"Created temporary directory: {temp_dir}")

            if self._clone_repository(clone_url, temp_dir):
                logger.info("Repository cloned successfully, analyzing files...")

                # Robust Test Suite (+0.3)
                test_files: int = self._count_test_files(temp_dir)
                source_files: int = self._count_source_files(temp_dir)
                test_score: float = 0.0
                if source_files > 0:
                    test_ratio: float = min(test_files / source_files, 1.0)
                    test_score = test_ratio * 0.3
                    logger.debug(
                        f"Test analysis: {test_files} test files, "
                        f"{source_files} source files "
                        f"→ ratio: {test_ratio:.2f}, score: {test_score}"
                    )
                else:
                    logger.warning("No source files found in repository")

                # Documentation (+0.2)
                doc_score: float = self._evaluate_documentation(temp_dir)

                logger.info(
                    f"Repository analysis complete. Test score: {test_score}, "
                    f"Doc score: {doc_score}"
                )
                return test_score, doc_score
            else:
                logger.error("Failed to clone repository")
                return 0.0, 0.0

    def _clone_repository(self, clone_url: str, temp_dir: str) -> bool:
        """Clone repository to temp directory. Returns True if successful."""
        logger.debug(f"Cloning repository: {clone_url} → {temp_dir}")

        try:
            # TODO: Should not be using the "git" shell command!!
            subprocess.run(
                [
                    'git', 'clone',
                    '--depth', '1',
                    clone_url,
                    temp_dir
                ],
                check=True,
                capture_output=True,
                timeout=30,
                text=True
            )
            logger.debug("Git clone completed successfully")
            return True
        except subprocess.TimeoutExpired:
            logger.error(f"Git clone timed out after 30 seconds for {clone_url}")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Git clone failed for {clone_url}: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during git clone: {e}")
            return False

    def _count_test_files(self, repo_path: str) -> int:
        """Count test files in test directories."""
        logger.debug("Counting test files...")

        test_patterns: List[str] = [
            'tests/**/*.py',
            'test/**/*.py'
        ]

        count: int = 0
        repo: Path = Path(repo_path)

        for pattern in test_patterns:
            pattern_matches: List[Path] = list(repo.glob(pattern))
            pattern_count: int = len(pattern_matches)
            count += pattern_count
            if pattern_count > 0:
                logger.debug(f"Pattern '{pattern}' matched {pattern_count} files")

        logger.debug(f"Total test files found: {count}")
        return count

    def _count_source_files(self, repo_path: str) -> int:
        """Count source files excluding test/doc directories."""
        logger.debug("Counting source files...")

        repo: Path = Path(repo_path)
        exclude_dirs: set[str] = {
            'tests', 'test', 'docs', 'examples', '.git', '__pycache__'
        }

        count: int = 0
        excluded_count: int = 0

        for py_file in repo.rglob('*.py'):
            if any(excl in py_file.parts for excl in exclude_dirs):
                excluded_count += 1
                continue
            count += 1

        logger.debug(f"Source files found: {count}, excluded: {excluded_count}")
        return count

    def _evaluate_documentation(self, repo_path: str) -> float:
        """Check for documentation files in cloned repo."""
        logger.debug("Evaluating documentation files...")

        repo: Path = Path(repo_path)

        # Check for license file
        has_license: bool = (
            any(repo.glob("LICENSE*")) or any(repo.glob("license*"))
        )

        # Check for README
        has_readme: bool = (
            any(repo.glob("README*")) or any(repo.glob("readme*"))
        )

        # Check for CONTRIBUTING
        has_contributing: bool = (
            any(repo.glob("CONTRIBUTING*")) or any(repo.glob("contributing*"))
        )

        doc_score: float = (
            (0.05 if has_license else 0.0)
            + (0.05 if has_readme else 0.0)
            + (0.10 if has_contributing else 0.0)
        )

        found_docs: List[str] = []
        if has_license:
            found_docs.append("LICENSE")
        if has_readme:
            found_docs.append("README")
        if has_contributing:
            found_docs.append("CONTRIBUTING")

        logger.debug(f"Documentation found: {found_docs} → score: {doc_score}")
        return doc_score
