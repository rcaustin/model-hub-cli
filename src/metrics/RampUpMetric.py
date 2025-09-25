import re
from typing import Optional

from loguru import logger

from src.Interfaces import ModelData
from src.Metric import Metric

try:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
except Exception:
    HfApi = None  # type: ignore
    HfHubHTTPError = Exception  # type: ignore
    hf_hub_download = None  # type: ignore


class RampUpMetric(Metric):
    """
    Ramp-Up Time checklist (each = +0.20):
      1) README present
      2) Install instructions present
      3) Usage example present
      4) Dataset info present (README or model_index.json)
      5) Training script/command present

    If neither README.md nor model_index.json is present, return 0.0.
    """

    # ----------------- simple heuristics -----------------
    @staticmethod
    def _has_install_instructions(text: str) -> bool:
        t = text.lower()
        patterns = [
            r"\bpip\s+install\b",
            r"\bpip3\s+install\b",
            r"\bconda\s+install\b",
            r"requirements\.txt",
            r"\bpoetry\s+add\b",
            r"\bdocker\s+(pull|run|build)\b",
        ]
        return any(re.search(p, t) for p in patterns)

    @staticmethod
    def _has_usage_example(text: str) -> bool:
        # Look for code fences or typical import/usage cues
        t = text
        cues = [
            r"```(?:python|py|bash|sh)?\s",  # fenced code blocks
            r"\bfrom\s+\w+\s+import\s+\w+",
            r"\bimport\s+\w+",
            r"\bpipeline\(",
            r"\bAuto(Model|Tokenizer|Processor)\b",
            r"\bUsage\b",
            r"\bExample(s)?\b",
        ]
        return any(re.search(p, t, flags=re.I) for p in cues)

    @staticmethod
    def _has_dataset_info(readme: str, model_index_json: Optional[str]) -> bool:
        # README hints
        readme_hit = bool(re.search(r"\bdataset(s)?\b", readme, flags=re.I)) or \
                     bool(re.search(r"\btrained\s+on\b", readme, flags=re.I)) or \
                     bool(
                        re.search(r"\bdata\s+(source|collection)\b", readme, flags=re.I)
                    )
        # model_index.json hints
        idx_hit = False
        if model_index_json:
            idx_hit = (
                bool(re.search(r'"dataset(s)?"\s*:', model_index_json, flags=re.I)) or
                bool(re.search(r'"trained_on"\s*:', model_index_json, flags=re.I)) or
                bool(re.search(r'"evaluation"\s*:', model_index_json, flags=re.I))
            )
        return readme_hit or idx_hit

    @staticmethod
    def _has_training_script_or_cmd(text: str) -> bool:
        t = text.lower()
        patterns = [
            r"\btrain\.py\b",
            r"\bpython\s+train(\.py)?\b",
            r"\baccelerate\s+launch\b",
            r"\btrainer\b",            # HF Trainer mention
            r"\bfine[- ]?tune\b",
            r"\bscripts?\/?train",     # scripts/train*.sh, etc.
        ]
        return any(re.search(p, t) for p in patterns)

    # ----------------- main entry -----------------
    def evaluate(self, model: ModelData) -> float:
        # Resolve a HF model repo id from ModelData
        repo_id = getattr(model, "model_id", None) or getattr(model, "model_url", None)
        if not repo_id:
            # Many projects put it in metadata
            meta = getattr(model, "metadata", {}) or {}
            repo_id = meta.get("model") or meta.get("huggingface_model")

        if not repo_id:
            logger.error("RampUpMetric: No model id/url provided on ModelData.")
            return 0.0

        if HfApi is None or hf_hub_download is None:
            logger.error(
                "RampUpMetric: huggingface_hub not installed."
            )
            return 0.0

        api = HfApi()

        # --- Pull README (model repo) ---
        readme_md = ""
        try:
            readme_md = api.get_repo_readme(repo_id=repo_id, repo_type="model") or ""
        except HfHubHTTPError:
            readme_md = ""
        except Exception:
            readme_md = ""

        # --- Pull model_index.json if present ---
        model_index_text: Optional[str] = None
        try:
            files = api.list_repo_files(repo_id=repo_id, repo_type="model") or []
            if "model_index.json" in files:
                fp = hf_hub_download(
                    repo_id=repo_id, filename="model_index.json", repo_type="model"
                )
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        model_index_text = f.read()
                except Exception:
                    model_index_text = None
        except HfHubHTTPError:
            model_index_text = None
        except Exception:
            model_index_text = None

        # If neither README nor model_index.json exists, short-circuit to 0.0
        if not readme_md and not model_index_text:
            logger.debug(
                f"RampUpMetric: {repo_id} has no README or model_index.json → score 0.0"
            )
            return 0.0

        # Checklist booleans
        has_readme = bool(readme_md.strip())
        has_install = self._has_install_instructions(readme_md)
        has_usage = self._has_usage_example(readme_md)
        has_dataset = self._has_dataset_info(readme_md, model_index_text)
        has_training = self._has_training_script_or_cmd(readme_md)

        # Scoring: +0.20 each
        score = (
            0.20 * has_readme +
            0.20 * has_install +
            0.20 * has_usage +
            0.20 * has_dataset +
            0.20 * has_training
        )

        logger.debug(
            "RampUpMetric: repo={repo} score={score:.2f} "
            " | README={readme} install={install} "
            "usage={usage} dataset={dataset} training={training}",
            repo=repo_id,
            score=score,
            readme=has_readme,
            install=has_install,
            usage=has_usage,
            dataset=has_dataset,
            training=has_training,
        )
        return float(score)
