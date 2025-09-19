from loguru import logger
import pytest
from src.metrics.LicenseMetric import LicenseMetric
from tests.metric_tests.base_metric_test import BaseMetricTest


class TestLicenseMetric(BaseMetricTest):
    @pytest.fixture
    def metric(self):
        return LicenseMetric()

    @pytest.fixture
    def hf_model_mit(self, base_model):
        model = base_model
        model._hf_metadata = {"cardData": {"license": "MIT"}}
        return model

    @pytest.fixture
    def hf_model_unknown(self, base_model):
        model = base_model
        model._hf_metadata = {"cardData": {"license": "unknown"}}
        return model

    @pytest.fixture
    def github_model_gpl(self, base_model):
        model = base_model
        model._github_metadata = {"license": {"spdx_id": "GPL-3.0"}}
        return model

    # --- Tests ---

    def test_hf_model_mit(self, metric, hf_model_mit):
        logger.info("Testing HF model with MIT license...")
        score = metric.evaluate(hf_model_mit)
        assert score == 1.0

    def test_hf_model_unknown(self, metric, hf_model_unknown):
        logger.info("Testing HF model with unknown license...")
        score = metric.evaluate(hf_model_unknown)
        assert score == 0.5

    def test_github_model_gpl(self, metric, github_model_gpl):
        logger.info("Testing GitHub model with GPL license...")
        score = metric.evaluate(github_model_gpl)
        assert score == 0.0
