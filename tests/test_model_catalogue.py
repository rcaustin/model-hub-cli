import json

import pytest

from src.Metric import Metric
from src.Model import Model
from src.ModelCatalogue import ModelCatalogue


class StubMetric(Metric):
    """Metric that returns a constant float value."""
    def __init__(self, name="StubMetric", value=0.42):
        self._name = name
        self._value = value

    def evaluate(self, model: Model) -> float:
        return self._value


class StubMetric1(StubMetric):
    pass


class StubMetric2(StubMetric):
    pass


class DictMetric(Metric):
    """Metric that returns a dictionary with an 'average' key."""
    def evaluate(self, model: Model) -> dict[str, float]:
        return {"average": 0.75, "other": 0.9}


def test_add_model_adds_to_internal_list(sample_model):
    catalogue = ModelCatalogue()
    assert len(catalogue.models) == 0

    catalogue.addModel(sample_model)
    assert len(catalogue.models) == 1
    assert catalogue.models[0] == sample_model


def test_evaluate_models_runs_all_metrics(sample_model):
    catalogue = ModelCatalogue()

    # Override the default metrics with two stubs
    catalogue.metrics = [
        StubMetric1("StubMetric1", 0.3),
        StubMetric2("StubMetric2", 0.7)
    ]

    catalogue.addModel(sample_model)
    catalogue.evaluateModels()

    model = catalogue.models[0]
    assert "StubMetric1" in model.evaluations
    assert "StubMetric2" in model.evaluations
    assert "NetScore" in model.evaluations


def test_generate_report_format(sample_model):
    catalogue = ModelCatalogue()

    # Override metrics with predictable output
    catalogue.metrics = [
        StubMetric("LicenseMetric", 1.0),
        StubMetric("BusFactorMetric", 0.8),
        StubMetric("RampUpMetric", 0.6),
        StubMetric("AvailabilityMetric", 0.7),
        StubMetric("DatasetQualityMetric", 0.6),
        StubMetric("CodeQualityMetric", 0.65),
        StubMetric("PerformanceClaimsMetric", 0.55),
        DictMetric(),  # SizeMetric returning {"average": 0.75}
    ]

    catalogue.addModel(sample_model)
    catalogue.evaluateModels()
    report = catalogue.generateReport()

    # Check format: sections split by -----, each line is a JSON object
    report_sections = report.split("\n-----\n")
    assert len(report_sections) == 1  # 1 model only

    lines = report_sections[0].strip().splitlines()
    for line in lines:
        try:
            entry = json.loads(line)
            assert isinstance(entry, dict)
            assert len(entry) == 1  # each line is a single key-value pair
        except json.JSONDecodeError:
            pytest.fail("NDJSON line is not valid JSON")

    # Check expected keys are present
    keys = [json.loads(line) for line in lines]
    keys = {k for d in keys for k in d.keys()}
    expected_keys = {
        "name", "category", "net_score", "net_score_latency",
        "ramp_up_time", "ramp_up_time_latency",
        "bus_factor", "bus_factor_latency",
        "performance_claims", "performance_claims_latency",
        "license", "license_latency",
        "size_score", "size_score_latency",
        "availability_score", "availability_score_latency",
        "dataset_quality", "dataset_quality_latency",
        "code_quality", "code_quality_latency",
    }

    assert expected_keys.issubset(keys)
