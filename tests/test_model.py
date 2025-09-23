from src.Metric import Metric
from src.Model import Model


class DummyMetric(Metric):
    """A stub metric that returns fixed score and takes no time."""
    def evaluate(self, model) -> float:
        return 0.5


class ComplexDummyMetric(Metric):
    """Returns a dict result simulating a metric with sub-scores."""
    def evaluate(self, model) -> dict[str, float]:
        return {"average": 0.7, "detail_1": 0.8, "detail_2": 0.6}


def test_model_initialization_and_url_classification(sample_urls):
    model = Model(sample_urls)

    # Check that URLs were classified correctly
    assert model.modelLink.startswith("https://huggingface.co/")
    assert model.codeLink.startswith("https://github.com/")
    assert model.datasetLink.startswith("https://huggingface.co/datasets/")


def test_get_category_string(sample_urls):
    model = Model(sample_urls)
    category = model.getCategory()
    assert category == "MODEL"


def test_evaluation_and_latency_storage(sample_urls):
    model = Model(sample_urls)
    metric = DummyMetric()

    model.evaluate(metric)

    assert "DummyMetric" in model.evaluations
    assert model.evaluations["DummyMetric"] == 0.5

    assert "DummyMetric" in model.evaluationsLatency
    assert isinstance(model.evaluationsLatency["DummyMetric"], float)


def test_get_score_and_latency_methods_with_float(sample_urls):
    model = Model(sample_urls)
    metric = DummyMetric()
    model.evaluate(metric)

    assert model.get_score("DummyMetric") == 0.5
    assert isinstance(model.get_latency("DummyMetric"), int)


def test_get_score_with_dict(sample_urls):
    model = Model(sample_urls)
    metric = ComplexDummyMetric()
    model.evaluate(metric)

    assert model.get_score("ComplexDummyMetric") == 0.7  # "average" from dict
    assert model.get_score("NonExistentMetric") == 0.0


def test_compute_net_score_with_stub_metrics(sample_urls):
    model = Model(sample_urls)

    # Preload fake evaluation scores for all metrics in NetScore
    model.evaluations = {
        "LicenseMetric": 1.0,
        "SizeMetric": {"average": 0.6},
        "RampUpMetric": 0.7,
        "BusFactorMetric": 0.8,
        "AvailabilityMetric": 0.9,
        "DatasetQualityMetric": 0.6,
        "CodeQualityMetric": 0.7,
        "PerformanceClaimsMetric": 0.5
    }

    net_score = model.computeNetScore()

    # Manually compute expected NetScore
    expected_weighted = (
        0.2 * 0.6 +   # Size
        0.3 * 0.7 +   # Ramp-Up
        0.1 * 0.8 +   # Bus Factor
        0.1 * 0.9 +   # Availability
        0.1 * 0.6 +   # Dataset Quality
        0.1 * 0.7 +   # Code Quality
        0.1 * 0.5     # Performance Claims
    )
    expected_score = 1.0 * expected_weighted

    assert abs(net_score - expected_score) < 1e-6
    assert model.evaluations["NetScore"] == expected_score
    assert model.evaluationsLatency["NetScore"] == 0.0


def test_evaluate_all_runs_multiple_metrics(sample_urls):
    model = Model(sample_urls)

    metrics = [DummyMetric(), ComplexDummyMetric()]
    model.evaluate_all(metrics)

    assert "DummyMetric" in model.evaluations
    assert "ComplexDummyMetric" in model.evaluations
    assert "NetScore" in model.evaluations
