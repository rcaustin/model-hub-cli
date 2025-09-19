from abc import ABC


class BaseMetricTest(ABC):

    # --- Generic Tests ---
    def run_metric_test(self, metric, model, expected_score):
        score = metric.evaluate(model)
        assert score == expected_score


