import pytest

from src.metrics.AvailabilityMetric import AvailabilityMetric
from tests.conftest import StubModelData


@pytest.mark.parametrize(
    "code_link, dataset_link, github_meta, dataset_meta, expected_score",
    [
        (
            "https://github.com/org/repo",
            "https://huggingface.co/datasets/org/data",
            {"stars": 123},
            {"name": "dataset"},
            1.0,
        ),
        (
            "https://github.com/org/repo",
            "https://huggingface.co/datasets/org/data",
            {"stars": 999},
            {},
            0.5,
        ),
        (
            "https://github.com/org/repo",
            "https://huggingface.co/datasets/org/data",
            {},
            {"name": "dataset"},
            0.5,
        ),
        (
            "https://github.com/org/repo",
            "https://huggingface.co/datasets/org/data",
            {},
            {},
            0.0,
        ),
        (
            "https://github.com/org/repo",
            None,
            {"stars": 42},
            {},
            1.0,
        ),
        (
            None,
            "https://huggingface.co/datasets/org/data",
            {},
            {"name": "dataset"},
            1.0,
        ),
        (
            None,
            None,
            {},
            {},
            0.0,
        ),
    ],
)
def test_availability_metric_scores(
    code_link,
    dataset_link,
    github_meta,
    dataset_meta,
    expected_score,
):
    model = StubModelData(
        modelLink="https://huggingface.co/org/model",
        codeLink=code_link,
        datasetLink=dataset_link,
    )
    model.github_metadata = github_meta
    model.dataset_metadata = dataset_meta
    model.hf_metadata = {}
    score = AvailabilityMetric().evaluate(model)
    assert pytest.approx(score, 0.01) == expected_score


def test_availability_metric_none_metadata():
    model = StubModelData(
        modelLink="https://huggingface.co/org/model",
        codeLink="https://github.com/org/repo",
        datasetLink="https://huggingface.co/datasets/org/data",
    )
    model.github_metadata = None
    model.dataset_metadata = None
    model.hf_metadata = {}
    score = AvailabilityMetric().evaluate(model)
    assert score == 0.0
