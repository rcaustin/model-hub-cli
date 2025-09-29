import pytest
from unittest.mock import MagicMock

from src.metrics.RampUpMetric import RampUpMetric
from tests.conftest import StubModelData


class TestRampUpMetric:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.metric = RampUpMetric()
        # Patch LLMClient methods
        self.metric.llm.send_prompt = MagicMock()
        self.metric.llm.extract_score = MagicMock()

    def test_evaluate_with_both_readme_and_model_index(self):
        # Arrange
        readme = "This is a README"
        model_index = '{"name": "model_index"}'
        model = StubModelData(
            modelLink="https://huggingface.co/org/model",
            codeLink=None,
            datasetLink=None,
            _hf_metadata={"readme": readme, "model_index": model_index},
        )
        expected_score = 0.75

        self.metric.llm.send_prompt.return_value = "0.75\nDetails about docs"
        self.metric.llm.extract_score.return_value = expected_score

        # Act
        score = self.metric.evaluate(model)

        # Assert
        assert score == expected_score
        # Check that the prompt contains README and model_index content
        prompt_arg = self.metric.llm.send_prompt.call_args[0][0]
        assert readme in prompt_arg
        assert model_index in prompt_arg
        self.metric.llm.extract_score.assert_called_once()

    def test_evaluate_with_only_readme(self):
        readme = "Only README content here"
        model = StubModelData(
            modelLink="https://huggingface.co/org/model",
            codeLink=None,
            datasetLink=None,
            _hf_metadata={"readme": readme},
        )
        expected_score = 0.6

        self.metric.llm.send_prompt.return_value = "0.6\nSome explanation"
        self.metric.llm.extract_score.return_value = expected_score

        score = self.metric.evaluate(model)

        assert score == expected_score
        prompt_arg = self.metric.llm.send_prompt.call_args[0][0]
        assert readme in prompt_arg
        self.metric.llm.extract_score.assert_called_once()

    def test_evaluate_with_only_model_index(self):
        model_index = '{"some": "index"}'
        model = StubModelData(
            modelLink="https://huggingface.co/org/model",
            codeLink=None,
            datasetLink=None,
            _hf_metadata={"model_index": model_index},
        )
        expected_score = 0.4

        self.metric.llm.send_prompt.return_value = "0.4\nExplanation text"
        self.metric.llm.extract_score.return_value = expected_score

        score = self.metric.evaluate(model)

        assert score == expected_score
        prompt_arg = self.metric.llm.send_prompt.call_args[0][0]
        assert model_index in prompt_arg
        self.metric.llm.extract_score.assert_called_once()

    def test_evaluate_with_no_docs_returns_zero(self):
        model = StubModelData(
            modelLink="https://huggingface.co/org/model",
            codeLink=None,
            datasetLink=None,
            _hf_metadata={},  # No readme or model_index
        )

        # Should return 0.0 immediately without calling LLM
        score = self.metric.evaluate(model)

        assert score == 0.0
        self.metric.llm.send_prompt.assert_not_called()
        self.metric.llm.extract_score.assert_not_called()
