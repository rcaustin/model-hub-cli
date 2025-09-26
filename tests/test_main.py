import sys
from unittest.mock import MagicMock, patch
import pytest

import src.main as main


@pytest.fixture
def mock_model_catalogue():
    with patch("src.main.ModelCatalogue") as MockCatalogue:
        instance = MockCatalogue.return_value
        instance.addModel = MagicMock()
        instance.evaluateModels = MagicMock()
        instance.generateReport = MagicMock(return_value="Report Output")
        yield instance


@pytest.fixture
def mock_model():
    with patch("src.main.Model") as MockModel:
        yield MockModel


@patch("src.main.validate_github_token", return_value=True)
def test_run_catalogue_success(
    mock_validate, tmp_path, mock_model_catalogue, mock_model
):
    # Create a temp file with valid lines (3 comma-separated URLs)
    file_content = "url1,url2,url3\nurl4,url5,url6\n"
    file_path = tmp_path / "input.txt"
    file_path.write_text(file_content, encoding="ascii")

    exit_code = main.run_catalogue(str(file_path))

    # addModel called twice, one per line
    assert mock_model_catalogue.addModel.call_count == 2
    mock_model_catalogue.evaluateModels.assert_called_once()
    mock_model_catalogue.generateReport.assert_called_once()
    assert exit_code == 0


def test_run_catalogue_file_not_found():
    exit_code = main.run_catalogue("nonexistentfile.txt")
    assert exit_code == 1


def test_configure_logging_silent(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("LOG_FILE", raising=False)
    main.configure_logging()
    # Should have no handlers configured
    assert len(main.logger._core.handlers) == 0


def test_configure_logging_info(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "1")
    monkeypatch.delenv("LOG_FILE", raising=False)
    main.configure_logging()
    handlers = main.logger._core.handlers.values()
    assert any(h.levelno == 20 for h in handlers)  # INFO level


def test_configure_logging_debug(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "2")
    monkeypatch.delenv("LOG_FILE", raising=False)
    main.configure_logging()
    handlers = main.logger._core.handlers.values()
    assert any(h.levelno == 10 for h in handlers)  # DEBUG level


def test_configure_logging_log_file_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_LEVEL", "1")
    log_file = tmp_path / "app.log"
    log_file.write_text("")  # create empty file to simulate existence
    monkeypatch.setenv("LOG_FILE", str(log_file))

    main.configure_logging()
    handlers = main.logger._core.handlers.values()
    assert any(h.levelno == 20 for h in handlers)  # INFO level


def test_main_entrypoint_with_arg(
    tmp_path, mock_model_catalogue, mock_model, monkeypatch
):
    monkeypatch.setenv("LOG_LEVEL", "1")
    file_content = "url1,url2,url3\n"
    file_path = tmp_path / "input.txt"
    file_path.write_text(file_content, encoding="ascii")

    test_argv = ["main.py", str(file_path)]

    with patch("sys.argv", test_argv), \
         patch("sys.exit") as mock_exit, \
         patch("src.main.validate_github_token", return_value=True):
        main.configure_logging()
        # call main block code manually:
        if len(sys.argv) < 2:
            print("Usage: run <absolute_path_to_input_file>")
            sys.exit(1)
        exit_code = main.run_catalogue(sys.argv[1])
        mock_exit.assert_not_called()
        assert exit_code == 0