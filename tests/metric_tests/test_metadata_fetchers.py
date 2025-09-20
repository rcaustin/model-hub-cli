from unittest.mock import MagicMock
from src.util.metadata_fetchers import HuggingFaceFetcher, GitHubFetcher


# HuggingFaceFetcher Tests
def test_huggingface_fetcher_success():
    session = MagicMock()
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {"id": "model-id", "downloads": 1000}
    session.get.return_value = mock_response

    fetcher = HuggingFaceFetcher(session=session)
    url = "https://huggingface.co/organization/model-id"
    metadata = fetcher.fetch_metadata(url)

    session.get.assert_called_once_with(
        "https://huggingface.co/api/models/organization/model-id",
        timeout=5
    )
    assert metadata == {"id": "model-id", "downloads": 1000}


def test_huggingface_fetcher_invalid_url():
    session = MagicMock()
    fetcher = HuggingFaceFetcher(session=session)

    # Missing model path
    metadata = fetcher.fetch_metadata("https://huggingface.co/")
    assert metadata is None
    session.get.assert_not_called()


def test_huggingface_fetcher_http_failure():
    session = MagicMock()
    mock_response = MagicMock(ok=False, status_code=404)
    session.get.return_value = mock_response

    fetcher = HuggingFaceFetcher(session=session)
    metadata = fetcher.fetch_metadata("https://huggingface.co/org/model")

    assert metadata is None
    session.get.assert_called_once()


# GitHubFetcher Tests
def test_github_fetcher_success():
    session = MagicMock()

    # Mock contributors response
    contrib_response = MagicMock(ok=True)
    contrib_response.json.return_value = [{"login": "alice"}, {"login": "bob"}]

    # Mock license response
    license_response = MagicMock(ok=True)
    license_response.json.return_value = {"license": {"spdx_id": "MIT"}}

    session.get.side_effect = [contrib_response, license_response]

    fetcher = GitHubFetcher(session=session)
    url = "https://github.com/org/repo"
    metadata = fetcher.fetch_metadata(url)

    assert metadata == {
        "contributors": [{"login": "alice"}, {"login": "bob"}],
        "license": "MIT"
    }
    assert session.get.call_count == 2


def test_github_fetcher_invalid_url():
    session = MagicMock()
    fetcher = GitHubFetcher(session=session)

    # Not a GitHub URL
    metadata = fetcher.fetch_metadata("https://example.com/org/repo")
    assert metadata is None
    session.get.assert_not_called()


def test_github_fetcher_missing_path_parts():
    session = MagicMock()
    fetcher = GitHubFetcher(session=session)

    # Path too short
    metadata = fetcher.fetch_metadata("https://github.com/org")
    assert metadata is None
    session.get.assert_not_called()


def test_github_fetcher_partial_failure():
    session = MagicMock()

    # Contributors fetch fails
    contrib_response = MagicMock(ok=False, status_code=403)

    # License fetch succeeds
    license_response = MagicMock(ok=True)
    license_response.json.return_value = {"license": {"spdx_id": "Apache-2.0"}}

    session.get.side_effect = [contrib_response, license_response]

    fetcher = GitHubFetcher(session=session)
    metadata = fetcher.fetch_metadata("https://github.com/org/repo")

    assert metadata == {"license": "Apache-2.0"}
    assert session.get.call_count == 2
