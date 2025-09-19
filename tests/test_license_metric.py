from src.metrics.LicenseMetric import LicenseMetric


def test_license_from_huggingface_metadata(base_model):
    class ModelWithHFMeta(base_model.__class__):
        @property
        def hf_metadata(self):
            return {"cardData": {"license": "MIT"}}

        @property
        def github_metadata(self):
            return None

    model = ModelWithHFMeta(
        modelLink=base_model.modelLink,
        codeLink=base_model.codeLink,
        datasetLink=base_model.datasetLink,
    )

    metric = LicenseMetric()
    score = metric.evaluate(model)
    assert score == 1.0


def test_license_from_github_metadata(base_model):
    class ModelWithGHMeta(base_model.__class__):
        @property
        def hf_metadata(self):
            return None

        @property
        def github_metadata(self):
            return {"license": {"spdx_id": "GPL-3.0"}}

    model = ModelWithGHMeta(
        modelLink=base_model.modelLink,
        codeLink=base_model.codeLink,
        datasetLink=base_model.datasetLink,
    )

    metric = LicenseMetric()
    score = metric.evaluate(model)
    assert score == 0.0


def test_license_unknown_defaults_to_0_5(base_model):
    class ModelWithUnknownLicenses(base_model.__class__):
        @property
        def hf_metadata(self):
            return {"cardData": {"license": "Unknown-License"}}

        @property
        def github_metadata(self):
            return {"license": {"spdx_id": "Unknown-License"}}

    model = ModelWithUnknownLicenses(
        modelLink=base_model.modelLink,
        codeLink=base_model.codeLink,
        datasetLink=base_model.datasetLink,
    )

    metric = LicenseMetric()
    score = metric.evaluate(model)
    assert score == 0.5


def test_no_metadata_returns_default(base_model):
    class ModelWithNoMeta(base_model.__class__):
        @property
        def hf_metadata(self):
            return None

        @property
        def github_metadata(self):
            return None

    model = ModelWithNoMeta(
        modelLink=base_model.modelLink,
        codeLink=base_model.codeLink,
        datasetLink=base_model.datasetLink,
    )

    metric = LicenseMetric()
    score = metric.evaluate(model)
    assert score == 0.5


def test_fallback_to_github_when_hf_missing(base_model):
    class ModelWithFallback(base_model.__class__):
        @property
        def hf_metadata(self):
            return None

        @property
        def github_metadata(self):
            return {"license": {"spdx_id": "BSD-3-Clause"}}

    model = ModelWithFallback(
        modelLink=base_model.modelLink,
        codeLink=base_model.codeLink,
        datasetLink=base_model.datasetLink,
    )

    metric = LicenseMetric()
    score = metric.evaluate(model)
    assert score == 1.0
