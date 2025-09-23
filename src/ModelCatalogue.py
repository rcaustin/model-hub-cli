import json

from loguru import logger

from src.Metric import Metric
from src.metrics.BusFactorMetric import BusFactorMetric
from src.metrics.LicenseMetric import LicenseMetric
from src.metrics.SizeMetric import SizeMetric
from src.metrics.CodeQualityMetric import CodeQualityMetric
from src.Model import Model


class ModelCatalogue:

    # models holds all Model instances in the catalogue.
    # metrics holds all Metric instances to be applied to models.

    def __init__(self):
        self.models: list[Model] = []
        self.metrics: list[Metric] = [
            LicenseMetric(),
            BusFactorMetric(),
            SizeMetric(),
            CodeQualityMetric()
        ]

    def addModel(self, model: Model):
        self.models.append(model)

        logger.debug(
            """Model added to the catalogue:
            Model URL = '{}',
            Dataset URL = '{}',
            Code URL = '{}'""",
            model.modelLink,
            model.datasetLink,
            model.codeLink
        )

    def evaluateModels(self):
        for model in self.models:
            model.evaluate_all(self.metrics)

    def generateReport(self):
        ndjson_report = []
        for model in self.models:
            ndjson_report.append(self.getModelNDJSON(model))

        logger.debug("Report generated for {} models.", len(self.models))
        return "\n".join(ndjson_report)

    def getModelNDJSON(self, model: Model) -> str:
        ndjson_obj = {
            "name": model.name,
            "category": model.getCategory(),
            "net_score": model.getScore("NetScore"),
            "net_score_latency": model.getLatency("NetScore"),
            "ramp_up_time": model.getScore("RampUpMetric"),
            "ramp_up_time_latency": model.getLatency("RampUpMetric"),
            "bus_factor": model.getScore("BusFactorMetric"),
            "bus_factor_latency": model.getLatency("BusFactorMetric"),
            "performance_claims": model.getScore("PerformanceClaimsMetric"),
            "performance_claims_latency": model.getLatency("PerformanceClaimsMetric"),
            "license": model.getScore("LicenseMetric"),
            "license_latency": model.getLatency("LicenseMetric"),
            "size_score": model.getScore("SizeMetric", {}),  # may be dict
            "size_score_latency": model.getLatency("SizeMetric"),
            "dataset_and_code_score": model.getScore("AvailabilityMetric"),
            "dataset_and_code_score_latency": model.getLatency("AvailabilityMetric"),
            "dataset_quality": model.getScore("DatasetQualityMetric"),
            "dataset_quality_latency": model.getLatency("DatasetQualityMetric"),
            "code_quality": model.getScore("CodeQualityMetric"),
            "code_quality_latency": model.getLatency("CodeQualityMetric"),
        }

        # Convert dictionary to NDJSON (one key-value pair per line)
        return json.dumps(ndjson_obj, separators=(",", ":"))
