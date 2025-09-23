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
        return "".join(ndjson_report)

    def getModelNDJSON(self, model: Model) -> str:
        ndjson_obj = {
            "name": model.name,
            "category": model.getCategory(),
            "net_score": model.get_score("NetScore"),
            "net_score_latency": model.get_latency("NetScore"),
            "ramp_up_time": model.get_score("RampUpMetric"),
            "ramp_up_time_latency": model.get_latency("RampUpMetric"),
            "bus_factor": model.get_score("BusFactorMetric"),
            "bus_factor_latency": model.get_latency("BusFactorMetric"),
            "performance_claims": model.get_score("PerformanceClaimsMetric"),
            "performance_claims_latency": model.get_latency("PerformanceClaimsMetric"),
            "license": model.get_score("LicenseMetric"),
            "license_latency": model.get_latency("LicenseMetric"),
            "size_score": model.evaluations.get("SizeMetric", {}),  # may be dict
            "size_score_latency": model.get_latency("SizeMetric"),
            "dataset_and_code_score": model.get_score("AvailabilityMetric"),
            "dataset_and_code_score_latency": model.get_latency("AvailabilityMetric"),
            "dataset_quality": model.get_score("DatasetQualityMetric"),
            "dataset_quality_latency": model.get_latency("DatasetQualityMetric"),
            "code_quality": model.get_score("CodeQualityMetric"),
            "code_quality_latency": model.get_latency("CodeQualityMetric"),
        }

        # Convert dictionary to NDJSON (one key-value pair per line)
        return json.dumps(ndjson_obj, separators=(",", ":"))
