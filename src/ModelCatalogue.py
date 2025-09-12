from Model import Model
from Metric import Metric
import json


class ModelCatalogue:

    # models holds all Model instances in the catalogue.
    # metrics holds all Metric instances to be applied to models.

    def __init__(self):
        self.models: list[Model] = []
        self.metrics: list[Metric] = []

    def fetchModels(self):
        # Implement logic to fetch and populate self.models
        pass

    def evaluateModels(self):
        # Evaluate each model with each metric and store results
        for model in self.models:
            for metric in self.metrics:
                model.evaluate(metric)

    def generateReport(self):
        # Generate a consolidated NDJSON report for all models.
        ndjson_report = []
        for model in self.models:
            ndjson_report.append(self.getModelNDJSON(model))
        return "\n-----\n".join(ndjson_report)

    def getModelNDJSON(self, model: Model) -> str:
        # Create a dictionary with the required fields for NDJSON output.
        ndjson_obj = {
            "name": model.name,
            "category": model.getCategory(),
            "net_score": model.evaluations.get(
                "NetScore", 0.0
            ),
            "net_score_latency": int(
                model.evaluationsLatency.get(
                    "NetScore", 0.0
                ) * 1000
            ),
            "ramp_up_time": model.evaluations.get(
                "RampUpMetric", 0.0
            ),
            "ramp_up_time_latency": int(
                model.evaluationsLatency.get(
                    "RampUpMetric", 0.0
                ) * 1000
            ),
            "bus_factor": model.evaluations.get(
                "BusFactorMetric", 0.0
            ),
            "bus_factor_latency": int(
                model.evaluationsLatency.get(
                    "BusFactorMetric", 0.0
                ) * 1000
            ),
            "performance_claims": model.evaluations.get(
                "PerformanceClaimsMetric", 0.0
            ),
            "performance_claims_latency": int(
                model.evaluationsLatency.get(
                    "PerformanceClaimsMetric", 0.0
                ) * 1000
            ),
            "license": model.evaluations.get(
                "LicenseMetric", 0.0
            ),
            "license_latency": int(
                model.evaluationsLatency.get(
                    "LicenseMetric", 0.0
                ) * 1000
            ),
            "size_score": model.evaluations.get(
                "SizeMetric", {}
            ),
            "size_score_latency": int(
                model.evaluationsLatency.get(
                    "SizeMetric", 0.0
                ) * 1000
            ),
            "availability_score": model.evaluations.get(
                "AvailabilityMetric", 0.0
            ),
            "availability_score_latency": int(
                model.evaluationsLatency.get(
                    "AvailabilityMetric", 0.0
                ) * 1000
            ),
            "dataset_quality": model.evaluations.get(
                "DatasetQualityMetric", 0.0
            ),
            "dataset_quality_latency": int(
                model.evaluationsLatency.get(
                    "DatasetQualityMetric", 0.0
                ) * 1000
            ),
            "code_quality": model.evaluations.get(
                "CodeQualityMetric", 0.0
            ),
            "code_quality_latency": int(
                model.evaluationsLatency.get(
                    "CodeQualityMetric", 0.0
                ) * 1000
            ),
        }
        # Convert the dictionary to NDJSON format
        ndjson_lines = []
        for k, v in ndjson_obj.items():
            ndjson_lines.append(json.dumps({k: v}))
        return "\n".join(ndjson_lines) + "\n"
