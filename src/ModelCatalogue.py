from Model import Model
from Metric import Metric


class ModelCatalogue:

    # models holds all Model instances in the catalogue.
    # metrics holds all Metric instances to be applied to models.

    def __init__(self):
        self.models: list[Model] = []
        self.metrics: list[Metric] = []

    def getModel(self, model: Model):
        self.models.append(model)

    def evaluateModels(self):
        # Evaluate each model with each metric and store results
        for model in self.models:
            for metric in self.metrics:
                model.evaluate(metric)

    def generateReport(self):
        # Implement logic to generate a report from model evaluations
        pass
