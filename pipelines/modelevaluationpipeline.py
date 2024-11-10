from zenml import pipeline
from zenml.client import Client
from steps import evaluateneuralnetwork


@pipeline(enable_cache=False)
def modelevaluationpipeline():
    client = Client()

    X_test = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_1')
    y_test = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_3')
    model = client.get_artifact_version(name_id_or_prefix='trainandpredictpipeline::trainmodel::output')

    evaluateneuralnetwork.evaluate(model=model, X_test=X_test, y_test=y_test)
