from zenml import pipeline
from zenml.client import Client
from steps import trainandpredictrf
from steps import evaluaterandomforest


@pipeline(enable_cache=False)
def train_randomforest_pipeline(random_state: int, n_splits: int):
    client = Client()

    X = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::splitdata::output_0')
    y = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::splitdata::output_1')

    X_train = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_0')
    X_test = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_1')
    y_train = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_2')
    y_test = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_3')

    rf_model, cv_scores = trainandpredictrf.trainrf_withcv(X=X, y=y, random_state=random_state, n_splits=n_splits)
