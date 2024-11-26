from zenml import pipeline
from zenml.client import Client
from steps import trainandpredictextratree


@pipeline(enable_cache=False)
def train_extra_tree_pipeline(n_estimators: int, criterion: str):
    client = Client()

    X_train = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_0')
    y_train = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_2')

    model = trainandpredictextratree.train_extra_tree(X_train=X_train, y_train=y_train, n_estimators=n_estimators,
                                                      criterion=criterion)
