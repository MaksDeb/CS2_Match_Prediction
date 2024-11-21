from zenml import pipeline
from zenml.client import Client
from steps import evaluateextratree


@pipeline(enable_cache=False)
def extratree_modelevaluationpipeline():
    client = Client()

    et_model = client.get_artifact_version(name_id_or_prefix='train_extra_tree_pipeline::train_extra_tree::output')

    X_test = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_1')
    y_test = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_3')

    evaluateextratree.evaluate_extra_tree(clf=et_model, X_test=X_test, y_test=y_test)

