from zenml import pipeline
from zenml.client import Client
from steps import evaluaterandomforest


@pipeline(enable_cache=False)
def randomforest_modelevaluationpipeline():
    client = Client()

    rf_model = client.get_artifact_version(name_id_or_prefix='train_randomforest_pipeline::trainrf_withcv::output_0')
    cv_scores = client.get_artifact_version(name_id_or_prefix='train_randomforest_pipeline::trainrf_withcv::output_1')

    X_train = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_0')
    X_test = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_1')
    y_train = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_2')
    y_test = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_3')

    evaluaterandomforest.evaluate_randomoforest_withcv(clf=rf_model, cv_scores=cv_scores, X_train=X_train, y_train=y_train,
                                                       X_test=X_test, y_test=y_test)
