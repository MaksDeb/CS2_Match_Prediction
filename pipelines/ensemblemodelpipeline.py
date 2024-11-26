from zenml import pipeline
from zenml.client import Client
from steps import ensemblemodels


@pipeline(enable_cache=False)
def ensemblemodel_pipeline():
    client = Client()

    X_test = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_1')
    y_test = client.get_artifact_version(name_id_or_prefix='splittingdatapipeline::traintestsplit::output_3')
    neural_network_model = client.get_artifact_version(name_id_or_prefix='train_neuralnetwork_pipeline::trainmodel::output')
    random_forest_model = client.get_artifact_version(name_id_or_prefix='train_randomforest_pipeline::trainrf_withcv::output_0')
    extra_tree_model = client.get_artifact_version(name_id_or_prefix='train_extra_tree_pipeline::train_extra_tree::output')

    ensemblemodels.ensemble_models(model1=neural_network_model, model2=random_forest_model, model3=extra_tree_model,
                                   X_test=X_test, y_test=y_test)
