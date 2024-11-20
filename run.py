from pipelines import datapipeline, dataengineeringpipeline, splittinddatapipeline, trainpipelineneuralnetwork
from pipelines import neuralnetwork_modelevaluationpipeline, trainpipelinerandomforest, randomforest_modelevaluationpipeline
from pipelines import trainpipelineextratree

# Loading dataset
dataloader = datapipeline.datapipeline.with_options(
    config_path='dataloaderconfig.yml')()

# Performing important steps on the data
dataengineeringpipeline_instance = dataengineeringpipeline.dataengineeringpipeline.with_options(
    config_path='dataengineeringconfig.yml')()

# Splitting dataset into X and y and train-test sets
splittingdatapipeline_instance = splittinddatapipeline.splittingdatapipeline.with_options(
    config_path='splittingdataconfig.yml')()

# Test
trainandpredictpipeline_et_instace = trainpipelineextratree.train_extra_tree_pipeline.with_options(
    config_path='extratreeconfig.yml'
)()

# Training and evaluating - Neural Network
trainandpredictpipeline_nn_instace = trainpipelineneuralnetwork.train_neuralnetwork_pipeline()
modelevaluationpipeline_nn_instance = neuralnetwork_modelevaluationpipeline.neuralnetwork_modelevaluationpipeline()

# Training and evaluating - Random Forest
trainandpredictpipeline_rf_instace = trainpipelinerandomforest.train_randomforest_pipeline.with_options(
    config_path='randomforestconfig.yml'
)()
modelevaluationpipeline_rf_instance = randomforest_modelevaluationpipeline.randomforest_modelevaluationpipeline()
