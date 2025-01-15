from pipelines import datapipeline, dataengineeringpipeline, splittinddatapipeline, trainpipelineneuralnetwork
from pipelines import neuralnetwork_modelevaluationpipeline, trainpipelinerandomforest, randomforest_modelevaluationpipeline
from pipelines import trainpipelineextratree, extra_tree_modelevaluationpipeline, ensemblemodelpipeline, createplots_pipeline

# Loading dataset
dataloader = datapipeline.datapipeline.with_options(
    config_path='dataloaderconfig.yml')()

# Performing important steps on the data
dataengineeringpipeline_instance = dataengineeringpipeline.dataengineeringpipeline.with_options(
    config_path='dataengineeringconfig.yml')()

# Splitting dataset into X and y and train-test sets
splittingdatapipeline_instance = splittinddatapipeline.splittingdatapipeline.with_options(
    config_path='splittingdataconfig.yml')()

# Training and evaluation - Extra Tree
trainandpredictpipeline_et_instace = trainpipelineextratree.train_extra_tree_pipeline.with_options(
    config_path='extratreeconfig.yml'
)()
modelevaluationpipeline_et_instace = extra_tree_modelevaluationpipeline.extratree_modelevaluationpipeline()

# Training and evaluation - Neural Network
trainandpredictpipeline_nn_instace = trainpipelineneuralnetwork.train_neuralnetwork_pipeline()
modelevaluationpipeline_nn_instance = neuralnetwork_modelevaluationpipeline.neuralnetwork_modelevaluationpipeline()

# Training and evaluation - Random Forest
trainandpredictpipeline_rf_instace = trainpipelinerandomforest.train_randomforest_pipeline.with_options(
    config_path='randomforestconfig.yml'
)()
modelevaluationpipeline_rf_instance = randomforest_modelevaluationpipeline.randomforest_modelevaluationpipeline()

# Ensembled models
enesembled_models_instance = ensemblemodelpipeline.ensemblemodel_pipeline()

# manual plots
plots_instance = createplots_pipeline.createplots_pipeline()
