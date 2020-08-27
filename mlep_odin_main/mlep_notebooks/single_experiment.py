import os, time, json, sys, pdb, click

# import mlflow

import mlep.core.MLEPDriftAdaptor as MLEPDriftAdaptor

import mlep.data_model.BatchedLocal as BatchedLocal
import mlep.data_model.StreamLocal as StreamLocal
import mlep.data_set.PseudoJsonTweets as PseudoJsonTweets
import mlep.utils.io_utils as io_utils
import mlep.utils.time_utils as time_utils

"""
Arguments

python application.py experimentName [updateSchedule] [weightMethod] [selectMethod] [filterMethod] [kVal]

"""
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

@click.command()
@click.argument('experimentname')

@click.option('--allow_explicit_drift', default=True, type=bool)
@click.option('--explicit_drift_class', default="LabeledDriftDetector", type=click.Choice(["LabeledDriftDetector"]))
@click.option('--explicit_drift_mode', default="EDDM", type=click.Choice(["DDM", "EDDM", "PageHinkley", "ADWIN"]))
@click.option('--explicit_update_mode', default="all", type=click.Choice(["all", "errors", "weighted"]))

@click.option('--allow_unlabeled_drift', default=False, type=bool)
@click.option('--unlabeled_drift_class', default="UnlabeledDriftDetector", type=click.Choice(["UnlabeledDriftDetector"]))
@click.option('--unlabeled_drift_mode', default="EnsembleDisagreement", type=click.Choice(["EnsembleDisagreement"]))
@click.option('--unlabeled_update_mode', default="all", type=click.Choice(["all", "errors", "weighted"]))

@click.option('--allow_update_schedule', default=False, type=bool)
@click.option('--update_schedule', default=2592000000, type=int)
@click.option('--schedule_update_mode', default="all", type=click.Choice(["all", "errors", "weighted"]))

@click.option('--weight_method', default="performance", type=click.Choice(["unweighted", "performance"]))
@click.option('--select_method', default="recent", type=click.Choice(["train", "historical", "historical-new", "historical-updates","recent","recent-new","recent-updates"]))
@click.option('--filter_method', default="nearest", type=click.Choice(["no-filter", "top-k", "nearest"]))
@click.option('--kval', default=5, type=int)
@click.option('--update_prune', default="5", type=str)
def main(experimentname, 
            allow_explicit_drift, explicit_drift_class, explicit_drift_mode, explicit_update_mode,
            allow_unlabeled_drift, unlabeled_drift_class, unlabeled_drift_mode, unlabeled_update_mode,
            allow_update_schedule, update_schedule, schedule_update_mode,
            weight_method, select_method, filter_method, kval, update_prune):

    # Tracking URI -- yeah it's not very secure, but w/e
    # mlflow.set_tracking_uri("mysql://mlflow:mlflow@127.0.0.1:3306/mlflow_runs")
    # Where to save data:
    # mlflow.start_run(run_name=experimentname)


    # We'll load the config file, make changes, and write a secondary file for experiments
    mlepConfig = io_utils.load_json('./MLEPServer.json')

    # Get the option if in args, else use the one present in the config...
    for _item in mlepConfig["config"]:
        try:
            mlepConfig["config"][_item] = eval(_item)   # If option not given furing launch, eval(*) will return NameError
        except NameError:
            pass    # Here the value of config in the MLEPServer.json file is used
    

    internalTimer = 0
    streamData = StreamLocal.StreamLocal(data_source="data/realisticStreamComb_2013_feb19.json", data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)

    augmentation = BatchedLocal.BatchedLocal(data_source='data/collectedIrrelevant.json', data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
    augmentation.load_by_class()

    trainingData = BatchedLocal.BatchedLocal(data_source='data/initialTrainingData.json', data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
    trainingData.load()
    

    # Now we have the data
    MLEPLearner = MLEPDriftAdaptor.MLEPDriftAdaptor(config_dict=mlepConfig, safe_mode=False)

    # Perform initial traininig
    MLEPLearner.initialTrain(traindata=trainingData)
    io_utils.std_flush("Completed training at", time_utils.readable_time())
    MLEPLearner.addAugmentation(augmentation)
    io_utils.std_flush("Added augmentation at", time_utils.readable_time())

    totalCounter = 0
    implicit_mistakes = 0.0
    implicit_count = 0
    explicit_mistakes = 0.0
    explicit_count = 0


    while streamData.next():
        if internalTimer < streamData.getObject().getValue("timestamp"):
            internalTimer = streamData.getObject().getValue("timestamp")
            MLEPLearner.updateTime(internalTimer)

        if streamData.getLabel() is None:
            classification = MLEPLearner.classify(streamData.getObject(), "implicit")
        else:
            classification = MLEPLearner.classify(streamData.getObject(), "explicit")

        
        if streamData.getLabel() is None:
            if classification != streamData.getObject().getValue("true_label"):
                implicit_mistakes += 1.0
            implicit_count += 1
        else:
            if classification != streamData.getLabel():
                explicit_mistakes += 1.0
            explicit_count += 1
        totalCounter += 1

        
        if totalCounter % 100 == 0 and totalCounter>0.0:
            implicit_running_error = 2.00
            explicit_running_error = 2.00
            if implicit_count:
                implicit_running_error = implicit_mistakes/float(implicit_count)
            if explicit_count:
                explicit_running_error = explicit_mistakes/float(explicit_count)
            io_utils.std_flush("Fin: %6i samples\t\texplicit error: %2.4f\t\t implicit error: %2.4f"%(totalCounter, explicit_running_error, implicit_running_error))
            
            implicit_mistakes = 0.0
            implicit_count = 0
            explicit_mistakes = 0.0
            explicit_count = 0
        
    

    MLEPLearner.shutdown()

    io_utils.std_flush("\n-----------------------------\nCOMPLETED\n-----------------------------\n")
    
    
    #mlflow.log_param("run_complete", True)
    #mlflow.log_param("total_samples", totalCounter)  
    #mlflow.end_run()

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter