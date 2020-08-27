import os, time, json, sys, pdb, click

import mlep.core.MLEPDriftAdaptor as MLEPDriftAdaptor

import mlep.data_model.BatchedLocal as BatchedLocal
import mlep.data_model.StreamLocal as StreamLocal
import mlep.data_set.PseudoJsonTweets as PseudoJsonTweets
import mlep.utils.io_utils as io_utils
import mlep.utils.time_utils as time_utils

import mlflow

from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

import traceback

LOG_FILE = "./logfiles/experiment.log"
EXP_STATUS = "./logfiles/status.log"

class dumbwrite:
    def __init__(self, *args, **kwargs):
        pass
    def write(self,*args,**kwargs):
        pass
    def flush(self,*args, **kwargs):
        pass

class dumbflow:
    """ This is a replacement for mlflow that doesn't do anything -- in case we want a blank run..."""
    def __init__(self,*args, **kwargs):
        pass
    def log_param(self,*args, **kwargs):
        pass
    def start_run(self,*args, **kwargs):
        pass
    def log_metric(self,*args, **kwargs):
        pass
    def set_tracking_uri(self,*args, **kwargs):
        pass
    def end_run(self,*args, **kwargs):
        pass



@click.command()
@click.argument('runname')
@click.option('--expstatslog', default=0,type=int, help='0 for stdout only. Any positive integer: Log exp to ./logfiles/experiment.log and status to ./logfiles/status.log')
@click.option('--mlflowlog', default=0,type=int, help='0 for no mlflow. Else it will log to mlflow')
@click.option('--earlystop', default=0,type=int, help='0 for full run. Else it will stop after earlystop examples')
def main(runname, expstatslog, mlflowlog, earlystop):
    if mlflowlog:
        pass
    else:
        global mlflow
        mlflow = dumbflow()
    if expstatslog:
        exp_status_write = open(EXP_STATUS, "a")
    else:
        exp_status_write = sys.stdout

    exp_status_write.write("\n\n\n\n")
    exp_status_write.write("--------------------------")
    exp_status_write.write("  BEGINNING NEW EXECUTION (" + runname + ") AT " + str(time_utils.readable_time("%Y-%m-%d %H:%M:%S"))) 
    exp_status_write.write("  ------------------------"+ "\n\n")
    # We are tracking drift adaptivity
    # namely labeled drift detection

    # Set up explicit drift detection params
    explicit_drift_param_grid = {   "allow_explicit_drift": [(True,"ExpDr")],
                                    "explicit_drift_class": [("LabeledDriftDetector","LDD")],
                                    "explicit_drift_mode":[ ("PageHinkley", "PageHinkley"), ("ADWIN","ADWIN"),  ("EDDM","EDDM"),("DDM","DDM")], 
                                    "explicit_update_mode":[("all","A"), ("errors", "E")],

                                    "allow_unlabeled_drift": [(False,"")],
                                    "allow_update_schedule": [(False,"")],

                                    "weight_method":[( "unweighted","U"),( "performance","P" )],
                                    "select_method":[(  "recent","RR" ) , ( "recent-new","RN" ) , ( "recent-updates","RU" ) ],
                                    "filter_method":[ ("no-filter","F"), ("top-k","T"),("nearest","N")],
                                    "kval":[(5,"5"), (10,"10")]}
    explicit_drift_params = ParameterGrid(explicit_drift_param_grid)

    for param_set in explicit_drift_params:
        # This is an experiment
        if param_set["explicit_update_mode"][0] == "all":
            continue
        # Load up configuration file
        mlepConfig = io_utils.load_json('./MLEPServer.json')

        # Update config file and generate an experiment name
        experiment_name=''
        for _param in param_set:
            if param_set[_param][1] != "":
                experiment_name+=param_set[_param][1] + '-'
            mlepConfig["config"][_param] = param_set[_param][0]
        experiment_name = experiment_name[:-1]
    
        
        # Now we have the Experimental Coonfig we can use for running an experiment
        # generate an experiment name
        exp_status_write.write("--STATUS-- " + experiment_name + "   ")
        exp_status_write.flush()
        try:
            runExperiment(runname, mlepConfig, experiment_name, expstatslog, earlystop)
            exp_status_write.write("SUCCESS\n")
        except Exception as e:
            exp_status_write.write("FAILED\n")
            exp_status_write.write(traceback.format_exc())
            exp_status_write.write(str(e))
            exp_status_write.write("\n")
            exp_status_write.flush()
            mlflow.end_run()
        exp_status_write.flush()


    exp_status_write.write("\n\n")
    exp_status_write.write("--------------------------")
    exp_status_write.write("  FINISHED EXECUTION OF (" + runname + ") AT " + str(time_utils.readable_time("%Y-%m-%d %H:%M:%S"))) 
    exp_status_write.write("  ------------------------"+ "\n\n")
    exp_status_write.close()




def runExperiment(runname, mlepConfig, experiment_name, expstatuslog, earlystop):

    # set up mlflow access
    # mlflow.set_tracking_uri -- not needed, defaults to mlruns
    # mlflow.create_experiment -- need experiment name. Should I programmatically create one? or go by timestamp
    if expstatuslog:
        sys.stdout = open(LOG_FILE, "w")
    else:
        sys.stdout = dumbwrite()

    mlflow.set_tracking_uri("mysql://mlflow:mlflow@127.0.0.1:3306/mlflow_runs")
    mlflow.start_run(run_name=runname)

    # Log relevant details
    for _key in mlepConfig["config"]:
        # possible error
        if _key != "drift_metrics":
            mlflow.log_param(_key, mlepConfig["config"][_key])
    mlflow.log_param("experiment_name", experiment_name)


    internalTimer = 0
    streamData = StreamLocal.StreamLocal(data_source="./data/2014_to_feb2019_offline.json", data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)

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

    totalCounter = 0.0
    mistakes = []
    _earlystopcond = False

    while streamData.next() and not _earlystopcond:
        if internalTimer < streamData.getObject().getValue("timestamp"):
            internalTimer = streamData.getObject().getValue("timestamp")
            MLEPLearner.updateTime(internalTimer)

        classification = MLEPLearner.classify(streamData.getObject())


        totalCounter += 1.0
        if classification != streamData.getLabel():
            mistakes.append(1.0)
        else:
            mistakes.append(0.0)
        if totalCounter % 1000 == 0 and totalCounter>0.0:
            io_utils.std_flush("Completed", int(totalCounter), " samples, with running error (past 100) of", sum(mistakes[-100:])/100.0)
        if earlystop and totalCounter == earlystop:
            _earlystopcond = True
        if totalCounter % 100 == 0 and totalCounter>0.0:
            running_error = sum(mistakes[-100:])/100.0
            mlflow.log_metric("running_err"+str(int(totalCounter/100)), running_error)
    

    MLEPLearner.shutdown()

    io_utils.std_flush("\n-----------------------------\nCOMPLETED\n-----------------------------\n")
    
    
    
    mlflow.log_param("total_samples", totalCounter)
    if expstatuslog:
        mlflow.log_artifact(LOG_FILE)
    mlflow.log_param("run_complete", True)
    mlflow.end_run()

    if expstatuslog:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
    else:
        sys.stdout = sys.__stdout__




if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter



"""
# set up scheduled params:
    scheduled_param_grid = {  "update": [("2592000000", "M"), ( "1210000000","F")], 
                    "weights":[( "unweighted","U"),( "performance","P" )], 
                    "select":[( "train","TT" ) , (  "recent","RR" ) , ( "recent-new","RN" ) , ( "recent-updates","RU" ) , ( "historical-new","HN" ) , ( "historical-updates","HU" ) , ( "historical","HH" )],
                    "filter":[("no-filter","F") , ("top-k","T"),("nearest","N")],
                    "kval":[("5","5")],
                    "allow_update_schedule": [True]}
    scheduled_param = ParameterGrid(scheduled_param_grid)


    # Set up unlabeled drift detection params
    unlabeled_drift_param_grid = {   "allow_unlabeled_drift": [True],
                                    "unlabeled_drift_class": ["UnlabeledDriftDetector"],
                                    "explicit_drift_mode":["EnsembleDisagreement"]}
    unlabeled_drift_param = ParameterGrid(unlabeled_drift_param_grid)
    # Set up parameters:
"""