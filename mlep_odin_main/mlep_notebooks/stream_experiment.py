import os, time, json, sys, pdb, click

import mlep.core.MLEPModelDriftAdaptor as MLEPModelDriftAdaptor
import mlep.data_model.BatchedLocal as BatchedLocal
import mlep.data_model.StreamLocal as StreamLocal
import mlep.data_set.PseudoJsonTweets as PseudoJsonTweets

import mlep.utils.io_utils as io_utils
import mlep.utils.time_utils as time_utils

import warnings
# warnings.filterwarnings(action="ignore", category=FutureWarning)

import traceback

@click.command()
@click.argument('experimentname')
def main(experimentname):
    #
    f_write = open(experimentname+".txt", "a")

    # set up the base config
    mlepConfig = io_utils.load_json("./MLEPServer.json")

    # update as per experiment requires
    mlepConfig["config"]["weight_method"] = "unweighted"
    mlepConfig["config"]["select_method"] = "recent"
    mlepConfig["config"]["filter_select"] = "nearest"

    # we are not updating internal timer...
    streamData = StreamLocal.StreamLocal(data_source="./data/realisticStreamComb_2013_feb19.json", data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
    
    augmentation = BatchedLocal.BatchedLocal(data_source="./data/collectedIrrelevant.json", data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
    augmentation.load_by_class()

    trainingData = BatchedLocal.BatchedLocal(data_source="./data/initialTrainingData.json", data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
    trainingData.load()

    MLEPLearner = MLEPModelDriftAdaptor.MLEPModelDriftAdaptor(config_dict=mlepConfig)
    MLEPLearner.initialTrain(traindata=trainingData)
    io_utils.std_flush("Completed training at", time_utils.readable_time())
    MLEPLearner.addAugmentation(augmentation)
    io_utils.std_flush("Added augmentation at", time_utils.readable_time())

    totalCounter = 0
    implicit_mistakes = 0.0
    implicit_count = 0
    explicit_mistakes = 0.0
    explicit_count = 0
    implicit_error_rate = []
    explicit_error_rate = []
    while streamData.next():
        if streamData.getLabel() is None:
            classification = MLEPLearner.classify(streamData.getObject(), classify_mode="implicit")
            if classification != streamData.getObject().getValue("true_label"):
                implicit_mistakes += 1.0
            implicit_count += 1
        else:
            classification = MLEPLearner.classify(streamData.getObject(), classify_mode="explicit")
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
            implicit_error_rate.append(implicit_running_error)
            explicit_error_rate.append(explicit_running_error)
            implicit_mistakes = 0.0
            implicit_count = 0
            explicit_mistakes = 0.0
            explicit_count = 0
    f_write.write(experimentname+",implicit," + ",".join([str(item) for item in implicit_error_rate])+"\n")
    f_write.write(experimentname+",explicit," + ",".join([str(item) for item in explicit_error_rate])+"\n")
    f_write.close()

    
if __name__ == "__main__":
    main()