import os, time, sys, random
import pdb
from mlep.utils import io_utils, sqlite_utils, time_utils
from math import log as ln_e, exp
from numpy import vstack, asarray
import sqlite3
from sklearn.neighbors import KDTree
import mlep.trackers.MemoryTracker as MemoryTracker
import mlep.representations.ZonedDistribution as ZonedDistribution
from copy import deepcopy

#import mlep.tools.distributions.CosineSimilarityDistribution as CosineSimilarityDistribution
#import mlep.tools.distributions.DistanceDistribution as DistanceDistribution
#import mlep.tools.metrics.NumericMetrics as NumericMetrics
#import mlep.tools.metrics.TextMetrics as TextMetrics
from mlep.tools.distributions.DistanceDistribution import DistanceDistribution as PrimaryDistribution
from mlep.tools.metrics.NumericMetrics import L2Norm as PrimaryMetrics
#from mlep.tools.metrics.NumericMetrics import Manhattan as PrimaryMetrics
#from mlep.tools.metrics.TextMetrics import inverted_cosine_similarity as PrimaryMetrics
class MLEPModelDriftAdaptor():
    def __init__(self, config_dict):
        """Initialize the learning server.

        config_dift -- [dict] JSON Configuration dictionary
        """
        io_utils.std_flush("Initializing MLEP_MODEL_DRIFT_ADAPTOR")

        import mlep.trackers.MetricsTracker as MetricsTracker
        import mlep.trackers.ModelDB as ModelDB
        import mlep.trackers.ModelTracker as ModelTracker

        self.setUpCoreVars()
        self.ModelDB = ModelDB.ModelDB()
        self.loadConfig(config_dict)
        self.setUpEncoders()
        self.METRICS = MetricsTracker.MetricsTracker()
        self.setUpExplicitDriftTracker()
        self.setUpUnlabeledDriftTracker()
        self.setUpMemories()
        self.ModelTracker = ModelTracker.ModelTracker()
        io_utils.std_flush("Finished initializing MLEP...")

    def setUpCoreVars(self,):
        self.KNOWN_EXPLICIT_DRIFT_CLASSES = ["LabeledDriftDetector"]
        self.KNOWN_UNLABELED_DRIFT_CLASSES = ["UnlabeledDriftDetector"]
        self.ALPHA = 0.5
        # Setting of 'hosted' models + data cetroids
        self.MODELS = {}
        # Augmenter
        self.AUGMENT = None
        # Statistics
        self.LAST_CLASSIFICATION, self.LAST_ENSEMBLE = 0, []
        self.HASHMAX = sys.maxsize
        self.overallTimer = None

    def setUpUnlabeledDriftTracker(self,):
        if self.MLEPConfig["allow_unlabeled_drift"]:
            if self.MLEPConfig["unlabeled_drift_class"] not in self.KNOWN_UNLABELED_DRIFT_CLASSES:
                raise ValueError("Unlabeled drift class '%s' in configuration is not part of any known Unlabeled Drift Classes: %s"%(self.MLEPConfig["unlabeled_drift_class"], str(self.KNOWN_UNLABELED_DRIFT_CLASSES)))
            driftTracker = self.MLEPConfig["unlabeled_drift_mode"]
            driftModule = self.MLEPConfig["unlabeled_drift_class"]
            driftArgs = self.MLEPConfig["drift_args"] if "drift_args" in self.MLEPConfig else {}
            driftModuleImport = __import__("mlep.drift_detector.%s.%s"%(driftModule, driftTracker), fromlist=[driftTracker])
            driftTrackerClass = getattr(driftModuleImport,driftTracker)
            if self.MLEPConfig["unlabeled_drift_mode"] == "EnsembleDisagreement":    
                self.UNLABELED_DRIFT_TRACKER = driftTrackerClass(**driftArgs)
            if self.MLEPConfig["unlabeled_drift_mode"] == "KullbackLeibler":
                # Just set it up, but don't __init__ it. We do that in initialTrain...
                self.UNLABELED_DRIFT_TRACKER = driftTrackerClass
                self.kullback = {}
                self.kullback["secondary_distribution"] = driftTrackerClass
                self.kullback["process_length"] = 1000
                self.kullback["raw_vals"] = []
                self.kullback["threshold"] = 0.007

        else:
            self.UNLABELED_DRIFT_TRACKER = None
            io_utils.std_flush("\tUnlabeled drift tracker not used in this run", time_utils.readable_time())

    def setUpExplicitDriftTracker(self,):
        if self.MLEPConfig["allow_explicit_drift"]:
            if self.MLEPConfig["explicit_drift_class"] not in self.KNOWN_EXPLICIT_DRIFT_CLASSES:
                raise ValueError("Explicit drift class '%s' in configuration is not part of any known Explicit Drift Classes: %s"%(self.MLEPConfig["explicit_drift_class"], str(self.KNOWN_EXPLICIT_DRIFT_CLASSES)))
            driftTracker = self.MLEPConfig["explicit_drift_mode"]
            driftModule = self.MLEPConfig["explicit_drift_class"]
            driftArgs = self.MLEPConfig["drift_args"] if "drift_args" in self.MLEPConfig else {}
            driftModuleImport = __import__("mlep.drift_detector.%s.%s"%(driftModule, driftTracker), fromlist=[driftTracker])
            driftTrackerClass = getattr(driftModuleImport,driftTracker)
            self.EXPLICIT_DRIFT_TRACKER = driftTrackerClass(**driftArgs)
        else:
            self.EXPLICIT_DRIFT_TRACKER = None
            io_utils.std_flush("\tExplicit drift tracker not used in this run", time_utils.readable_time())


    def loadConfig(self, config_dict):
        """Load JSON configuration file and initialize attributes.

        config_path -- [str] Path to the JSON configuration file.
        """
        io_utils.std_flush("\tStarted loading JSON configuration file at", time_utils.readable_time())
        self.config = config_dict
        if self.config["config"]["filter_select"] != "nearest":
            raise ValueError("MLEPModelDriftAdaptor requires nearest for filter_select")
        self.MLEPConfig = self.config["config"]; self.MLEPModels = self.config["models"]; self.MLEPPipelines = self.getValidPipelines(); self.MLEPEncoders = self.getValidEncoders()
        io_utils.std_flush("\tFinished loading JSON configuration file at", time_utils.readable_time())

    def updateTime(self,timerVal):
        """ Manually updating time for experimental evaluation """
        self.overallTimer = timerVal

    def setUpEncoders(self):
        """Set up built-in encoders (Google News w2v)."""
        io_utils.std_flush("\tStarted setting up encoders at", time_utils.readable_time())
        self.ENCODERS = {}
        for _ , encoder_config in self.MLEPEncoders.items():
            io_utils.std_flush("\t\tSetting up encoder", encoder_config["name"], "at", time_utils.readable_time())
            encoderName = encoder_config["scriptName"]
            encoderModule = __import__("mlep.data_encoder.%s" % encoderName,
                    fromlist=[encoderName])
            encoderClass = getattr(encoderModule, encoderName)
            self.ENCODERS[encoder_config["name"]] = encoderClass()
            try:
                self.ENCODERS[encoder_config["name"]].setup(**encoder_config["args"])
            # Value Error is for joblib load -- need a message to convey as such
            except (IOError, ValueError) as e:
                io_utils.std_flush("Encoder load failed with error:", e, ". Attempting fix.")
                self.ENCODERS[encoder_config["name"]].failCondition(**encoder_config["fail-args"])
                self.ENCODERS[encoder_config["name"]].setup(**encoder_config["args"])                
        io_utils.std_flush("\tFinished setting up encoders at", time_utils.readable_time())    

    def MLEPModelBasedUpdate(self):
        ensembleModelNames = [modelSaveName for modelSaveName in self.ModelTracker.get("recent")]
        for model_name in ensembleModelNames:
            # encode the explicit data ...            
            if self.MODELS[model_name]["memoryTracker"].memorySize("edge-mem-explicit") > 0:
                io_utils.std_flush("Performing update for %s"%model_name)
                
                explicit_memory = self.MODELS[model_name]["memoryTracker"].transferMemory("edge-mem-explicit")
                explicit_memory.load_by_class()
                explicit_memory = self.augmentTrainingData(explicit_memory)
                encoded_explicit = self.ENCODERS[self.MODELS[model_name]["encoder"]].batchEncode(explicit_memory.getData())
                io_utils.std_flush("\tObtained %i explicit edge labels"%encoded_explicit.shape[0])

                
                implicit_memory = self.MODELS[model_name]["memoryTracker"].transferMemory("edge-mem-implicit")
                implicit_memory.load_by_class()
                implicit_memory = self.augmentTrainingData(implicit_memory)
                encoded_implicit = self.ENCODERS[self.MODELS[model_name]["encoder"]].batchEncode(implicit_memory.getData())
                io_utils.std_flush("\tObtained %i implicit edge labels"%encoded_implicit.shape[0])
                
                if self.MLEPConfig["reset_memories"]:
                    self.MODELS[model_name]["memoryTracker"].clearMemory("edge-mem-explicit")
                    self.MODELS[model_name]["memoryTracker"].clearMemory("edge-mem-implicit")

                explicit_labels = explicit_memory.getLabels()
                implicit_labels = implicit_memory.getLabels()
                TrainingData, trainlabels, update_weights = None, None, None
                if len(encoded_implicit) > 0:
                    core_kdtree = KDTree(encoded_explicit, metric='euclidean')
                    io_utils.std_flush("\tGenerated KD-Tree")
                    # get the closest items; response[0] --> distance; response[1] --> indices
                    response = core_kdtree.query(encoded_implicit)
                    io_utils.std_flush("\tObtained distances for implicit memory")
                    # Label matching -- keep 'weakly supervised' correct ones. For each implicit label, compare to nearest explicit. if it matches, keep
                    supervision = []
                    supervised_implicit_labels = []
                    for _idx in range(response[1].shape[0]):
                        # get label of implicit
                        implicit_weaklabel = implicit_labels[_idx]
                        explicit_stronglabel = explicit_labels[response[1][_idx,0]]
                        if implicit_weaklabel == explicit_stronglabel:
                            supervision.append(_idx)
                            supervised_implicit_labels.append(implicit_weaklabel)
                    encoded_implicit = encoded_implicit[supervision,:]
                    trainlabels = explicit_labels+supervised_implicit_labels
                    io_utils.std_flush("\tWeak supervision -- reduced from %i to %i supervised implicit labels"%(len(implicit_labels), len(supervised_implicit_labels)))
                    
                    scale_fac = ln_e(self.ALPHA)/self.MODELS[model_name]["model"].getDataCharacteristic("delta_high")
                    update_weights = [exp(scale_fac*item) for item in response[0][supervision,0].tolist()]
                    io_utils.std_flush("\tGenerated weights for implicit samples")
                    # now we have explicit memory with weights (1) and impllicit memory, also with weights (update_weights)
                    # time to perform an update...using encoded_explicit, encoded_implicit, and weights...
                    TrainingData = vstack((encoded_explicit, encoded_implicit))
                    update_weights = [1]*encoded_explicit.shape[0] + update_weights
                else:
                    TrainingData = encoded_explicit
                    trainlabels = explicit_labels
                    update_weights = [1]*encoded_explicit.shape[0]
                
                self.updateSingle(TrainingData, trainlabels, model_name, update_weights)
                self.ModelTracker.updateModelStore(self.ModelDB)

        # Perform generate using explicit data...
        if self.MEMTRACK.memorySize("gen-mem-explicit") > 0:
            explicit_memory = self.MEMTRACK.transferMemory("gen-mem-explicit")
            explicit_memory.load_by_class()
            io_utils.std_flush("\tObtained %i explicit general labels"%self.MEMTRACK.memorySize("gen-mem-explicit"))
            implicit_memory = self.MEMTRACK.transferMemory("gen-mem-implicit")
            implicit_memory.load_by_class()
            io_utils.std_flush("\tObtained %i implicit general labels"%self.MEMTRACK.memorySize("gen-mem-implicit"))
            
            explicit_memory = self.augmentTrainingData(explicit_memory)
            implicit_memory = self.augmentTrainingData(implicit_memory)

            explicit_labels = explicit_memory.getLabels()
            try:
                implicit_labels = implicit_memory.getLabels()
            except AttributeError:
                pdb.set_trace()

            general_training = {}
            general_labels={}
            update_weights = {}
            if self.MLEPConfig["reset_memories"]:
                self.MEMTRACK.clearMemory("gen-mem-explicit")
                self.MEMTRACK.clearMemory("gen-mem-implicit")
            for encoder in self.ENCODERS:
                explicit_encoded = self.ENCODERS[encoder].batchEncode(explicit_memory.getData())
                implicit_encoded = self.ENCODERS[encoder].batchEncode(implicit_memory.getData())
                
                if len(implicit_encoded) > 0:
                    kdtree_gen = KDTree(explicit_encoded, metric='euclidean')
                    response = kdtree_gen.query(implicit_encoded)
                    #scale_fac = ln_e(self.ALPHA)/self.MODELS[model_name]["model"].getDataCharacteristic("delta_high")
                    io_utils.std_flush("\tObtained distances for implicit memory")
                    supervision = []
                    supervised_implicit_labels = []
                    for _idx in range(response[1].shape[0]):
                        # get label of implicit
                        implicit_weaklabel = implicit_labels[_idx]
                        explicit_stronglabel = explicit_labels[response[1][_idx,0]]
                        if implicit_weaklabel == explicit_stronglabel:
                            supervision.append(_idx)
                            supervised_implicit_labels.append(implicit_weaklabel)
                    implicit_encoded = implicit_encoded[supervision,:]
                    io_utils.std_flush("\tWeak supervision -- reduced from %i to %i supervised implicit labels"%(len(implicit_labels), len(supervised_implicit_labels)))

                    __update_weights__ = [exp(item) for item in response[0][supervision,0].tolist()]

                    general_labels[encoder] = explicit_labels+supervised_implicit_labels
                    general_training[encoder] = vstack((explicit_encoded, implicit_encoded))
                    update_weights[encoder] = [1]*explicit_encoded.shape[0] + __update_weights__
                else:
                    general_labels[encoder] = explicit_labels
                    general_training[encoder] = explicit_encoded
                    update_weights[encoder] = [1]*explicit_encoded.shape[0]
            self.trainGeneralMemory(general_training, general_labels, update_weights)
        self.ModelTracker.updateModelStore(self.ModelDB)

    def trainGeneralMemory(self,traindata, trainlabels, sample_weights=None):
        """ Function to train traindata """
        for pipeline in self.MLEPPipelines:
            # set up pipeline
            currentPipeline = self.MLEPPipelines[pipeline] 
            currentEncoder = currentPipeline["sequence"][0]           
            # get the closest items; response[0] --> distance; response[1] --> indices
            precision, recall, score, pipelineTrained, data_centroid = self.createDensePipeline(traindata[currentEncoder], trainlabels[currentEncoder], currentPipeline, sample_weight = sample_weights[currentEncoder])
            timestamp = time.time()
            modelIdentifier = self.createModelId(timestamp, currentPipeline["name"],score) 
            modelSavePath = "_".join([currentPipeline["name"], modelIdentifier])
            trainDataSavePath, testDataSavePath = "", ""
            # save the model
            self.buildModel(_name=modelSavePath, _model=pipelineTrained, _encoder = currentEncoder)
            del pipelineTrained            

            self.ModelDB.insertModelToDb(modelid=modelIdentifier, parentmodelid=None, pipelineName=str(currentPipeline["name"]),
                                timestamp=timestamp, data_centroid=data_centroid, training_model=str(modelSavePath), 
                                training_data=str(trainDataSavePath), test_data=str(testDataSavePath), precision=precision, recall=recall, score=score,
                                _type=str(currentPipeline["type"]), active=1)


    def updateSingle(self,traindata, trainlabels, model_name, sample_weights=None):
        if not self.MODELS[model_name]["model"].isUpdatable():
            # generate a model, instead of updating TODO
            # Additional TODO -- cross validation, or separate test set passing...
            pass
        else:
            # update this single model...
            modelDetails = self.ModelDB.getModelDetails([model_name]) # Gets fscore, pipelineName, modelSaveName
            pipelineNameDict = self.ModelDB.getDetails(modelDetails, 'pipelineName', 'dict')
            currentPipeline = self.MLEPPipelines[pipelineNameDict[model_name]]
            precision, recall, score, pipelineTrained, data_centroid = self.createDensePipeline(traindata, trainlabels, currentPipeline, model_name, sample_weights)
            timestamp = time.time()
            modelIdentifier = self.createModelId(timestamp, currentPipeline["name"], score)
            modelSavePath = "_".join([currentPipeline["name"], modelIdentifier])
            trainDataSavePath, testDataSavePath = "", ""
            self.buildModel(_name=modelSavePath, _model=pipelineTrained, _encoder=currentPipeline["sequence"][0])
            del pipelineTrained            

            self.ModelDB.insertModelToDb(modelid=modelIdentifier, parentmodelid=model_name, pipelineName=str(currentPipeline["name"]),
                            timestamp=timestamp, data_centroid=data_centroid, training_model=str(modelSavePath), 
                            training_data=str(trainDataSavePath), test_data=str(testDataSavePath), precision=precision, recall=recall, score=score,
                            _type=str(currentPipeline["type"]), active=1)


    def MLEPMemoryBasedUpdate(self,memory_type="scheduled"):
        if self.MEMTRACK.memorySize(memory_name=memory_type) < self.MLEPConfig["min_train_size"]:
            io_utils.std_flush("Attemped update using", memory_type, "-memory with", self.MEMTRACK.memorySize(memory_name=memory_type),"data samples. Failed due to requirement of", self.MLEPConfig["min_train_size"], "samples." )    
            return
        io_utils.std_flush("Update using", memory_type, "-memory at", time_utils.ms_to_readable(self.overallTimer), "with", self.MEMTRACK.memorySize(memory_name=memory_type),"data samples." )
        # Get the training data from Memory (and clear the memory)
        TrainingData = self.getTrainingData(memory_type=memory_type)
        # Generate and update
        self.train(TrainingData)
        io_utils.std_flush("Completed", memory_type, "-memory based Model generation at", time_utils.readable_time())
        self.update(TrainingData,models_to_update=self.MLEPConfig["models_to_update"])
        io_utils.std_flush("Completed", memory_type, "-memory based Model Update at", time_utils.readable_time())
        # Now we update model store.
        self.ModelTracker.updateModelStore(self.ModelDB)

    def augmentTrainingData(self,trainingDataModel):
        negDataLength = trainingDataModel.class_size(0)
        posDataLength = trainingDataModel.class_size(1)
        if negDataLength < 0.8*posDataLength:
            io_utils.std_flush("Too few negative results. Adding more")
            if self.AUGMENT.class_size(0) < posDataLength:
                # We'll need a random sampled for self.negatives BatchedLoad
                trainingDataModel.augment_by_class(self.AUGMENT.getObjectsByClass(0), 0)
            else:
                trainingDataModel.augment_by_class(random.sample(self.AUGMENT.getObjectsByClass(0), posDataLength-negDataLength), 0)
        elif negDataLength > 1.2 *posDataLength:
            # Too many negative data; we'll prune some
            io_utils.std_flush("Too many  negative samples. NOT Pruning")
            pass
            #trainingDataModel.prune_by_class(0,negDataLength-posDataLength)
            # TODO
        else:
            # Just right
            io_utils.std_flush("No augmentation necessary")
        # return combination of all classes
        return trainingDataModel

    def getTrainingData(self, memory_type="scheduled"):
        """ Get the data in self.SCHEDULED_DATA_FILE """
        # perform augmentation for the binary case if there is not enough of each type; enriching with existing negatives
        scheduledTrainingData = None
        scheduledTrainingData = self.MEMTRACK.transferMemory(memory_name=memory_type)
        self.MEMTRACK.clearMemory(memory_name=memory_type)
        scheduledTrainingData.load_by_class()

        if self.MEMTRACK.getClassifyMode() == "binary":
            return self.augmentTrainingData(scheduledTrainingData)
        else:
            raise NotImplementedError()

    def createDensePipeline(self,data, trainlabels, pipeline, source=None, sample_weight = None):
        """ Generate or Update a pipeline 
        
        If source is None, this is create. Else this is a generate.
        """
        # Data setup
        encoderName = pipeline["sequence"][0]
        centroid = self.ENCODERS[encoderName].getCentroid(data)

        # Model setup
        pipelineModel = pipeline["sequence"][1]
        pipelineModelName = self.MLEPModels[pipelineModel]["scriptName"]
        pipelineModelModule = __import__("mlep.learning_model.%s"%pipelineModelName, fromlist=[pipelineModelName])
        pipelineModelClass = getattr(pipelineModelModule,pipelineModelName)

        model = pipelineModelClass()

        precision, recall, score = None, None, None
        if source is None:
            # Generate
            precision, recall, score = model.fit_and_test(data, trainlabels, sample_weight=sample_weight)
        else:
            # Update
            model.clone(self.MODELS[source]["model"])
            precision, recall, score = model.update_and_test(data, trainlabels, sample_weight=sample_weight)

        # Store model's data characteristics
        modelCharacteristics = ZonedDistribution.ZonedDistribution(nBins=40, alpha=self.ALPHA, metric_callback=PrimaryMetrics, distribution_callback=PrimaryDistribution)
        modelCharacteristics.build(centroid,data)
        model.addDataCharacteristics(modelCharacteristics)

        return precision, recall, score, model, centroid

    def createPipeline(self,data, pipeline, source=None, sample_weight = None):
        """ Generate or Update a pipeline 
        
        If source is None, this is create. Else this is a generate.
        """
        # Data setup
        encoderName = pipeline["sequence"][0]

        X_train = self.ENCODERS[encoderName].batchEncode(data.getData())
        centroid = self.ENCODERS[encoderName].getCentroid(X_train)
        y_train = data.getLabels()

        # Model setup
        pipelineModel = pipeline["sequence"][1]

        pipelineModelName = self.MLEPModels[pipelineModel]["scriptName"]
        pipelineModelModule = __import__("mlep.learning_model.%s"%pipelineModelName, fromlist=[pipelineModelName])
        pipelineModelClass = getattr(pipelineModelModule,pipelineModelName)

        model = pipelineModelClass()

        precision, recall, score = None, None, None
        if source is None:
            # Generate
            pass
            precision, recall, score = model.fit_and_test(X_train, y_train, sample_weight=sample_weight)
        else:
            # Update
            model.clone(self.MODELS[source]["model"])
            precision, recall, score = model.update_and_test(X_train, y_train, sample_weight=sample_weight)

        # Store model's data characteristics
        modelCharacteristics = ZonedDistribution.ZonedDistribution(nBins=40, alpha=self.ALPHA, metric_callback=PrimaryMetrics, distribution_callback=PrimaryDistribution)
        modelCharacteristics.build(centroid,X_train)
        model.addDataCharacteristics(modelCharacteristics)

        return precision, recall, score, model, centroid


    def update(self, traindata, models_to_update='recent', sample_weights=None):
        # forEach(self.MODELS) --> create a copy; update copy; push details to DB            
        if self.MLEPConfig["update_prune"] == "C":  # Keep constant to number of valid pipelines
            prune_val = len(self.MLEPPipelines)
        else:
            prune_val = int(self.MLEPConfig["update_prune"])
        
        temporaryModelStore = []
        modelSaveNames = [modelSaveName for modelSaveName in self.ModelTracker.get(models_to_update)]
        modelDetails = self.ModelDB.getModelDetails(modelSaveNames) # Gets fscore, pipelineName, modelSaveName
        pipelineNameDict = self.ModelDB.getDetails(modelDetails, 'pipelineName', 'dict')
        for modelSaveName in modelSaveNames:
            # copy model and  set up new model after checking if model can be updated
            if not self.MODELS[modelSaveName]["model"].isUpdatable():
                continue
            currentPipeline = self.MLEPPipelines[pipelineNameDict[modelSaveName]]
            precision, recall, score, pipelineTrained, data_centroid = self.createPipeline(traindata, currentPipeline, modelSaveName)
            timestamp = time.time()
            modelIdentifier = self.createModelId(timestamp, currentPipeline["name"], score)
            modelSavePath = "_".join([currentPipeline["name"], modelIdentifier])
            trainDataSavePath, testDataSavePath = "", ""
            
            # We temporarily load to dictionary for sorting later.
            dicta={}
            dicta["name"] = modelSavePath; dicta["MODEL"] = pipelineTrained; dicta["CENTROID"] = data_centroid; dicta["modelid"] = modelIdentifier
            dicta["parentmodelid"] = str(modelSaveName); dicta["pipelineName"] = str(currentPipeline["name"]); dicta["timestamp"] = timestamp
            dicta["data_centroid"] = data_centroid; dicta["training_model"] = str(modelSavePath); dicta["training_data"] = str(trainDataSavePath)
            dicta["test_data"] = str(testDataSavePath); dicta["precision"] = precision; dicta["recall"] = recall; dicta["score"] = score
            dicta["_type"] = str(currentPipeline["type"]); dicta["active"] = 1; dicta["__pipeline__"] = currentPipeline
            temporaryModelStore.append(dicta)

        if len(temporaryModelStore) > prune_val:
            io_utils.std_flush("Pruning models -- reducing from", str(len(temporaryModelStore)),"to",str(prune_val),"update models." )
            # keep the highest scoring update models
            temporaryModelStore = sorted(temporaryModelStore, key=lambda k:k["score"], reverse=True)
            temporaryModelStore = temporaryModelStore[:prune_val]

        for item in temporaryModelStore:
            # save the model
            self.buildModel(_name=item["name"], _model=item["MODEL"], _encoder=item["__pipeline__"]["sequence"][0])

            self.ModelDB.insertModelToDb(modelid=item["modelid"], parentmodelid=item["parentmodelid"], pipelineName=item["pipelineName"],
                                timestamp=item["timestamp"], data_centroid=item["data_centroid"], training_model=item["training_model"], 
                                training_data=item["training_data"], test_data=item["test_data"], precision=item["precision"], recall=item["recall"], score=item["score"],
                                _type=item["_type"], active=item["active"])

    # trainData is BatchedLocal
    def initialTrain(self,traindata,models= "all"):
        
        self.train(traindata)
        self.ModelTracker._set("train", self.ModelDB.getModelsSince())
        self.ModelTracker.updateModelStore(self.ModelDB)

        if self.MLEPConfig["allow_unlabeled_drift"] and self.MLEPConfig["unlabeled_drift_mode"] == "KullbackLeibler":
            distr = ZonedDistribution.ZonedDistribution(nBins=40,alpha=self.ALPHA, metric_callback=PrimaryMetrics, distribution_callback=PrimaryDistribution)
            encoded = self.ENCODERS["w2v-main"].batchEncode(traindata.getData())
            _centroid = self.ENCODERS["w2v-main"].getCentroid(encoded)
            distr.build(centroid=_centroid, data=encoded)
            self.UNLABELED_DRIFT_TRACKER = self.UNLABELED_DRIFT_TRACKER(distr.distribution)
            self.kullback["centroid"] = _centroid
            self.kullback["secondary_distribution"] = deepcopy(distr.distribution)


    def train(self,traindata, models = 'all'):
        """ Function to train traindata """

        for pipeline in self.MLEPPipelines:
            # set up pipeline
            currentPipeline = self.MLEPPipelines[pipeline]
            precision, recall, score, pipelineTrained, data_centroid = self.createPipeline(traindata, currentPipeline)
            timestamp = time.time()
            modelIdentifier = self.createModelId(timestamp, currentPipeline["name"],score) 
            modelSavePath = "_".join([currentPipeline["name"], modelIdentifier])
            trainDataSavePath, testDataSavePath = "", ""
            # save the model
            self.buildModel(_name=modelSavePath, _model=pipelineTrained, _encoder = currentPipeline["sequence"][0])
            del pipelineTrained            

            self.ModelDB.insertModelToDb(modelid=modelIdentifier, parentmodelid=None, pipelineName=str(currentPipeline["name"]),
                                timestamp=timestamp, data_centroid=data_centroid, training_model=str(modelSavePath), 
                                training_data=str(trainDataSavePath), test_data=str(testDataSavePath), precision=precision, recall=recall, score=score,
                                _type=str(currentPipeline["type"]), active=1)

    def buildModel(self,_name,_model, _encoder):
        _model.trackDrift(self.MLEPConfig["allow_model_confidence"])
        if _name in self.MODELS:
            raise RuntimeError("Error: %s already exists in self.MODELS"%_name)
        else:
            self.MODELS[_name] = {}
            self.MODELS[_name]["model"] = _model
            # Add memories
            self.MODELS[_name]["memoryTracker"] = MemoryTracker.MemoryTracker()
            self.MODELS[_name]["memoryTracker"].addNewMemory(memory_name="core-mem-explicit",memory_store='memory')
            self.MODELS[_name]["memoryTracker"].addNewMemory(memory_name="edge-mem-explicit",memory_store='memory')
            self.MODELS[_name]["memoryTracker"].addNewMemory(memory_name="core-mem-implicit",memory_store='memory')
            self.MODELS[_name]["memoryTracker"].addNewMemory(memory_name="edge-mem-implicit",memory_store='memory')
            self.MODELS[_name]["encoder"] = _encoder


    def getTopKNearestModels(self,ensembleModelNames, data):
        # data is a DataSet object
        ensembleModelPerformance = None
        ensembleModelDistance = None
        # find top-k nearest centroids
        k_val = self.MLEPConfig["kval"]
        # don't need any fancy stuff if k-val is more than the number of models we have
        #if k_val >= len(ensembleModelNames):
        if k_val == 0:
            #pass
            raise RuntimeError("Why is k 0????????????")
        else:
            # dictify for O(1) check
            ensembleModelNamesValid = {item:1 for item in ensembleModelNames}
            # 1. First, collect list of Encoders -- model mapping
            pipelineToModel = self.ModelDB.getPipelineDetails()
            # 2. Then create mapping of encoders -- model_save_path
            encoderToModel = {}
            for _pipeline in pipelineToModel:
                # Multiple pipelines can have the same encoder
                if self.MLEPPipelines[_pipeline]["sequence"][0] not in encoderToModel:
                    encoderToModel[self.MLEPPipelines[_pipeline]["sequence"][0]] = []
                # encoderToModel[PIPELINE_NAME] = [(MODEL_NAME, PERF),(MODEL_NAME, PERF)]
                encoderToModel[self.MLEPPipelines[_pipeline]["sequence"][0]] += pipelineToModel[_pipeline]
            
            # 3. Then for each encoder, find k-closest model_save_path
            kClosestPerEncoder = {}
            for _encoder in encoderToModel:
                kClosestPerEncoder[_encoder] = []
                _encodedData = self.ENCODERS[_encoder].encode(data.getData())
                # Find distance, then sort and take top-k. item[0] (modelName). item[1] : fscore
                # np.linalg.norm(_encodedData-CENTROID)
                kClosestPerEncoder[_encoder]=[(self.ENCODERS[_encoder].getDistance(_encodedData, self.MODELS[item[0]]["model"].getDataCharacteristic("centroid")), item[1], item[0]) for item in encoderToModel[_encoder] if item[0] in ensembleModelNamesValid]
                # tup[0] --> norm; tup[1] --> fscore; tup[2] --> modelName
                kClosestPerEncoder[_encoder].sort(key=lambda tup:tup[0])
                # Truncate to top-k
                kClosestPerEncoder[_encoder] = kClosestPerEncoder[_encoder][:k_val]
            # 4. Put them all together and sort on performance; distance weighted performance (or sge is actually better...)
            kClosest = []
            for _encoder in kClosestPerEncoder:
                kClosest+=kClosestPerEncoder[_encoder]
            # Sorting by tup[1] --> fscore
            kClosest.sort(key=lambda tup:tup[1], reverse=True)

            # 5. Return top-k (item[0] --> norm; item[1] --> fscore; item[2] --> modelName)
            numRetrieved = len(kClosest)
            if numRetrieved <= k_val:
                k_val = numRetrieved
            else:
                kClosest = kClosest[:k_val]
            ensembleModelNames = [None]*k_val
            ensembleModelDistance = [None]*k_val
            ensembleModelPerformance = [None]*k_val
            for idx,_item in enumerate(kClosest):
                ensembleModelNames[idx] = _item[2]
                ensembleModelPerformance[idx] = _item[1]
                ensembleModelDistance[idx] = _item[0]
        return ensembleModelNames, ensembleModelPerformance, ensembleModelDistance

    def classify(self, data, classify_mode="explicit"):          
        """
        MLEPModelDriftAdaptor's classifier. 
        
        The MLEPModelDriftAdaptor's classifier performs model retrieval, distance calculation, and drift updates.

        Args:
            data -- a single data sample for classification
            classify_mode -- "explicit" if data is supposed to have a label. "implicit" if data is unlabeled. Since this is an experimentation framework, data is technically supposed to have a label. The distinction is on whether the label is something MLEPModelDriftAdaptor would have access to during live operation. "explicit" refers to this case, while "implicit" refers to the verso.

        Returns:
            classification -- INT -- currently only binary classification is supported
            
        """
        # First set up list of correct models
        ensembleModelNames = self.getValidModels()
        # Now that we have collection of candidaate models, we use filter_select to decide how to choose the right model
        # self.MLEPConfig["filter_select"] == "nearest":
        ensembleModelNames, ensembleModelPerformance, ensembleModelDistance = self.getTopKNearestModels(ensembleModelNames, data)

        # Given ensembleModelNames, use all of them as part of ensemble. Run the sqlite query to get model details
        modelDetails = self.ModelDB.getModelDetails(ensembleModelNames)
        if self.MLEPConfig["weight_method"] == "performance":
            if ensembleModelPerformance is not None:
                weights = ensembleModelPerformance
            else:
                # request DB for performance (f-score)
                weights = self.ModelDB.getDetails(modelDetails, 'fscore', 'list', order=ensembleModelNames)
            sumWeights = sum(weights)
            weights = [item/sumWeights for item in weights]
        elif self.MLEPConfig["weight_method"] == "unweighted":
            weights = len(ensembleModelNames)*[1.0/len(ensembleModelNames)]
        else:
            weights = len(ensembleModelNames)*[1.0/len(ensembleModelNames)]
        
        # Get encoder types in ensembleModelNames; build local dictionary of data --> encodedVersion             
        pipelineNameDict = self.ModelDB.getDetails(modelDetails, 'pipelineName', 'dict')
        localEncoder = {}
        for modelName in pipelineNameDict:
            pipelineName = pipelineNameDict[modelName]
            localEncoder[self.MLEPPipelines[pipelineName]["sequence"][0]] = 0
        
        for encoder in localEncoder:
            localEncoder[encoder] = self.ENCODERS[encoder].encode(data.getData())

        #---------------------------------------------------------------------
        # Time to classify
        
        classification = 0
        ensembleWeighted = [0]*len(ensembleModelNames)
        ensembleRaw = [0]*len(ensembleModelNames)
        for idx,_name in enumerate(ensembleModelNames):
            locally_encoded_data = localEncoder[self.MLEPPipelines[pipelineNameDict[_name]]["sequence"][0]]
            # get the model name; get pipeline name from pipelineNameDict. Then get the encodername from sequence[0], the get encoded data
            cls_ = None
            # for regular mode
            if not self.MLEPConfig["allow_model_confidence"]:
                cls_=self.MODELS[_name]["model"].predict(locally_encoded_data)
            else:
                if classify_mode == "explicit":
                    cls_=self.MODELS[_name]["model"].predict(locally_encoded_data, mode="explicit", y_label = data.getLabel())
                elif classify_mode == "implicit":
                    cls_=self.MODELS[_name]["model"].predict(locally_encoded_data, mode="implicit")
                else:
                    raise ValueError("classify_mode must be one of: 'explicit', 'implicit'. Unrecognized mode %s"%classify_mode)

            ensembleWeighted[idx] = float(weights[idx]*cls_)
            ensembleRaw[idx] = float(cls_)
        # Assume binary. We'll deal with others later
        classification = sum(ensembleWeighted)
        classification =  0 if classification < 0.5 else 1
    
        
        error = 1 if classification != data.getLabel() else 0
        ensembleError = [(1 if ensembleRawScore != data.getLabel() else 0) for ensembleRawScore in ensembleRaw]

        self.METRICS.updateMetrics(classification, error, ensembleError, ensembleRaw, ensembleWeighted)

        # We need to store the sample in one of:
        # core/edge/gen-men-explicit/implicit
        # Need memory for each model...store in self.MODELS
        if self.MLEPConfig["allow_model_drift"]:
            if classify_mode == "implicit":
                #set label
                data.setLabel(classification)
            dataHasBeenAdded = False
            for model_name, model_distance in zip(ensembleModelNames, ensembleModelDistance):

                if model_distance < self.MODELS[model_name]["model"].getDataCharacteristic("delta_low"):
                    #self.MODELS[model_name]["memoryTracker"].addToMemory("core-mem-"+classify_mode, data)

                    self.MODELS[model_name]["memoryTracker"].addToMemory("edge-mem-"+classify_mode, data)
                    dataHasBeenAdded = True
                    # TODO handle core centroids...
                elif model_distance > self.MODELS[model_name]["model"].getDataCharacteristic("delta_high"):
                    pass
                else:
                    self.MODELS[model_name]["memoryTracker"].addToMemory("edge-mem-"+classify_mode, data)
            if not dataHasBeenAdded:
                self.MEMTRACK.addToMemory("gen-mem-"+classify_mode, data)
                

        # perform explicit drift detection and update (if classift mode is explicit)
        if self.MLEPConfig["allow_explicit_drift"]:
            # send the input appropriate for the drift mode
            if classify_mode == "explicit":
                driftDetected = self.EXPLICIT_DRIFT_TRACKER.detect(self.METRICS.get(self.MLEPConfig["drift_metrics"][self.MLEPConfig["explicit_drift_mode"]]))
                if driftDetected:
                    io_utils.std_flush(self.MLEPConfig["explicit_drift_mode"], "has detected drift at", len(self.METRICS.get("all_errors")), "samples. Resetting")
                    self.EXPLICIT_DRIFT_TRACKER.reset()
                    self.MLEPModelBasedUpdate()


        # perform implicit/unlabeled drift detection and update. This is performed :
        if self.MLEPConfig["allow_unlabeled_drift"]:
            if self.MLEPConfig["unlabeled_drift_mode"] == "EnsembleDisagreement":
                if classify_mode == "explicit":
                    self.MEMTRACK.addToMemory(memory_name="unlabeled_drift", data=data)
                    if error:
                        self.MEMTRACK.addToMemory(memory_name="unlabeled_errors", data=data)
                if classify_mode == "implicit":
                    driftDetected = self.UNLABELED_DRIFT_TRACKER.detect(self.METRICS.get(self.MLEPConfig["drift_metrics"][self.MLEPConfig["unlabeled_drift_mode"]]))
                    if driftDetected:
                        io_utils.std_flush(self.MLEPConfig["unlabeled_drift_mode"], "has detected drift at", len(self.METRICS.get("all_errors")), "samples. Resetting")
                        self.UNLABELED_DRIFT_TRACKER.reset()
            if self.MLEPConfig["unlabeled_drift_mode"] == "KullbackLeibler":
                pass
                
                self.MEMTRACK.addToMemory("unlabeled_drift", data)
                _encoded = localEncoder["w2v-main"]
                _distance = PrimaryMetrics(_encoded, self.kullback["centroid"])
                self.kullback["secondary_distribution"].update(_distance)
                raw_val = self.UNLABELED_DRIFT_TRACKER.detect(_distance, self.kullback["secondary_distribution"])
                self.kullback["raw_vals"].append(raw_val)

                if self.MEMTRACK.memorySize("unlabeled_drift") > self.kullback["process_length"]:
                    if self.kullback["raw_vals"][-1]>self.kullback["threshold"]:
                        training_data = self.MEMTRACK.transferMemory("unlabeled_drift")
                        self.MEMTRACK.clearMemory("unlabeled_drift")

                        X_train = self.ENCODERS["w2v-main"].batchEncode(training_data.getData())
                        self.kullback["centroid"] = self.ENCODERS["w2v-main"].getCentroid(X_train)

                        distr = ZonedDistribution.ZonedDistribution(nBins=40,alpha=self.ALPHA, metric_callback=PrimaryMetrics, distribution_callback=PrimaryDistribution)
                        distr.build(centroid=self.kullback["centroid"], data=X_train)
                        self.UNLABELED_DRIFT_TRACKER.reset(distr.distribution)
                        
                        self.kullback["secondary_distribution"] = deepcopy(distr.distribution)
                        io_utils.std_flush(self.MLEPConfig["unlabeled_drift_mode"], "has detected drift at", len(self.METRICS.get("all_errors")), "samples. Resetting")

                        self.MLEPModelBasedUpdate()

                

        if self.MLEPConfig["allow_model_confidence"]:
            for idx,_name in enumerate(ensembleModelNames):
                modelDrifting = self.MODELS[_name]["model"].isDrifting()
                if modelDrifting:
                    io_utils.std_flush(_name, "has detected drift at", len(self.METRICS.get("all_errors")), "samples. Resetting")
                    
        self.saveClassification(classification)
        self.saveEnsemble(ensembleModelNames)

        return classification


    def saveClassification(self, classification):
        self.LAST_CLASSIFICATION = classification
    def saveEnsemble(self,ensembleModelNames):
        self.LAST_ENSEMBLE = [item for item in ensembleModelNames]


    def setUpMemories(self,):
        io_utils.std_flush("\tStarted setting up memories at", time_utils.readable_time())
        self.MEMTRACK = MemoryTracker.MemoryTracker()

        
        self.MEMTRACK.addNewMemory(memory_name="gen-mem-explicit",memory_store="memory")
        self.MEMTRACK.addNewMemory(memory_name="gen-mem-implicit",memory_store="memory")
        io_utils.std_flush("\t\tAdded General Memory memory")

        """
        if self.MLEPConfig["allow_explicit_drift"]:
            self.MEMTRACK.addNewMemory(memory_name="explicit_drift",memory_store="memory")
            self.MEMTRACK.addNewMemory(memory_name="explicit_errors",memory_store="memory")
            io_utils.std_flush("\t\tAdded explicit drift memory")

        """
        if self.MLEPConfig["allow_unlabeled_drift"]:
            self.MEMTRACK.addNewMemory(memory_name="unlabeled_drift",memory_store="memory")
            io_utils.std_flush("\t\tAdded unlabeled drift memory")
        io_utils.std_flush("\tFinished setting up memories at", time_utils.readable_time())

    





    def createModelId(self, timestamp, pipelineName, fscore):
        strA = time_utils.time_to_id(timestamp)
        strB = time_utils.time_to_id(hash(pipelineName)%self.HASHMAX)
        strC = time_utils.time_to_id(fscore, 5)
        return "_".join([strA,strB,strC])
    
    def addAugmentation(self,augmentation):
        self.AUGMENT = augmentation
    
    def getValidPipelines(self,):
        """ get pipelines that are, well, valid """
        return {item:self.config["pipelines"][item] for item in self.config["pipelines"] if self.config["pipelines"][item]["valid"]}

    def getValidEncoders(self,):
        """ get valid encoders """
        # iterate through pipelines, get encoders that are valid, and return those from config->encoders
        return {item:self.config["encoders"][item] for item in {self.MLEPPipelines[_item]["encoder"]:1 for _item in self.MLEPPipelines}}

    def getValidModels(self,):
        """ get valid models """    
        ensembleModelNames = [item for item in self.ModelTracker.get(self.MLEPConfig["select_method"])]
        return ensembleModelNames
    def shutdown(self):
        self.ModelDB.close()
    
"""
{
    "name": "Python: SimpleExperiment",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/mlep/experiments/single_experiment.py",
    "console": "integratedTerminal",
    "args": [
        "test",
        "--allow_explicit_drift", "True",
        "--allow_update_schedule", "True"
    ],
    "cwd":"${workspaceFolder}/mlep/experiments/"
},
"""
