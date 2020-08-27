import os, time
import pdb
from mlep.utils import io_utils, sqlite_utils, time_utils
import sqlite3


class MLEPDriftAdaptor():
    def __init__(self, config_dict, safe_mode=True):
        """Initialize the learning server.

        config_dift -- [dict] JSON Configuration dictionary
        """
        io_utils.std_flush("Initializing MLEP...")

        # In safe mode, existing ./.MLEPServer is not deleted
        self.SAFE_MODE=safe_mode
        
        self.setUpCoreVars()
        self.configureSqlite()
        self.loadConfig(config_dict)
        self.initializeTimers()
        self.setupDirectoryStructure()
        self.setupDbConnection()
        self.initializeDb()
        self.setUpEncoders()
        self.setUpMetrics()
        self.setUpExplicitDriftTracker()
        self.setUpUnlabeledDriftTracker()
        self.setUpMemories()
        self.setUpModelTracker()

        io_utils.std_flush("Finished initializing MLEP...")

    def setUpCoreVars(self,):
        self.KNOWN_EXPLICIT_DRIFT_CLASSES = ["LabeledDriftDetector"]
        self.KNOWN_UNLABELED_DRIFT_CLASSES = ["UnlabeledDriftDetector"]
        
        # Setting of 'hosted' models + data cetroids
        self.MODELS = {}
        self.CENTROIDS={}

        # Augmenter
        self.AUGMENT = None

        # Statistics
        self.LAST_CLASSIFICATION = 0
        self.LAST_ENSEMBLE = []

        import sys
        self.HASHMAX = sys.maxsize


    def setUpModelTracker(self,):
        io_utils.std_flush("\tStarted setting up model tracking at", time_utils.readable_time())
        
        self.MODEL_TRACK = {}
        self.MODEL_TRACK["recent"] = []
        self.MODEL_TRACK["recent-new"] = []
        self.MODEL_TRACK["recent-new"] = []
        self.MODEL_TRACK["recent-updates"] = []
        self.MODEL_TRACK["historical"] = []
        self.MODEL_TRACK["historical-new"] = []
        self.MODEL_TRACK["historical-updates"] = []
        self.MODEL_TRACK["train"] = []

        io_utils.std_flush("\tFinished setting up model tracking at", time_utils.readable_time())

    def updateMetrics(self, classification, error, ensembleError, ensembleRaw, ensembleWeighted):
        self.METRICS["all_errors"].append(error)
        self.METRICS['classification'] = classification
        self.METRICS['error'] = error
        self.METRICS['ensembleRaw'] = ensembleRaw
        self.METRICS['ensembleWeighted'] = ensembleWeighted
        self.METRICS['ensembleError'] = ensembleError


    def setUpMetrics(self,):
        io_utils.std_flush("\tStarted setting up metrics tracking at", time_utils.readable_time())

        self.METRICS={}
        self.METRICS["all_errors"] = []
        self.METRICS['classification'] = None
        self.METRICS['error'] = None
        self.METRICS['ensembleRaw'] = []
        self.METRICS['ensembleWeighted'] = []
        self.METRICS['ensembleError'] = []

        io_utils.std_flush("\tFinished setting up metrics tracking at", time_utils.readable_time())


    def setUpUnlabeledDriftTracker(self,):
        if self.MLEPConfig["allow_unlabeled_drift"]:
            io_utils.std_flush("\tStarted setting up unlabeled drift tracker at", time_utils.readable_time())
            
            if self.MLEPConfig["unlabeled_drift_class"] not in self.KNOWN_UNLABELED_DRIFT_CLASSES:
                raise ValueError("Unlabeled drift class '%s' in configuration is not part of any known Unlabeled Drift Classes: %s"%(self.MLEPConfig["unlabeled_drift_class"], str(self.KNOWN_UNLABELED_DRIFT_CLASSES)))
            if self.MLEPConfig["unlabeled_drift_mode"] != "EnsembleDisagreement":
                raise NotImplementedError()
            driftTracker = self.MLEPConfig["unlabeled_drift_mode"]
            driftModule = self.MLEPConfig["unlabeled_drift_class"]
            driftArgs = self.MLEPConfig["drift_args"] if "drift_args" in self.MLEPConfig else {}
            driftModuleImport = __import__("mlep.drift_detector.%s.%s"%(driftModule, driftTracker), fromlist=[driftTracker])
            driftTrackerClass = getattr(driftModuleImport,driftTracker)
            self.UNLABELED_DRIFT_TRACKER = driftTrackerClass(**driftArgs)

            io_utils.std_flush("\tFinished setting up unlabeled drift tracker at", time_utils.readable_time())
        else:
            self.UNLABELED_DRIFT_TRACKER = None
            io_utils.std_flush("\tUnlabeled drift tracker not used in this run", time_utils.readable_time())

    def setUpExplicitDriftTracker(self,):
        if self.MLEPConfig["allow_explicit_drift"]:
            io_utils.std_flush("\tStarted setting up explicit drift tracker at", time_utils.readable_time())
            
            if self.MLEPConfig["explicit_drift_class"] not in self.KNOWN_EXPLICIT_DRIFT_CLASSES:
                raise ValueError("Explicit drift class '%s' in configuration is not part of any known Explicit Drift Classes: %s"%(self.MLEPConfig["explicit_drift_class"], str(self.KNOWN_EXPLICIT_DRIFT_CLASSES)))

            driftTracker = self.MLEPConfig["explicit_drift_mode"]
            driftModule = self.MLEPConfig["explicit_drift_class"]
            driftArgs = self.MLEPConfig["drift_args"] if "drift_args" in self.MLEPConfig else {}
            driftModuleImport = __import__("mlep.drift_detector.%s.%s"%(driftModule, driftTracker), fromlist=[driftTracker])
            driftTrackerClass = getattr(driftModuleImport,driftTracker)
            self.EXPLICIT_DRIFT_TRACKER = driftTrackerClass(**driftArgs)

            io_utils.std_flush("\tFinished setting up explicit drift tracker at", time_utils.readable_time())
        else:
            self.EXPLICIT_DRIFT_TRACKER = None
            io_utils.std_flush("\tExplicit drift tracker not used in this run", time_utils.readable_time())


    def configureSqlite(self):
        """Configure SQLite to convert numpy arrays to TEXT when INSERTing, and TEXT back to numpy
        arrays when SELECTing."""
        io_utils.std_flush("\tStarted configuring SQLite at", time_utils.readable_time())

        import numpy as np
        sqlite3.register_adapter(np.ndarray, sqlite_utils.adapt_array)
        sqlite3.register_converter("array", sqlite_utils.convert_array)

        io_utils.std_flush("\tFinished configuring SQLite at", time_utils.readable_time())


    def loadConfig(self, config_dict):
        """Load JSON configuration file and initialize attributes.

        config_path -- [str] Path to the JSON configuration file.
        """
        io_utils.std_flush("\tStarted loading JSON configuration file at", time_utils.readable_time())

        self.config = config_dict
        self.MLEPConfig = self.config["config"]
        self.MLEPModels = self.config["models"]
        self.MLEPPipelines = self.getValidPipelines()
        self.MLEPEncoders = self.getValidEncoders()

        io_utils.std_flush("\tFinished loading JSON configuration file at", time_utils.readable_time())

    def initializeTimers(self):
        """Initialize time attributes."""
        io_utils.std_flush("\tStarted initializing timers at", time_utils.readable_time())

        # Internal clock of the server.
        self.overallTimer = None
        # Internal clock of the models.
        self.MLEPModelTimer = time.time()
        # Clock used to schedule filter generation and update.
        self.scheduledFilterGenerateUpdateTimer = 0
        # Schedule filter generation and update as defined in the configuration file or every 30
        # days.
        self.scheduledSchedule = self.MLEPConfig.get("update_schedule", 86400000 * 30)

        io_utils.std_flush("\tFinished initializing timers at", time_utils.readable_time())

    def setupDirectoryStructure(self):
        """Set up directory structure."""
        io_utils.std_flush("\tStarted setting up directory structure at", time_utils.readable_time())

        import shutil
        
        self.SOURCE_DIR = "./.MLEPServer"
        self.setups = ['models', 'data', 'modelSerials', 'db']
        self.DB_FILE = './.MLEPServer/db/MLEP.db'
        # Remove SOURCE_DIR if it already exists and we are in normal mode
        if os.path.exists(self.SOURCE_DIR):
            if self.SAFE_MODE:
                raise RuntimeError("There is an existing ./.MLEPServer folder. Delete it or start MLEPServer without safe_mode=True")
            else:
                shutil.rmtree(self.SOURCE_DIR)
        # Make directory tree.
        os.makedirs(self.SOURCE_DIR)
        for directory in self.setups:
            os.makedirs(os.path.join(self.SOURCE_DIR, directory))

        io_utils.std_flush("\tFinished setting up directory structure at", time_utils.readable_time())

    def setupDbConnection(self):
        """Set up connection to a SQLite database."""
        io_utils.std_flush("\tStarted setting up database connection at", time_utils.readable_time())
            
        self.DB_CONN = None
        try:
            self.DB_CONN = sqlite3.connect(self.DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
        except sqlite3.Error as e:
            # TODO handle these
            print(e)

        io_utils.std_flush("\tFinished setting up database connection at", time_utils.readable_time())

    def initializeDb(self):
        """Create tables in a SQLite database."""

        io_utils.std_flush("\tStarted initializing database at", time_utils.readable_time())
        cursor = self.DB_CONN.cursor()
        cursor.execute("""Drop Table IF EXISTS Models""")
        cursor.execute("""
            CREATE TABLE Models(
                modelid         text,
                parentmodel     text,
                pipelineName    text,
                timestamp       real,
                data_centroid   array,
                trainingModel   text,
                trainingData    text,
                testData        text,
                precision       real,
                recall          real,
                fscore          real,
                type            text,
                active          integer,
                PRIMARY KEY(modelid),
                FOREIGN KEY(parentmodel) REFERENCES Models(trainingModel)
            )
        """)
        self.DB_CONN.commit()
        cursor.close()

        io_utils.std_flush("\tFinished initializing database at", time_utils.readable_time())

    def updateModelStore(self,):
        # These are models generated and updated in the prior update
        # Only generated models in prior update
        RECENT_NEW = self.getNewModelsSince(self.MLEPModelTimer)
        # Only update models in prior update
        RECENT_UPDATES = self.getUpdateModelsSince(self.MLEPModelTimer)
        # All models in prior update
        RECENT_MODELS = self.getModelsSince(self.MLEPModelTimer)
        
        # All models
        self.MODEL_TRACK["historical"] = self.getModelsSince()
        # All generated models
        self.MODEL_TRACK["historical-new"] = self.getNewModelsSince()
        # All update models
        self.MODEL_TRACK["historical-updates"] = self.getUpdateModelsSince()

        if len(RECENT_NEW) > 0:
            self.MODEL_TRACK["recent-new"] = [item for item in RECENT_NEW]
            #else fallback
        if len(self.MODEL_TRACK["recent-new"]) == 0:
            self.MODEL_TRACK["recent-new"] = [item for item in self.MODEL_TRACK["train"]]
        
        if len(RECENT_UPDATES) > 0:
            self.MODEL_TRACK["recent-updates"] = [item for item in RECENT_UPDATES]
        #else fallback
        if len(self.MODEL_TRACK["recent-updates"]) == 0:
            self.MODEL_TRACK["recent-updates"] = [item for item in self.MODEL_TRACK["recent-new"]]
                
        if len(RECENT_MODELS) > 0:
            self.MODEL_TRACK["recent"] = [item for item in RECENT_MODELS]
            #else don't change it
        if len(self.MODEL_TRACK["recent"]) == 0:
            self.MODEL_TRACK["recent"] = list(set(self.MODEL_TRACK["recent-updates"] + self.MODEL_TRACK["recent-new"]))
            

        if len(self.MODEL_TRACK["historical-updates"]) == 0:
            # No update models found. Fall back on Historical New
            self.MODEL_TRACK["historical-updates"] = [item for item in self.MODEL_TRACK["historical-new"]]
        
        # Update Model Timer
        self.MLEPModelTimer = time.time()

        # Clean display
        self.MODEL_TRACK["recent-new-display"] = [item[:item.find('_')] for item in self.MODEL_TRACK["recent-new"]]
        self.MODEL_TRACK["recent-updates-display"] = [item[:item.find('_')] for item in self.MODEL_TRACK["recent-updates"]]

        io_utils.std_flush("New Models: ", self.MODEL_TRACK["recent-new-display"])
        io_utils.std_flush("Update Models: ", self.MODEL_TRACK["recent-updates-display"])


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
                self.ENCODERS[encoder_config["name"]].failCondition(
                        **encoder_config["fail-args"])
                self.ENCODERS[encoder_config["name"]].setup(**encoder_config["args"])
                
        io_utils.std_flush("\tFinished setting up encoders at", time_utils.readable_time())

    def shutdown(self, deltree=True):
        # save models - because they are all heald in memory??
        # Access the save path
        # pick.dump models to that path
        pass
        if deltree:
            import shutil
            shutil.rmtree(self.SOURCE_DIR)
        self.closeDBConnection()

        

    def closeDBConnection(self,):
        try:
            self.DB_CONN.close()
        except sqlite3.Error:
            pass
    
    def enoughTimeElapsedBetweenUpdates(self,):
        return abs(self.overallTimer - self.scheduledFilterGenerateUpdateTimer) > self.scheduledSchedule

    def updateTime(self,timerVal):
        """ Manually updating time for experimental evaluation """

        self.overallTimer = timerVal
        
        # check if we are allowed to update -- .
        if not self.MLEPConfig["allow_update_schedule"]:
            return
    
        # Check scheduled time difference if there need to be updates
        if self.enoughTimeElapsedBetweenUpdates():
            # TODO change this to check if scheduledMemoryTrack exists
            if self.MEMTRACK.isEmpty(memory_name="scheduled"):
                io_utils.std_flush("Attempted update at", time_utils.ms_to_readable(self.overallTimer), ", but 0 data samples." ) 
            else:  
                # Select the right memory (all or errors)
                if self.MLEPConfig["schedule_update_mode"] == "all":
                    self.MLEPUpdate(memory_type="scheduled")
                elif self.MLEPConfig["schedule_update_mode"] == "errors":
                    self.MLEPUpdate(memory_type="scheduled_errors")
                else:
                    raise NotImplementedError()
            
            self.scheduledFilterGenerateUpdateTimer = self.overallTimer
    
    def MLEPUpdate(self,memory_type="scheduled"):
        if self.MEMTRACK.memorySize(memory_name=memory_type) < self.MLEPConfig["min_train_size"]:
            io_utils.std_flush("Attemped update using", memory_type, "-memory with", self.MEMTRACK.memorySize(memory_name=memory_type),"data samples. Failed due to requirement of", self.MLEPConfig["min_train_size"], "samples." )    
            return
            # TODO update the learning model itself to reject update with too few? Or let user handle this issue?
        io_utils.std_flush("Update using", memory_type, "-memory at", time_utils.ms_to_readable(self.overallTimer), "with", self.MEMTRACK.memorySize(memory_name=memory_type),"data samples." )
        # Get the training data from Memory
        TrainingData = self.getTrainingData(memory_type=memory_type)
        self.MEMTRACK.clearMemory(memory_name=memory_type)
        
        # Generate
        self.train(TrainingData)
        io_utils.std_flush("Completed", memory_type, "-memory based Model generation at", time_utils.readable_time())

        # update
        self.update(TrainingData,models_to_update=self.MLEPConfig["models_to_update"])
        io_utils.std_flush("Completed", memory_type, "-memory based Model Update at", time_utils.readable_time())

        # Now we update model store.
        self.updateModelStore()

    def getTrainingData(self, memory_type="scheduled"):
        """ Get the data in self.SCHEDULED_DATA_FILE """

        # need to load it as  BatchedModel...
        # (So, first scheduledDataFile needs to save stuff as BatchedModel...)

        # We load stuff from batched model
        # Then we check how many for each class
        # perform augmentation for the binary case
        import random
        scheduledTrainingData = None

        # TODO close the opened one before opening a read connection!!!!!
        scheduledTrainingData = self.MEMTRACK.transferMemory(memory_name = memory_type)
        scheduledTrainingData.load_by_class()

        if self.MEMTRACK.getClassifyMode() == "binary":
            negDataLength = scheduledTrainingData.class_size(0)
            posDataLength = scheduledTrainingData.class_size(1)
            if negDataLength < 0.8*posDataLength:
                io_utils.std_flush("Too few negative results. Adding more")
                if self.AUGMENT.class_size(0) < posDataLength:
                    # We'll need a random sampled for self.negatives BatchedLoad
                    scheduledTrainingData.augment_by_class(self.AUGMENT.getObjectsByClass(0), 0)
                else:
                    scheduledTrainingData.augment_by_class(random.sample(self.AUGMENT.getObjectsByClass(0), posDataLength-negDataLength), 0)
            elif negDataLength > 1.2 *posDataLength:
                # Too many negative data; we'll prune some
                io_utils.std_flush("Too many  negative samples. Pruning")
                scheduledTrainingData.prune_by_class(0,negDataLength-posDataLength)
                # TODO
            else:
                # Just right
                io_utils.std_flush("No augmentation necessary")
            # return combination of all classes
            return scheduledTrainingData
        else:
            raise NotImplementedError()


    # data is BatchedLocal
    def generatePipeline(self,data, pipeline):
        """ Generate a model using provided pipeline """
        
        # Simplified pipeline. First entry is Encoder; Second entry is the actual Model

        encoderName = pipeline["sequence"][0]
        pipelineModel = pipeline["sequence"][1]

        # Perform lookup
        pipelineModelName = self.MLEPModels[pipelineModel]["scriptName"]
        pipelineModelModule = __import__("mlep.learning_model.%s"%pipelineModelName, fromlist=[pipelineModelName])
        pipelineModelClass = getattr(pipelineModelModule,pipelineModelName)
        # data is a BatchedLocal
        model = pipelineModelClass()
        X_train = self.ENCODERS[encoderName].batchEncode(data.getData())
        centroid = self.ENCODERS[encoderName].getCentroid(X_train)
        y_train = data.getLabels()

        precision, recall, score = model.fit_and_test(X_train, y_train)

        return precision, recall, score, model, centroid

    def updatePipelineModel(self,data, modelSaveName, pipeline):
        """ Update a pipeline model using provided data """
        
        # Simplified pipeline. First entry is Encoder; Second entry is the actual Model
        
        # Need to set up encoder and pipeline using parent modelSaveName...

        encoderName = pipeline["sequence"][0]
        pipelineModel = pipeline["sequence"][1]

        # Perform lookup
        pipelineModelName = self.MLEPModels[pipelineModel]["scriptName"]
        pipelineModelModule = __import__("mlep.learning_model.%s"%pipelineModelName, fromlist=[pipelineModelName])
        pipelineModelClass = getattr(pipelineModelModule,pipelineModelName)

        model = pipelineModelClass()
        model.clone(self.MODELS[modelSaveName])

        X_train = self.ENCODERS[encoderName].batchEncode(data.getData())
        centroid = self.ENCODERS[encoderName].getCentroid(X_train)
        y_train = data.getLabels()

        precision, recall, score = model.update_and_test(X_train, y_train)

        return precision, recall, score, model, centroid
    
    def update(self, traindata, models_to_update='recent'):
        # for each model in self.MODELS
        # create a copy; rename details across everything
        # update copy
        # push details to DB
        # if copy's source is in MODEL_TRACK["recent"], add it to MODEL_TRACK["recent"] as well
        prune_val = 5
        if self.MLEPConfig["update_prune"] == "C":
            # Keep constant to new
            prune_val = len(self.MLEPPipelines)
        else:
            prune_val = int(self.MLEPConfig["update_prune"])
        
        temporaryModelStore = []
        modelSaveNames = [modelSaveName for modelSaveName in self.MODEL_TRACK[models_to_update]]
        modelDetails = self.getModelDetails(modelSaveNames) # Gets fscore, pipelineName, modelSaveName
        pipelineNameDict = self.getDetails(modelDetails, 'pipelineName', 'dict')
        for modelSaveName in modelSaveNames:
            # copy model
            # set up new model
            
            # Check if model can be updated (some models cannot be updated)
            if not self.MODELS[modelSaveName].isUpdatable():
                continue

            currentPipeline = self.MLEPPipelines[pipelineNameDict[modelSaveName]]
            precision, recall, score, pipelineTrained, data_centroid = self.updatePipelineModel(traindata, modelSaveName, currentPipeline)
            timestamp = time.time()
            modelIdentifier = self.createModelId(timestamp, currentPipeline["name"], score)
            modelSavePath = "_".join([currentPipeline["name"], modelIdentifier])
            trainDataSavePath = ""
            testDataSavePath = ""
            
            # We temporarily load to dictionary for sorting later.
            dicta={}
            dicta["name"] = modelSavePath
            dicta["MODEL"] = pipelineTrained
            dicta["CENTROID"] = data_centroid
            dicta["modelid"] = modelIdentifier
            dicta["parentmodelid"] = str(modelSaveName)
            dicta["pipelineName"] = str(currentPipeline["name"])
            dicta["timestamp"] = timestamp
            dicta["data_centroid"] = data_centroid
            dicta["training_model"] = str(modelSavePath)
            dicta["training_data"] = str(trainDataSavePath)
            dicta["test_data"] = str(testDataSavePath)
            dicta["precision"] = precision
            dicta["recall"] = recall
            dicta["score"] = score
            dicta["_type"] = str(currentPipeline["type"])
            dicta["active"] = 1

            temporaryModelStore.append(dicta)

        if len(temporaryModelStore) > prune_val:
            io_utils.std_flush("Pruning models -- reducing from", str(len(temporaryModelStore)),"to",str(prune_val),"update models." )
            # keep the highest scoring update models
            temporaryModelStore = sorted(temporaryModelStore, key=lambda k:k["score"], reverse=True)
            temporaryModelStore = temporaryModelStore[:prune_val]

        for item in temporaryModelStore:
            # save the model (i.e. host it)
            item["MODEL"].trackDrift(self.MLEPConfig["allow_model_confidence"])
            self.MODELS[item["name"]] = item["MODEL"]
            # Because we are simplifying this implementation, we don't actually have pipeline families. Every pipelien is part of the w2v family
            # So we can actually just store data_centroids locally
            self.CENTROIDS[item["name"]] = item["data_centroid"]
            # Now we save deets.

            self.insertModelToDb(modelid=item["modelid"], parentmodelid=item["parentmodelid"], pipelineName=item["pipelineName"],
                                timestamp=item["timestamp"], data_centroid=item["data_centroid"], training_model=item["training_model"], 
                                training_data=item["training_data"], test_data=item["test_data"], precision=item["precision"], recall=item["recall"], score=item["score"],
                                _type=item["_type"], active=item["active"])


    def insertModelToDb(self,modelid=None, parentmodelid=None, pipelineName=None,
                                timestamp=None, data_centroid=None, training_model=None, 
                                training_data=None, test_data=None, precision=None, recall=None, score=None,
                                _type=None, active=1):
        columns=",".join([  "modelid","parentmodel","pipelineName","timestamp","data_centroid",
                                "trainingModel","trainingData","testData",
                                "precision","recall","fscore",
                                "type","active"])
            
        sql = "INSERT INTO Models (%s) VALUES " % columns
        sql += "(?,?,?,?,?,?,?,?,?,?,?,?,?)"
        cursor = self.DB_CONN.cursor()
        
        cursor.execute(sql, (   modelid,
                                parentmodelid,
                                pipelineName, 
                                timestamp,
                                data_centroid,
                                training_model,
                                training_data,
                                test_data,
                                precision,
                                recall,
                                score,
                                _type,
                                active) )
        
        self.DB_CONN.commit()
        cursor.close()

    # trainData is BatchedLocal
    def initialTrain(self,traindata,models= "all"):
        self.setUpInitialModels(traindata)
        self.MODEL_TRACK["train"] = self.getModelsSince()
        self.updateModelStore()

    # trainData is BatchedLocal
    def setUpInitialModels(self,traindata, models = 'all'):
        """ This is a function for initial training. Separated while working on data model. Will probably be recombined with self.train function later """
     
        for pipeline in self.MLEPPipelines:
            currentPipeline = self.MLEPPipelines[pipeline]
            # trainData is BatchedLocal
            precision, recall, score, pipelineTrained, data_centroid = self.generatePipeline(traindata, currentPipeline)
            timestamp = time.time()
            modelIdentifier = self.createModelId(timestamp, currentPipeline["name"],score) 
            modelSavePath = "_".join([currentPipeline["name"], modelIdentifier])
            trainDataSavePath = ""
            testDataSavePath = ""

            # save the model (i.e. host it)
            pipelineTrained.trackDrift(self.MLEPConfig["allow_model_confidence"])
            self.MODELS[modelSavePath] = pipelineTrained
            # Because we are simplifying this implementation, we don't actually have pipeline families. Every pipelien is part of the w2v family
            # So we can actually just store data_centroids locally
            self.CENTROIDS[modelSavePath] = data_centroid
            del pipelineTrained
            # Now we save deets.
            # Some cleaning
            self.insertModelToDb(modelid=modelIdentifier, parentmodelid=None, pipelineName=str(currentPipeline["name"]),
                                timestamp=timestamp, data_centroid=data_centroid, training_model=str(modelSavePath), 
                                training_data=str(trainDataSavePath), test_data=str(testDataSavePath), precision=precision, recall=recall, score=score,
                                _type=str(currentPipeline["type"]), active=1)



    def train(self,traindata, models = 'all'):
        # for each modelType in modelTypes
        #   for each encodingType (just 1)
        #       Create sklearn model using default details
        #       then train sklearn model using encoded data
        #       precision, recall, score, model = self.generate(encoder, traindata, model)
        #       push details to ModelDB
        #       save model to file using ID as filename.model -- serialized sklearn model
        
        

        # First load the Model configurations - identify what models exist
        
        for pipeline in self.MLEPPipelines:
            
            
            # We make the simplified assumption that all encoders are the same (pretrained w2v). 
            # So we don't have to handle pipeline families at this point for the distance function (if implemented)
            # Also, since our models are small-ish, we can make do by hosting models in memory
            # Production implementation (and going forward), models would be hosted as an API endpoint until "retirement"

            #io_utils.std_flush("Setting up", currentEncoder["name"], "at", time_utils.readable_time())
            
            # set up pipeline
            currentPipeline = self.MLEPPipelines[pipeline]
            precision, recall, score, pipelineTrained, data_centroid = self.generatePipeline(traindata, currentPipeline)
            timestamp = time.time()
            modelIdentifier = self.createModelId(timestamp, currentPipeline["name"],score) 
            modelSavePath = "_".join([currentPipeline["name"], modelIdentifier])
            trainDataSavePath = ""
            testDataSavePath = ""

            # save the model (i.e. host it)
            pipelineTrained.trackDrift(self.MLEPConfig["allow_model_confidence"])
            self.MODELS[modelSavePath] = pipelineTrained
            # Because we are simplifying this implementation, we don't actually have pipeline families. Every pipelien is part of the w2v family
            # So we can actually just store data_centroids locally
            self.CENTROIDS[modelSavePath] = data_centroid
            del pipelineTrained
            # Now we save deets.
            # Some cleaning
            
            self.insertModelToDb(modelid=modelIdentifier, parentmodelid=None, pipelineName=str(currentPipeline["name"]),
                                timestamp=timestamp, data_centroid=data_centroid, training_model=str(modelSavePath), 
                                training_data=str(trainDataSavePath), test_data=str(testDataSavePath), precision=precision, recall=recall, score=score,
                                _type=str(currentPipeline["type"]), active=1)


    def createModelId(self, timestamp, pipelineName, fscore):
        strA = time_utils.time_to_id(timestamp)
        strB = time_utils.time_to_id(hash(pipelineName)%self.HASHMAX)
        strC = time_utils.time_to_id(fscore, 5)
        return "_".join([strA,strB,strC])
        
    
    def addAugmentation(self,augmentation):
        self.AUGMENT = augmentation



    def getModelsSince(self, _time = None):
        cursor = self.DB_CONN.cursor()
        if _time is None:
            # We are getting ALL models
            sql = "select trainingModel from Models"
        else:
            # We are getting models since a time
            sql = "select trainingModel from Models where timestamp > %s" % _time
        
        cursor.execute(sql)
        tupleResults = cursor.fetchall()
        cursor.close()
        return [item[0] for item in tupleResults]

    def getNewModelsSince(self, _time = None):
        cursor = self.DB_CONN.cursor()
        if _time is None:
            # We are getting ALL models
            sql = "select trainingModel from Models where parentmodel IS NULL"
        else:
            # We are getting models since a time
            sql = "select trainingModel from Models where timestamp > %s and parentmodel IS NULL" % _time
        
        cursor.execute(sql)
        tupleResults = cursor.fetchall()
        cursor.close()
        return [item[0] for item in tupleResults]
        
        
    def getUpdateModelsSince(self, _time = None):
        cursor = self.DB_CONN.cursor()
        if _time is None:
            # We are getting ALL models
            sql = "select trainingModel from Models where parentmodel IS NOT NULL"
        else:
            # We are getting models since a time
            sql = "select trainingModel from Models where timestamp > %s and parentmodel IS NOT NULL" % _time
        
        cursor.execute(sql)
        tupleResults = cursor.fetchall()
        cursor.close()
        return [item[0] for item in tupleResults]

    def getModelDetails(self,ensembleModelNames, toGet = None):
        cursor = self.DB_CONN.cursor()
        if toGet is None:
            toGet = ["trainingModel","fscore","pipelineName"]
        sql = "select " + ",".join(toGet) + " from Models where trainingModel in ({seq})".format(seq=",".join(["?"]*len(ensembleModelNames)))
        cursor.execute(sql,ensembleModelNames)
        tupleResults = cursor.fetchall()
        cursor.close()
        dictResults = {}
        for entry in tupleResults:
            dictResults[entry[0]] = {}
            for idx,val in enumerate(toGet):
                dictResults[entry[0]][val] = entry[idx]
        return dictResults

    def getDetails(self,dataDict,keyVal,_format, order=None):
        if _format == "list":
            if order is None:
                # We need the order for lists
                raise RuntimeError("No order provided for getDetails with 'list' format")
            details = []
            details = [dataDict[item][keyVal] for item in order]
            return details
        if _format == "dict":
            details = {item:dataDict[item][keyVal] for item in dataDict}
            return details

    def getPipelineToModel(self,):
        cursor = self.DB_CONN.cursor()
        sql = "select pipelineName, trainingModel, fscore from Models"
        cursor.execute(sql)
        tupleResults = cursor.fetchall()
        cursor.close()
        dictResults = {}
        for entry in tupleResults:
            if entry[0] not in dictResults:
                dictResults[entry[0]] = []
            # entry[0] --> pipelineName
            # entry[1] --> trainingModel        item[0] in step 3 upon return
            # entry[2] --> fscore               item[1] in step 3 upon return
            dictResults[entry[0]].append((entry[1], entry[2]))
        return dictResults
    
    def getValidPipelines(self,):
        """ get pipelines that are, well, valid """
        return {item:self.config["pipelines"][item] for item in self.config["pipelines"] if self.config["pipelines"][item]["valid"]}

    def getValidEncoders(self,):
        """ get valid encoders """
        # iterate through pipelines, get encoders that are valid, and return those from config->encoders
        return {item:self.config["encoders"][item] for item in {self.MLEPPipelines[_item]["encoder"]:1 for _item in self.MLEPPipelines}}

    def getValidModels(self,):
        """ get valid models """    
        ensembleModelNames = [item for item in self.MODEL_TRACK[self.MLEPConfig["select_method"]]]
        return ensembleModelNames
    
    def getTopKPerformanceModels(self,ensembleModelNames):
        # basic optimization:
        if self.MLEPConfig["kval"] >= len(ensembleModelNames):
            pass 
        else:
            modelDetails = self.getModelDetails(ensembleModelNames)
            weights = self.getDetails(modelDetails, 'fscore', 'list', order=ensembleModelNames)
            mappedWeights = zip(weights, ensembleModelNames)
            mappedWeights = sorted(mappedWeights, key=lambda tup:tup[0], reverse=True) # pylint: disable=no-member
            # Now we have models sorted by performance. Truncate
            mappedWeights = mappedWeights[:self.MLEPConfig["kval"]]    # pylint: disable=unsubscriptable-object
            ensembleModelPerformance = [None]*self.MLEPConfig["kval"]
            ensembleModelNames = [None]*self.MLEPConfig["kval"]
            for idx,_item in mappedWeights:
                ensembleModelNames[idx] = _item[1]
                ensembleModelPerformance[idx] = _item[0]
        return ensembleModelNames, ensembleModelPerformance

    def getTopKNearestModels(self,ensembleModelNames, data):
        #data is a DataSet object
        ensembleModelPerformance = None
        ensembleModelDistance = None
        # find top-k nearest centroids
        k_val = self.MLEPConfig["kval"]
        # Basic optimization:
        # for tests:    if False and k_val >= len(ensembleModelNames):
        # regular       if k_val >= len(ensembleModelNames):
        if k_val >= len(ensembleModelNames):
            pass
        else:

            # We have the k_val
            # Normally, this part would use a DataModel construct (not implemented) to get the appropriate "distance" model for a specific data point
            # But we make the assumption that all data is encoded, etc, etc, and use the encoders to get distance.

            # 1. First, collect list of Encoders
            # 2. Then create mapping of encoders -- model_save_path
            # 3. Then for each encoder, find k-closest model_save_path that is part of valid-list (??)
            # 4. Put them all together and sort on performance
            # 5. Return top-k (so two levels of k, finally returning k models)

            # dictify for O(1) check
            ensembleModelNamesValid = {item:1 for item in ensembleModelNames}
            # 1. First, collect list of Encoders -- model mapping
            pipelineToModel = self.getPipelineToModel()
            
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
                # Find distance to all appropriate models
                # Then sort and take top-5
                # This can probably be optimized to not perform unneeded Distance calculations (if, e.g. two models have the same training dataset - something to consider)
                # kCPE[E] = [ (NORM(encoded - centroid(modelName), performance, modelName) ... ]
                #   NOTE --> We need to make sure item[0] (modelName)
                #   NOTE --> item[1] : fscore
                # Add additional check for whether modelName is in list of validModels (ensembleModelNames)

                # Use encoder specific distance metric
                # But how to normalize results????? 
                # So w2v cosine_similarity is between 0 and 1
                # BUT, bow distance is, well, distance. It might be any value
                # Allright, need to 
                #np.linalg.norm(_encodedData-self.CENTROIDS[item[0]])
                kClosestPerEncoder[_encoder]=[(self.ENCODERS[_encoder].getDistance(_encodedData, self.CENTROIDS[item[0]]), item[1], item[0]) for item in encoderToModel[_encoder] if item[0] in ensembleModelNamesValid]
                # Default sort on first param (norm); sort on distance - smallest to largest
                # tup[0] --> norm
                # tup[1] --> fscore
                # tup[2] --> modelName
                # Sorting by tup[0] --> norm
                # TODO normalize distance to 0:1
                # Need to do this during centroid construction
                # for the training data, in addition to storing centroid, store furthest data point distance
                # Then during distance getting, we compare the distance to max_distance in getDistance() and return a 0-1 normalized. Anything outside max_distance is floored to 1.
                kClosestPerEncoder[_encoder].sort(key=lambda tup:tup[0])
                # Truncate to top-k
                kClosestPerEncoder[_encoder] = kClosestPerEncoder[_encoder][:k_val]

            # 4. Put them all together and sort on performance
            # distance weighted performance
            kClosest = []
            for _encoder in kClosestPerEncoder:
                kClosest+=kClosestPerEncoder[_encoder]
            # Sorting by tup[1] --> fscore
            kClosest.sort(key=lambda tup:tup[1], reverse=True)

            # 5. Return top-k (so two levels of k, finally returning k models)
            # item[0] --> norm
            # item[1] --> fscore
            # item[2] --> modelName
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
        MLEPDriftAdaptor's classifier. 
        
        The MLEPDriftAdaptor's classifier performs model retrieval, distance calculation, and drift updates.

        Args:
            data -- a single data sample for classification
            classify_mode -- "explicit" if data is supposed to have a label. "implicit" if data is unlabeled. Since this is an experimentation framework, data is technically supposed to have a label. The distinction is on whether the label is something MLEPDriftAdaptor would have access to during live operation. "explicit" refers to this case, while "implicit" refers to the verso.

        Returns:
            classification -- INT -- currently only binary classification is supported
            
        """
        # First set up list of correct models
        ensembleModelNames = self.getValidModels()
        # Now that we have collection of candidaate models, we use filter_select to decide how to choose the right model
        if self.MLEPConfig["filter_select"] == "top-k":
            # sort on top-k best performers
            ensembleModelNames, ensembleModelPerformance = self.getTopKPerformanceModels(ensembleModelNames)
        elif self.MLEPConfig["filter_select"] == "nearest":
            # do knn
            ensembleModelNames, ensembleModelPerformance, ensembleModelDistance = self.getTopKNearestModels(ensembleModelNames, data)
        elif self.MLEPConfig["filter_select"] == "no-filter":
            # Use all models in ensembleModelNames as ensemble
            pass
        else:
            # Default - use all (no-filter)
            pass
            
        

        # Given ensembleModelNames, use all of them as part of ensemble
        # Run the sqlite query to get model details
        modelDetails = self.getModelDetails(ensembleModelNames)
        if self.MLEPConfig["weight_method"] == "performance":
            if ensembleModelPerformance is not None:
                weights = ensembleModelPerformance
            else:
                # request DB for performance (f-score)
                weights = self.getDetails(modelDetails, 'fscore', 'list', order=ensembleModelNames)
            sumWeights = sum(weights)
            weights = [item/sumWeights for item in weights]
        elif self.MLEPConfig["weight_method"] == "unweighted":
            weights = len(ensembleModelNames)*[1.0/len(ensembleModelNames)]
        else:
            weights = len(ensembleModelNames)*[1.0/len(ensembleModelNames)]
        

        # TODO; Another simplification for this implementation. Assume binary classifier, and have built in Ensemble weighting
        # Yet another simplification - single encoder.
        # Production environment - MLEP will use the 'type' field of pipeline
        # Binary - between 0 and 1. Trivial. If weighted average  is >0.5, it's 1, else, it's 0
        # Multiclass - Weighted average. if greater than [INT].5, round up. else round down
        # Regression - weighted average
        # All members of modellist MUST be of the same type. You can't mix binary and multiclass

        # Get encoder types in ensembleModelNames                       
        # build local dictionary of data --> encodedVersion             
        pipelineNameDict = self.getDetails(modelDetails, 'pipelineName', 'dict')
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
            # use the prescribed enc; ensembleModelNames are the modelSaveFile
            # We need the pipeline each is associated with (so that we can extract front-loaded encoder)
            locally_encoded_data = localEncoder[self.MLEPPipelines[pipelineNameDict[_name]]["sequence"][0]]
            # So we get the model name, access the pipeline name from pipelineNameDict
            # Then get the encodername from sequence[0]
            # Then get the locally encoded thingamajig of the data
            # And pass it into predict()
            cls_ = None
            # for regular mode
            if not self.MLEPConfig["allow_model_confidence"]:
                cls_=self.MODELS[_name].predict(locally_encoded_data)
            
            else:
                # TODO for model-drift mode, we first check if the data point is ground-truth stream or a prediction stream
                #  -- if getLabel() on data returns None, it is a prediction stream
                # -- if getLabel() on data returns a value, it is a ground-truth stream
                
                # TODO getLabel() returns value
                # if explicit_drift_mode is allowed
                # cls = self.MODELS[_name].predict("explicit", y_label = label) -- > this will perform drift detection internally
                if classify_mode == "explicit":
                    # TODO TODO TODO ALERT ALERT ALERT NOT FUNCTIONAL
                    cls_=self.MODELS[_name].predict(locally_encoded_data, mode="explicit", y_label = data.getLabel())



                # TODO getLabel() returns None
                # if implicit drift mode is allowed
                # cls = self.MODELS[_name].predict("implicit") --> also perform drift detection internally
                if classify_mode == "implicit":
                    # TODO TODO TODO ALERT ALERT ALERT NOT FUNCTIONAL
                    cls_=self.MODELS[_name].predict(locally_encoded_data, mode="implicit")

            ensembleWeighted[idx] = float(weights[idx]*cls_)
            ensembleRaw[idx] = float(cls_)

        # Assume binary. We'll deal with others later
        classification = sum(ensembleWeighted)
        classification =  0 if classification < 0.5 else 1
        

        error = 1 if classification != data.getLabel() else 0
        ensembleError = [(1 if ensembleRawScore != data.getLabel() else 0) for ensembleRawScore in ensembleRaw]

        self.updateMetrics(classification, error, ensembleError, ensembleRaw, ensembleWeighted)

        # add to scheduled memory if this is explicit data
        
        if self.MLEPConfig["allow_update_schedule"]:
            if classify_mode == "explicit":
                self.MEMTRACK.addToMemory(memory_name="scheduled", data=data)
                if error:
                    self.MEMTRACK.addToMemory(memory_name="scheduled_errors", data=data)
                # No drift detection necessary
                # No MLEPUpdate necessary

        # perform explicit drift detection and update (if classift mode is explicit)
        if self.MLEPConfig["allow_explicit_drift"]:
            # send the input appropriate for the drift mode
            # shuld be updated to be more readble; update so that users can define their own drift tracking method
            if classify_mode == "explicit":
                # add error -- 
                self.MEMTRACK.addToMemory(memory_name="explicit_drift", data=data)
                if error:
                    self.MEMTRACK.addToMemory(memory_name="explicit_errors", data=data)

                driftDetected = self.EXPLICIT_DRIFT_TRACKER.detect(self.METRICS[self.MLEPConfig["drift_metrics"][self.MLEPConfig["explicit_drift_mode"]]])
                if driftDetected:
                    io_utils.std_flush(self.MLEPConfig["explicit_drift_mode"], "has detected drift at", len(self.METRICS["all_errors"]), "samples. Resetting")
                    self.EXPLICIT_DRIFT_TRACKER.reset()
                    
                    # perform drift update (big whoo)
                    # perform update with the correct memory type
                    if self.MLEPConfig["explicit_update_mode"] == "all":
                        self.MLEPUpdate(memory_type="explicit_drift")
                    elif self.MLEPConfig["explicit_update_mode"] == "errors":
                        self.MLEPUpdate(memory_type="explicit_errors")
                    else:
                        raise NotImplementedError()
                

        # perform implicit/unlabeled drift detection and update. This is performed :
        if self.MLEPConfig["allow_unlabeled_drift"]:
            # send the input appropriate for the drift mode
            # shuld be updated to be more readble; update so that users can define their own drift tracking method
            # Add to memory only if explicit, else perform drift detection
            if classify_mode == "explicit":
                self.MEMTRACK.addToMemory(memory_name="unlabeled_drift", data=data)
                if error:
                    self.MEMTRACK.addToMemory(memory_name="unlabeled_errors", data=data)
            if classify_mode == "implicit":
                if self.MLEPConfig["unlabeled_drift_mode"] == "EnsembleDisagreement":

                    driftDetected = self.UNLABELED_DRIFT_TRACKER.detect(self.METRICS[self.MLEPConfig["drift_metrics"][self.MLEPConfig["unlabeled_drift_mode"]]])
                    if driftDetected:
                        io_utils.std_flush(self.MLEPConfig["unlabeled_drift_mode"], "has detected drift at", len(self.METRICS["all_errors"]), "samples. Resetting")
                        self.UNLABELED_DRIFT_TRACKER.reset()
                        
                        # perform drift update (big whoo)
                        # perform update with the correct memory type
                        # TODO uncomment this -- for now we are just checking if drift is being detected....
                        """
                        if self.MLEPConfig["unlabeled_update_mode"] == "all":
                            self.MLEPUpdate(memory_type="unlabeled_drift")
                        elif self.MLEPConfig["unlabeled_update_mode"] == "errors":
                            self.MLEPUpdate(memory_type="unlabeled_errors")
                        else:
                            raise NotImplementedError()
                        """
                else:
                    raise NotImplementedError()

        if self.MLEPConfig["allow_model_confidence"]:
            # TODO
            for idx,_name in enumerate(ensembleModelNames):
                # add data to proper memory (core-mem, gen-mem)
                # add data, if it doesn't fit either, to data-mem (from MEMORY_TRACK)
                pass

                # check if model is drifting
                # if so use core-mem and gen-mem to update the model.
                pass
                modelDrifting = self.MODELS[_name].isDrifting()
                if modelDrifting:
                    io_utils.std_flush(_name, "has detected drift at", len(self.METRICS["all_errors"]), "samples. Resetting")
                    
            # TODO for this, for now, just output size of data-mem. See if this changes significantly, and use heuristics ?????
            # TODO Check if there is -- explicit drift -- OR -- enough data in data-mem to update
            # If so, generate new models on data-mem and add them to the pile
            # TODO TODO TODO Better way to check --> if more and more unlabeled samples are close to data-mem, then strengthen data-mem. 
        
        self.saveClassification(classification)
        self.saveEnsemble(ensembleModelNames)

        return classification


    def saveClassification(self, classification):
        self.LAST_CLASSIFICATION = classification
    def saveEnsemble(self,ensembleModelNames):
        self.LAST_ENSEMBLE = [item for item in ensembleModelNames]


    def setUpMemories(self,):

        io_utils.std_flush("\tStarted setting up memories at", time_utils.readable_time())
        import mlep.trackers.MemoryTracker as MemoryTracker
        self.MEMTRACK = MemoryTracker.MemoryTracker()

        if self.MLEPConfig["allow_update_schedule"]:
            self.MEMTRACK.addNewMemory(memory_name="scheduled",memory_store="local", memory_path="./.MLEPServer/data/")
            self.MEMTRACK.addNewMemory(memory_name="scheduled_errors",memory_store="local", memory_path="./.MLEPServer/data/")
            io_utils.std_flush("\t\tAdded scheduled memory")

        if self.MLEPConfig["allow_explicit_drift"]:
            self.MEMTRACK.addNewMemory(memory_name="explicit_drift",memory_store="local", memory_path="./.MLEPServer/data/")
            self.MEMTRACK.addNewMemory(memory_name="explicit_errors",memory_store="local", memory_path="./.MLEPServer/data/")
            io_utils.std_flush("\t\tAdded explicit drift memory")

        if self.MLEPConfig["allow_unlabeled_drift"]:
            self.MEMTRACK.addNewMemory(memory_name="unlabeled_drift",memory_store="local", memory_path="./.MLEPServer/data/")
            self.MEMTRACK.addNewMemory(memory_name="unlabeled_errors",memory_store="local", memory_path="./.MLEPServer/data/")
            io_utils.std_flush("\t\tAdded unlabeled drift memory")
        
        io_utils.std_flush("\tFinished setting up memories at", time_utils.readable_time())


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