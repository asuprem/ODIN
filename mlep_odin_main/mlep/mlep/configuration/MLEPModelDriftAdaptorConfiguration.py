


class MLEPModelDriftAdaptorConfiguration:
    def __init__(self,):

        self.config = {
                "allow_explicit_drift": False,
                "explicit_drift_class":"LabeledDriftDetector",
                "explicit_drift_mode":"PageHinkley",
                "explicit_update_mode":"all",

                "allow_update_schedule": False,
                "update_schedule":2592000000,
                "schedule_update_mode":"all",

                "allow_model_drift":False,
                "allow_model_explicit_drift": False,
                "allow_model_implicit_drift": False,

                "allow_model_confidence":False,
                "allow_model_explicit_confidence": False,
                "allow_model_implicit_confidence": False,

                "allow_unlabeled_drift": False,
                "unlabeled_drift_class":"UnlabeledDriftDetector",
                "unlabeled_drift_mode":"EnsembleDisagreement",
                "unlabeled_update_mode":"all",

                
                "weight_method":"performance",
                "select_method":"historical",
                "filter_select":"nearest",
                "kval": 5,
                "update_prune":"C",
                "models_to_update":"recent",
                
                "min_train_size":50,
                "drift_metrics":{
                    "DDM":"error",
                    "EDDM":"error",
                    "EnsembleDisagreement":"ensembleRaw",
                    "PageHinkley":"classification", 
                    "ADWIN":"error"
                }}


    {
    "config":{
        
        "allow_explicit_drift": False,
        "explicit_drift_class":"LabeledDriftDetector",
        "explicit_drift_mode":"PageHinkley",
        "explicit_update_mode":"all",

        "allow_update_schedule": False,
        "update_schedule":2592000000,
        "schedule_update_mode":"all",

        "allow_model_drift":False,
        "allow_model_explicit_drift": False,
        "allow_model_implicit_drift": False,

        "allow_model_confidence":False,
        "allow_model_explicit_confidence": False,
        "allow_model_implicit_confidence": False,

        "allow_unlabeled_drift": False,
        "unlabeled_drift_class":"UnlabeledDriftDetector",
        "unlabeled_drift_mode":"EnsembleDisagreement",
        "unlabeled_update_mode":"all",

        
        "weight_method":"performance",
        "select_method":"historical",
        "filter_select":"nearest",
        "kval": 5,
        "update_prune":"C",
        "models_to_update":"recent",
        
        "min_train_size":50,
        "drift_metrics":{
            "DDM":"error",
            "EDDM":"error",
            "EnsembleDisagreement":"ensembleRaw",
            "PageHinkley":"classification", 
            "ADWIN":"error"
        }
        
    },

    "models": {
        "sgd" :{
            "name": "sgd",
            "desc": "Sklearn version of SVM, but batched - faster",
            "scriptName": "sklearnSGD"
        },
        "logreg" :{
            "name": "logreg",
            "desc": "Sklearn version of Logreg",
            "scriptName": "sklearnLogReg"
        },
        "decisiontree" :{
            "name": "decisiontree",
            "desc": "Sklearn version of Decision Tree",
            "scriptName": "sklearnDecisionTree"
        },
        "randforest" :{
            "name": "randforest",
            "desc": "Sklearn version of Random Forest",
            "scriptName": "sklearnRandomForest"
        }
    },



    "encoders": {
        "w2v-main":{
            "name": "w2v-main",
            "desc": "The Pretrained W2v Encoder. Needs access to the w2v model file (google something.bin)",
            "scriptName": "w2vGoogleNews",
            "args":{
            },
            "fail-args":{
            }
        },

        "bowDefault":{
            "name": "bowDefault",
            "desc": "Default bow Encoder using bow.model",
            "scriptName": "bowEncoder",
            "args":{
                "modelFileName":"bow.model"
            },
            "fail-args":{
                "rawFileName": "bow.txt",
                "modelFileName": "bow.model"
            }
        },

        "w2v-generic-10000":{
            "name": "w2v-generic-10000",
            "desc": "The Generic W2v Encoder. Needs access to the w2v model file",
            "scriptName": "w2vGeneric",
            "args":{
                "modelPath": "w2v-wiki-wikipedia-10000.bin", 
                "trainMode":"python"
            },
            "fail-args":{
                "dimensionSize":"10000",
                "seedName":"wikipedia"
            }
        },

        "w2v-generic-20000":{
            "name": "w2v-generic-20000",
            "desc": "The Generic W2v Encoder. Needs access to the w2v model file",
            "scriptName": "w2vGeneric",
            "args":{
                "modelPath": "w2v-wiki-wikipedia-20000.bin", 
                "trainMode":"python"
            },
            "fail-args":{
                "dimensionSize":"20000",
                "seedName":"wikipedia"
            }
        },

        "w2v-generic-5000":{
            "name": "w2v-generic-5000",
            "desc": "The Generic W2v Encoder. Needs access to the w2v model file",
            "scriptName": "w2vGeneric",
            "args":{
                "modelPath": "w2v-wiki-wikipedia-5000.bin", 
                "trainMode":"python"
            },
            "fail-args":{
                "dimensionSize":"5000",
                "seedName":"wikipedia"
            }
        }
    },

    "pipelines": {
        "pipelineA":{
            "name": "pipelineA",
            "sequence": ["w2v-main", "sgd"],
            "type":"binary",
            "encoder": "w2v-main",
            "valid":true
        },

        "pipelineB":{
            "name": "pipelineB",
            "sequence": ["w2v-main", "logreg"],
            "type":"binary",
            "encoder": "w2v-main",
            "valid":False
        },

        "pipelineC":{
            "name": "pipelineC",
            "sequence": ["w2v-main", "decisiontree"],
            "type":"binary",
            "encoder": "w2v-main",
            "valid":False
        },

        "pipelineD":{
            "name": "pipelineD",
            "sequence": ["w2v-main", "randforest"],
            "type":"binary",
            "encoder": "w2v-main",
            "valid":False
        },

        "pipelineE":{
            "name": "pipelineE",
            "sequence": ["w2v-generic-20000", "sgd"],
            "type":"binary",
            "encoder": "w2v-generic-20000",
            "valid":False
        }, 

        "pipelineF":{
            "name": "pipelineF",
            "sequence": ["w2v-generic-20000", "decisiontree"],
            "type":"binary",
            "encoder": "w2v-generic-20000",
            "valid":False
        }, 
        
        "w2v5ksgd":{
            "name": "w2v5ksgd",
            "sequence": ["w2v-generic-5000", "sgd"],
            "type":"binary",
            "encoder": "w2v-generic-5000",
            "valid":False
        }, 












        "bowDecision":{
            "name": "bowDecision",
            "sequence": ["bowDefault", "decisiontree"],
            "type":"binary",
            "encoder": "bowDefault",
            "valid":False
        },

        "bowlogreg":{
            "name": "bowlogreg",
            "sequence": ["bowDefault", "logreg"],
            "type":"binary",
            "encoder": "bowDefault",
            "valid":False
        },

        "bowrandforest":{
            "name": "bowrandforest",
            "sequence": ["bowDefault", "randforest"],
            "type":"binary",
            "encoder": "bowDefault",
            "valid":False
        },

        "bowsgd":{
            "name": "bowsgd",
            "sequence": ["bowDefault", "sgd"],
            "type":"binary",
            "encoder": "bowDefault",
            "valid":False
        }            
    }
}