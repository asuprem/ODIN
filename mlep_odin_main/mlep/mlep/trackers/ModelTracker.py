import time


class ModelTracker:
    def __init__(self,):
        self.MODEL_TRACK = {}
        self.MODEL_TRACK["recent"] = []
        self.MODEL_TRACK["recent-new"] = []
        self.MODEL_TRACK["recent-new"] = []
        self.MODEL_TRACK["recent-updates"] = []
        self.MODEL_TRACK["historical"] = []
        self.MODEL_TRACK["historical-new"] = []
        self.MODEL_TRACK["historical-updates"] = []
        self.MODEL_TRACK["train"] = []
        self.ModelTrackerTime = time.time()

    def get(self,_key):
        return self.MODEL_TRACK[_key]
    def _set(self,_key,vals):
        self.MODEL_TRACK[_key] = vals

    def updateModelStore(self,ModelDB):
        # These are models generated and updated in the prior update
        # Only generated models in prior update
        RECENT_NEW = ModelDB.getNewModelsSince(self.ModelTrackerTime)
        # Only update models in prior update
        RECENT_UPDATES = ModelDB.getUpdateModelsSince(self.ModelTrackerTime)
        # All models in prior update
        RECENT_MODELS = ModelDB.getModelsSince(self.ModelTrackerTime)
        
        # All models
        self.MODEL_TRACK["historical"] = ModelDB.getModelsSince()
        # All generated models
        self.MODEL_TRACK["historical-new"] = ModelDB.getNewModelsSince()
        # All update models
        self.MODEL_TRACK["historical-updates"] = ModelDB.getUpdateModelsSince()

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
        self.ModelTrackerTime = time.time()

        # Clean display
        #self.MODEL_TRACK["recent-new-display"] = [item[:item.find('_')] for item in self.MODEL_TRACK["recent-new"]]
        #self.MODEL_TRACK["recent-updates-display"] = [item[:item.find('_')] for item in self.MODEL_TRACK["recent-updates"]]

        #io_utils.std_flush("New Models: ", self.MODEL_TRACK["recent-new-display"])
        #io_utils.std_flush("Update Models: ", self.MODEL_TRACK["recent-updates-display"])

