import mlep.drift_detector.LabeledDriftDetector.LabeledDriftDetector as LabeledDriftDetector


class EDDM(LabeledDriftDetector.LabeledDriftDetector):
    def __init__(self,min_instances=30, min_errors = 30, drift_level=0.9):

        from math import sqrt
        self.min_instances = min_instances
        self.min_e = min_errors
        self.drift_level = float(drift_level)
        self.n = None
        self.e = None
        self.d = None
        self.prevD = None
        self.mean = None
        self.s = None
        self.oob = None
        self.sqrt=sqrt
        self.reset()


    
    def reset(self,):
        """ reset detector parameters, e.g. after drift has been detected """

        self.n = 0
        self.e = 0
        self.d = 0.0
        self.prevD = 0.0
        self.mean = 0.0
        self.s = 0.0
        self.oob = 0.0


    def detect(self,error):
        """

        error - 1 if classification was incorrect; 0 if classification was correct

        returns -- is there drift
        """
        #required for DDM

        self.n +=1
        if not error:
            return False
        self.e+=1
        self.prevD = self.d
        self.d=self.n-1
        mean  = self.mean
        self.mean = self.mean + (self.d-self.prevD - self.mean) / float(self.e)
        self.s = self.s + (self.d-self.prevD - self.mean) * (self.d-self.prevD - mean)
        oob = self.mean + 2.0 * self.sqrt(self.s/float(self.e))

        if self.n < self.min_instances:
            return False
        
        if oob > self.oob:
            self.oob = oob
        else:
            if (self.e > self.min_e) and (oob/self.oob < self.drift_level):
                return True
        return False
            
            
        

    