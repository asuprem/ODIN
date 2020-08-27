import mlep.drift_detector.LabeledDriftDetector.LabeledDriftDetector as LabeledDriftDetector

class PageHinkley(LabeledDriftDetector.LabeledDriftDetector):
    def __init__(self,min_instances=30, delta = 0.005, threshold=50, alpha = 0.9999):

        self.min_instances = min_instances
        self.delta=delta
        self.threshold = threshold
        self.alpha = alpha
        self.mean = None
        self.n = None
        self.sum = None
        self.reset()


    
    def reset(self,):
        """ reset detector parameters, e.g. after drift has been detected """

        self.n = 0
        self.mean = 0.0
        self.sum = 0.0


    def detect(self,classification):
        """

        the classification label. page Hinkley works with any class classification

        returns -- is there drift
        """
        #required for DDM

        self.n+=1
        self.mean = self.mean + (classification - self.mean)/float(self.n)
        self.sum = max(0.0, self.alpha*self.sum+(classification - self.mean - self.delta))

        if self.n < self.min_instances:
            return False
        if self.sum > self.threshold:
            return True
        return False    
            
        

    