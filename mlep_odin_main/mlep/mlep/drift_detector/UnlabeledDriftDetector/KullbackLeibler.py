import mlep.drift_detector.UnlabeledDriftDetector.UnlabeledDriftDetector as UnlabeledDriftDetector

class KullbackLeibler(UnlabeledDriftDetector.UnlabeledDriftDetector):
    """ This is an ensemble drift detector. It takes as input the ensemble of classifications, and uses distribution to check for drift"""


    def __init__(self,distribution_callback):
        
        self.p_distribution = distribution_callback
        self.q_distribution = None
        from math import log
        self.log = log
        self.raw_val = None

    def reset(self,distribution=None):
        """ reset detector parameters, e.g. after drift has been detected """
        if distribution is None:
            self.p_distribution = self.q_distribution
        else:
            self.p_distribution = distribution
        del self.q_distribution
        self.raw_val = None

    def detect(self,data, distribution_callback):
        """

        data 

        returns -- is there drift (T/F)
        """
        self.q_distribution = distribution_callback
        r_val = 0.0
        for val in self.p_distribution.dist_keys:
            p_val = self.p_distribution.get(val)
            q_val = self.q_distribution.get(val)

            if p_val == q_val:
                r_val += 0.0
            else:
                if p_val == 0:
                    # TODO Fix with correct davlue
                    p_val = 0.001
                if q_val == 0:
                    q_val = 0.001
                r_val += q_val * self.log(q_val/p_val)
        self.raw_val = r_val
        return self.raw_val
    