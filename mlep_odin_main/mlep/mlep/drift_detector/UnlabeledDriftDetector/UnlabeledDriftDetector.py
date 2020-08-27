__metaclass__ = type

class UnlabeledDriftDetector():
    """ This is an ensemble drift detector. It takes as input the ensemble of classifications, and uses distribution to check for drift"""


    def __init__(self,):
        raise NotImplementedError()

    def reset(self,):
        """ reset detector parameters, e.g. after drift has been detected """
        raise NotImplementedError()

    def detect(self,ensembleClassification):
        """

        ensembleClassification - list of classifications

        returns -- is there drift (T/F)
        """

        raise NotImplementedError()

    