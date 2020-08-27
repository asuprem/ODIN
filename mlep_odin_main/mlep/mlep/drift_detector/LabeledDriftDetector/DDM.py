# -*- coding: utf-8 -*-
#!/usr/bin/python
import mlep.drift_detector.LabeledDriftDetector.LabeledDriftDetector as LabeledDriftDetector

class DDM(LabeledDriftDetector.LabeledDriftDetector):
    """ Implements the DDM drift detection method.

    This drift detector is based on the paper on the DDM Paper (João Gama, Pedro Medas, Gladys Castillo, Pedro Pereira Rodrigues: Learning with Drift Detection. SBIA 2004: 286-295). We keep the highest alarm level of drift detection (out of control), leaving out warning level. 

    Attributes:
        min_instances: Minimum number of instances required for delivering a detection prediction
        drift_level: Alarm level for drift detection. DDM Paper specified 3.0 for out-of-control and 2.0 for warning

    Methods:
        __init__: Initialize the detector
        reset: Reset the Drift detector
        detect: Perform drift detection

    """
    
    def __init__(self,min_instances=30, drift_level=3.0):
        """ 
        Initialize the DDM Drift Detector
        
        Initialize the DDM Drift detector. Default parameters are provided as well.


        Args:
            min_instances: INT. Minimum number of instances for Detector to return a result.        
            drift_level: Alarm level for drift detector. 3.0 is from the DDM paper. 2.0 would be for drift warnings.

        """

        from math import sqrt
        self.min_instances = min_instances
        self.drift_level = float(drift_level)
        self.i = None
        self.pi = None
        self.si = None
        self.pi_min = None
        self.si_min = None
        self.sqrt=sqrt
        self.reset()


    
    def reset(self,):
        """ 
        Reset the drift detector.

        Resets the DDM Drift detector. This should be called after `detect()` returns True to reset internal parameters.

        """
        
        self.i = 0
        self.pi = 1.0
        self.si = 0.0
        self.pi_min = float("inf")
        self.si_min = float("inf")

    def detect(self,error):
        """ 
        Perform Drift Detection
        
        `detect` performs Drift Detection according to the method specified in the DDM Paper. It requires keeping track of error rate and standard deviation. When the sum of error rate and standard deviation is greater than sum of minimum error rate and minimum standard deviation (scaled by drift level), Drift is detected.


        Args:
            error: INT. 1 if classification was incorrect; 0 if classification was correct

        Returns:
            Bool: True is there is drift; False if there is no drift or number of samples seen is lower than min_instance.

        """

        self.i += 1
        self.pi = self.pi+ (error-self.pi)/float(self.i)
        self.si = self.sqrt(self.pi*  (1-self.pi)/self.i)

        if self.i < self.min_instances:
            return False

        if self.pi + self.si <= self.pi_min + self.si_min:
            self.pi_min = self.pi
            self.si_min = self.si

        if self.pi + self.si > self.pi_min  + self.drift_level * self.si_min:
            return True
        else:
            return False

        
        

    