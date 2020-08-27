__metaclass__ = type

class LabeledDriftDetector:
    """ abstract Labeled Drift Detector class
    
    A Labeled Drift Detector takes in labeled data or error rates and detets if drift has occured. Any LabeledDriftDetector must implement the functions provided here.

    A Labeled Drift Detector is also specified in the Drift Detector configuration (todo) if it is to be used with MLEPServer
    """


    def __init__(self,):
        """ 
        Initialize the Drift Detector Object.
        
        This function will initialize internal parameters for the Drift Detector Object.

        Args:
            None: None

        Returns:
            Nothing

        Raises:
            NotImplementedError

        """
        raise NotImplementedError()

    def reset(self,):
        """ 
        Reset Drift Detector parameters.
        
        This needs to be called by the user. Normally a detector is reset as soon as Drift has been detected.

        Args:
            None: None

        Returns:
            Nothing

        Raises:
            NotImplementedError
            
        """

        raise NotImplementedError()

    def detect(self,classification):
        """ 
        Detect drift using classification
        
        Detect drift using the argument. The following arguments are supported:
            - classification: The raw classification or class. In the case of binary, this is either 0 or 1.
            - error: Whether the classification was correct or incorrect. 0 means classification was correct. 
        Different detectors support different arguments.


        Args:
            classification: INT. The raw class.

        Returns:
            driftDetected: Bool
                A boolean indicating if drift has been detector or not

        Raises:
            NotImplementedError
            
        """

        raise NotImplementedError()

    