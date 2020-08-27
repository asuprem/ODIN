import mlep.learning_model.BaseLearningModel


class sklearnSGD(mlep.learning_model.BaseLearningModel.BaseLearningModel):
    """SVM learning model wrapper."""

    def __init__(self):
        """
        Initialize a SVM learning model wrapper.
        
        """
        from sklearn.linear_model import SGDClassifier
        super(sklearnSGD,self).__init__(SGDClassifier())

        self._internal_error = 0.0

    def _evaluate_implicit_confidence(self,X_sample):
        
        self._confidence = abs(self._model.decision_function(X_sample))
        self._confidence = 1.0 if self._confidence > 1.0 else self._confidence

        # Make it int for DDM
        self._internal_error - 1 - int(self._confidence)
        print(self._internal_error)

        self._drifting = self._driftTracker.detect(self._internal_error)
        return self._predict(X_sample = X_sample)

    def _setupDriftTracker(self,):
        """ 
        Set up Drift Tracker.

        For now, we are using only DDM. Will add support for generic drift trackers later.

        """
        import mlep.drift_detector.LabeledDriftDetector.DDM as DDM
        self._driftTracker = DDM.DDM()
