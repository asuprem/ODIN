__metaclass__ = type



class BaseLearningModel:
    """
    Abstract learning model.
    
    Attributes:
        _model -- Private. The actual model. For now, just sklearn. Plans to incorporate keras, tensorflow, rnns, others.
        mode -- "binary", "multiclass", "regression". Only "binary" is supported.
        classes -- The classes in data. Nominally in [0,NUM_CLASSES-1]
        track_drift -- BOOL. Whether this model tracks its own drift
        _confidence -- FLOAT. Model's confidence on the previous sample.
        _trust -- Model's trust on previous sample.
        _drifting -- BOOL. Whether model is currently drifting.

        _driftTracker -- Internal Drift Tracker. May use one of mlep's drift tracker methods.
        dataCharacteristics -- Characteristics of data the model was trained with

    """

    def __init__(self, model, mode="binary",classes=[0,1]):
        """Initialize a learning model.
        model -- [object] Learning model.
        mode  -- [str] Mode of learning (binary, multiclass, or regression)
        model -- [int] Number of classes. None for regression
        """
        self._model = model
        self.mode = mode
        self.classes = classes
        self.track_drift = False
        self._confidence = 1.0
        self._trust = 1.0
        self._drifting = False
        self._driftTracker = None
        self.dataCharacteristics = None

    def fit(self, X, y, **kwargs):
        """Fit the statistical learning model to the training data.
        X -- [array of shape (n_samples, n_features)] Training data.
        y -- [array of shape (n_samples)] Target values for the training data.
        """
        self._fit(X,y, **kwargs)
        
    def _fit(self,X,y, **kwargs):
        """ Internal function to fit statistical model

        This is the one that should be modified for derived classes

        """
        self._model.fit(X, y,**kwargs)

    def update(self, X, y,**kwargs):
        """Update the statistical learning model to the training data.
        X -- [array of shape (n_samples, n_features)] Training data.
        y -- [array of shape (n_samples)] Target values for the training data.
        """
        self._update(X,y,**kwargs)

    def _update(self,X,y,**kwargs):
        """ Internal function to update the statistical model

        This is the one that should be modified for derived classes

        """
        self._model.partial_fit(X, y, classes=self.classes, **kwargs)

    def update_and_test(self, X_train, y_train, split = 0.7, X_test = None, y_test = None,sample_weight=None):
        """Update the statistical learning model to the training data and test
        X -- [array of shape (n_samples, n_features)] Training data.
        y -- [array of shape (n_samples)] Target values for the training data.
        X_test -- [array of shape (n_samples, n_features)] Testing data.
        y_test -- [array of shape (n_samples)] Target values for the Testing data.

        If X_test and y_test are not provided, split value is used (default 0.7) to shuffle and split X_train and y_train
        """
        # TODO handle weird erros, such as X_test specified, but y_test not specified, etc
        precision, recall, score = self._update_and_test(X_train, y_train, split = split, X_test = X_test, y_test = y_test, sample_weight=sample_weight)

        return precision, recall, score

    def _update_and_test(self, X_train, y_train, split = 0.7, X_test = None, y_test = None, sample_weight=None):
        """ Internal update_and_test method

        This is the one that should be modified for derived classes

        """
        if X_test is None and y_test is None:
            from sklearn.model_selection import train_test_split
            if sample_weight is not None:
                X_train, X_test, y_train, y_test, sample_weight, _ = train_test_split(X_train, y_train, sample_weight, test_size=1.0-split, random_state = 42, shuffle=True, stratify=y_train)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=1.0-split, random_state = 42, shuffle=True, stratify=y_train)
    
        self.update(X_train, y_train, sample_weight=sample_weight)
        precision, recall, score = self._precision_recall_fscore(X_test, y_test)
        return precision, recall, score

    def fit_and_test(self, X_train, y_train, split = 0.7, X_test = None, y_test = None, sample_weight=None):
        """Fit the statistical learning model to the training data and test
        X -- [array of shape (n_samples, n_features)] Training data.
        y -- [array of shape (n_samples)] Target values for the training data.
        X_test -- [array of shape (n_samples, n_features)] Testing data.
        y_test -- [array of shape (n_samples)] Target values for the Testing data.

        If X_test and y_test are not provided, split value is used (default 0.7) to shuffle and split X_train and y_train
        """
        # TODO handle weird erros, such as X_test specified, but y_test not specified, etc
        precision, recall, score = self._fit_and_test(X_train, y_train, split = split, X_test = X_test, y_test = y_test, sample_weight=sample_weight)
        return precision, recall, score

    def _fit_and_test(self, X_train, y_train, split = 0.7, X_test = None, y_test = None, sample_weight=None):
        """ Internal function to fit the statistical model.

        This is the one that should be modified for derived classes

        """
        if X_test is None and y_test is None:
            from sklearn.model_selection import train_test_split
            if sample_weight is not None:
                X_train, X_test, y_train, y_test, sample_weight, _ = train_test_split(X_train, y_train, sample_weight, test_size=1.0-split, random_state = 42, shuffle=True, stratify=y_train)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=1.0-split, random_state = 42, shuffle=True, stratify=y_train)
    
        self.fit(X_train, y_train, sample_weight=sample_weight)
        precision, recall, score = self._precision_recall_fscore(X_test, y_test)
        return precision, recall, score

    def _precision_recall_fscore(self, X, y):
        """Return a 3-tuple where the first element is the precision of the model, the second is the
        recall, and the third is the F-measure.
        For statistical learning models, the test data is represented by an array of dimension
        (n_samples, n_features);
        
        X -- [array] Test data.
        y -- [array] Target values for the test data.
        """
        from sklearn.metrics import precision_recall_fscore_support
        y_pred = self.predict(X, mode="test")
        prs_ = tuple(precision_recall_fscore_support(y, y_pred, average="weighted")[0:3])
        return prs_[0], prs_[1], prs_[2]

    def predict(self, X_sample, mode = "predict", y_label = None):
        """Return predicted labels for the test data.

        Given a data point, predict will perform one of three things based on the value of mode:
            - predict/test -- Return the model's prediction of the label of a given sample X_sample
            - implicit --   Perform prediction and return the mode's prediction of label. In addition, store a history of the model's confidence and set internal drift detector
                            Then, calling model's driftDetected() method next time should return True if drift was detected
            
            - explicit --   Perform prediction. Detect drift based on predicted value, confidence, and actual value. Update internal drift detector

        
        Args:
            X_sample: [array of shape (n_samples, n_features)] Test data.
            mode:   "predict" -- for standard prediction. This includes no drift tracking
                    "implicit" -- for implicit drift tracking. This is for unlabeled examples
                    "explicit" -- for explicit drift tracking. This is for labeled examples
                    "test" -- internal-use. No drift tracking is performed. Used during testing/evaluating model post training or update.
        
        Raises:
            ValueError
        """
        
        # Reshaping
        if len(X_sample.shape) == 1:
            X_sample = X_sample.reshape(1,-1)

        if mode == "predict" or mode == "test":
            prediction = self._predict(X_sample = X_sample)
        elif mode == "implicit_confidence":
            if not self.track_drift or self.track_drift is None:
                raise ValueError("Cannot detect implicit drift if drift tracking is not enabled. Enable drift tracking with trackDrift().")
            prediction = self._evaluate_implicit_confidence(X_sample)
        elif mode == "explicit_confidence":
            if not self.track_drift or self.track_drift is None:
                raise ValueError("Cannot detect implicit drift if drift tracking is not enabled. Enable drift tracking with trackDrift().")
            prediction = self._evaluate_explicit_confidence(X_sample, y_label)
        else:
            raise NotImplementedError()
        return prediction


    def _predict(self,X_sample):
        """ Internal function to perform prediction.

        This is the function that should be modified for derived classes
        
        """
        return self._model.predict(X_sample)

    def _evaluate_implicit_confidence(self,X_sample):
        """ This evaluates a model's implicit drift """
        
        raise NotImplementedError()

    def _evaluate_explicit_confidence(self,X_sample, y_label):
        """ This evaluates a model's explicit drift using y_label """
        
        raise NotImplementedError()


    def clone(self, LearningModelToClone):
        """Clone the LearningModelToClone into this model. They must be the same model Type
        
        LearningModelToClone -- [LearningModel] The model to clone
        """
        self._model = LearningModelToClone._clonedModel()

    def _clonedModel(self):
        """Return a clone of the current model. This and clone work together
        
        """
        from sklearn.base import clone as sklearnClone
        return sklearnClone(self._model)

    def isUpdatable(self):
        return True

    def trackDrift(self,_track=None):
        """ 
        Set/get whether drift is being tracked in a model

        Args:
            _track: Boolean. True/False for set. None/empty for get

        Returns:
            Bool -- content of self.track_drift
        
        """

        if _track is not None:
            self.track_drift = _track
            if self.track_drift:
                self._setupDriftTracker()
        return self.track_drift

    def isDrifting(self):
        """ 
        Get whether drift is occuring

        Returns:
            Bool -- Whether drift is occuring
        
        """
        return self._drifting

    def _setupDriftTracker(self,):
        """ 
        Sets up a drift tracker (e.g. DDM, EDDM, etc)

        Raisees:
            NotImplementedError
        
        """
        raise NotImplementedError()

    def addDataCharacteristics(self,dataCharacteristics):
        self.dataCharacteristics = dataCharacteristics
    def getDataCharacteristic(self,_key):
        return self.dataCharacteristics.get(_key)
