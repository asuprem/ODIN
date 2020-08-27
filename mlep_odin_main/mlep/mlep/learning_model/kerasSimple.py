import mlep.learning_model.BaseLearningModel
import keras
from keras.models import Sequential
from keras.layers import  Dense, Activation
from keras.utils.np_utils import to_categorical
import numpy as np

class kerasSimple(mlep.learning_model.BaseLearningModel.BaseLearningModel):
    """ Wrapper for simple keras classifier (for testing) """

    def __init__(self,):
        """
        Initialize base classifier
        """

        self.in_shape = 300
        model = self._buildModel(self.in_shape)
        super(kerasSimple,self).__init__(model)

    def _fit(self,X,y,**kwargs):
        """ keras fit with sample_weight """
        # TODO FIX THIS to be generic? handle error here...if sample_weight is a list... ro add in documentation that sample_weight MUST be a list...
        if 'sample_weight' in kwargs and kwargs['sample_weight'] is not None:
            kwargs['sample_weight'] = np.asarray(kwargs['sample_weight'])
        self._model.fit(X,to_categorical(y), **kwargs)
        

    def _update(self,X,y,**kwargs):
        """ keras update/partial fit with sample_weight """
        if 'sample_weight' in kwargs and kwargs['sample_weight'] is not None:
            kwargs['sample_weight'] = np.asarray(kwargs['sample_weight'])
        self._model.fit(X,to_categorical(y), **kwargs)

    """
    def _update_and_test(self, X_train, y_train, split = 0.7, X_test = None, y_test = None, sample_weight=None):
        update using update ... 
        pass
    """

    """
    def _fit_and_test(self, X_train, y_train, split = 0.7, X_test = None, y_test = None, sample_weight=None):
        fit using fit ... 
        pass
    """


    
    def _predict(self,X_sample):
        """get prediction """
        
        predictions= self._model.predict(X_sample)
        return np.sum(np.floor(predictions+np.array([0,0.5])),1)

    def _clonedModel(self):
        """Return a clone of the current model. This and clone work together
        
        """
        nModel = self._buildModel(self.in_shape)
        nModel.set_weights(self._model.get_weights())
        return nModel


    def _buildModel(self,in_shape):
        model = Sequential()
        model.add(Dense(512,input_shape=(in_shape,)))
        model.add(Activation('relu'))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model