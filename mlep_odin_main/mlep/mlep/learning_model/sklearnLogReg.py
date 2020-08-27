
import mlep.learning_model.BaseLearningModel


class sklearnLogReg(mlep.learning_model.BaseLearningModel.BaseLearningModel):
    """Logistic regression learning model wrapper."""

    def __init__(self):
        """Initialize a logistic regression learning model wrapper."""
        from sklearn.linear_model import LogisticRegression
        super(sklearnLogReg,self).__init__(LogisticRegression(solver="lbfgs", max_iter=100000, warm_start=True))

    def _update(self, X, y):
        """Update the Logistic Regression model to the training data.
        X -- [array of shape (n_samples, n_features)] Training data.
        y -- [array of shape (n_samples)] Target values for the training data.
        """
        self.fit(X, y)

    