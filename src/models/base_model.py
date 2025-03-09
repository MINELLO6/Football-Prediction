# File: src/models/base_model.py
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    BaseModel is an abstract class that defines a common interface for all models.

    All derived models must implement the following methods:
      - fit(X, y): Train the model on training data.
      - predict(X): Predict the target for given input data.
      - evaluate(X, y): Evaluate the model performance on a given dataset.
      - summary(): Optionally, print or return a summary of the model.
    """

    @abstractmethod
    def fit(self, X, y=None):
        """
        Fit the model to the training data.

        Parameters:
            X: Features (e.g., pandas DataFrame or numpy array)
            y: Target values (e.g., pandas Series or numpy array)
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict target values for given input data.

        Parameters:
            X: Features to predict (e.g., pandas DataFrame or numpy array)

        Returns:
            Predictions corresponding to X.
        """
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """
        Evaluate the model performance on a given dataset.

        Parameters:
            X: Features of the dataset.
            y: True target values.

        Returns:
            A performance metric (e.g., accuracy, RMSE, etc.).
        """
        pass

    # def summary(self):
    #     """
    #     Print a summary of the model.
    #     This method can be optionally overridden by derived classes.
    #     """
    #     print("Model summary not implemented.")


# Example usage for testing
if __name__ == "__main__":
    print("This is the BaseModel. Derived classes should implement fit, predict, and evaluate methods.")
