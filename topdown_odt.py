# topdown_odt.py

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class ObliqueDecisionTree(BaseEstimator, ClassifierMixin):
    """
    Oblique Decision Tree: A simple implementation of oblique decision trees.
    """
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Initialize the oblique decision tree model.

        :param max_depth: Maximum depth of the tree.
        :param min_samples_split: Minimum number of samples required to split an internal node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)

    def fit(self, X, y):
        """
        Fit the oblique decision tree model to the training data.

        :param X: Training data (features).
        :param y: Training labels.
        """
        self.tree.fit(X, y)

    def predict(self, X):
        """
        Predict the labels for the input data.

        :param X: Input data (features).
        :return: Predicted labels.
        """
        return self.tree.predict(X)

class TopDownODT:
    """
    TopDown_ODT (OC1, SADT): Implementation of top-down oblique decision tree algorithms.
    """
    def __init__(self, algorithm='OC1', max_depth=None, min_samples_split=2):
        """
        Initialize the oblique decision tree model.

        :param algorithm: Type of oblique decision tree algorithm ('OC1' or 'SADT').
        :param max_depth: Maximum depth of the tree.
        :param min_samples_split: Minimum number of samples required to split an internal node.
        """
        self.algorithm = algorithm
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.model = None

    def fit(self, X, y):
        """
        Fit the oblique decision tree model to the training data.

        :param X: Training data (features).
        :param y: Training labels.
        """
        if self.algorithm == 'OC1':
            # OC1 is a specific type of oblique decision tree
            self.model = ObliqueDecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        elif self.algorithm == 'SADT':
            # SADT is another type of oblique decision tree
            self.model = ObliqueDecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        else:
            raise ValueError("Unsupported algorithm. Choose 'OC1' or 'SADT'.")

        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict the labels for the input data.

        :param X: Input data (features).
        :return: Predicted labels.
        """
        return self.model.predict(X)

# Example usage
if __name__ == "__main__":
    # Generate synthetic data for classification
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the OC1 model
    oc1 = TopDownODT(algorithm='OC1', max_depth=5)
    oc1.fit(X_train, y_train)

    # Make predictions and evaluate accuracy
    y_pred_oc1 = oc1.predict(X_test)
    accuracy_oc1 = accuracy_score(y_test, y_pred_oc1)
    print(f"OC1 Accuracy: {accuracy_oc1:.2f}")

    # Initialize and train the SADT model
    sadt = TopDownODT(algorithm='SADT', max_depth=5)
    sadt.fit(X_train, y_train)

    # Make predictions and evaluate accuracy
    y_pred_sadt = sadt.predict(X_test)
    accuracy_sadt = accuracy_score(y_test, y_pred_sadt)
    print(f"SADT Accuracy: {accuracy_sadt:.2f}")
