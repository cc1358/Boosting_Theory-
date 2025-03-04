# topdown_dt.py

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

class TopDownDT:
    """
    TopDown_DT (CART, C4.5): Implementation of top-down decision tree algorithms.
    """
    def __init__(self, algorithm='CART', max_depth=None, min_samples_split=2):
        """
        Initialize the decision tree model.

        :param algorithm: Type of decision tree algorithm ('CART' or 'C4.5').
        :param max_depth: Maximum depth of the tree.
        :param min_samples_split: Minimum number of samples required to split an internal node.
        """
        self.algorithm = algorithm
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        """
        Fit the decision tree model to the training data.

        :param X: Training data (features).
        :param y: Training labels.
        """
        if self.algorithm == 'CART':
            self.tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        elif self.algorithm == 'C4.5':
            # C4.5 is similar to CART but uses information gain ratio
            self.tree = DecisionTreeClassifier(criterion='entropy', max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        else:
            raise ValueError("Unsupported algorithm. Choose 'CART' or 'C4.5'.")

        self.tree.fit(X, y)

    def predict(self, X):
        """
        Predict the labels for the input data.

        :param X: Input data (features).
        :return: Predicted labels.
        """
        return self.tree.predict(X)

# Example usage
if __name__ == "__main__":
    # Generate synthetic data for classification
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the CART model
    cart = TopDownDT(algorithm='CART', max_depth=5)
    cart.fit(X_train, y_train)

    # Make predictions and evaluate accuracy
    y_pred_cart = cart.predict(X_test)
    accuracy_cart = accuracy_score(y_test, y_pred_cart)
    print(f"CART Accuracy: {accuracy_cart:.2f}")

    # Initialize and train the C4.5 model
    c45 = TopDownDT(algorithm='C4.5', max_depth=5)
    c45.fit(X_train, y_train)

    # Make predictions and evaluate accuracy
    y_pred_c45 = c45.predict(X_test)
    accuracy_c45 = accuracy_score(y_test, y_pred_c45)
    print(f"C4.5 Accuracy: {accuracy_c45:.2f}")
