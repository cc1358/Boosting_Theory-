# adaboost_trees.py

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class AdaBoostWithTrees:
    """
    AdaBoost + Trees: A simple implementation of AdaBoost using decision trees as weak learners.
    """
    def __init__(self, n_estimators=50):
        """
        Initialize the AdaBoost model.

        :param n_estimators: Number of weak learners (decision trees) to use.
        """
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        """
        Fit the AdaBoost model to the training data.

        :param X: Training data (features).
        :param y: Training labels.
        """
        n_samples, _ = X.shape
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            # Train a weak learner (decision stump)
            tree = DecisionTreeClassifier(max_depth=1)
            tree.fit(X, y, sample_weight=weights)
            predictions = tree.predict(X)

            # Compute the error and alpha
            error = np.sum(weights * (predictions != y)) / np.sum(weights)
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

            # Update the weights
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            # Save the model and alpha
            self.models.append(tree)
            self.alphas.append(alpha)

    def predict(self, X):
        """
        Predict the labels for the input data.

        :param X: Input data (features).
        :return: Predicted labels.
        """
        model_preds = np.array([model.predict(X) for model in self.models])
        return np.sign(np.dot(self.alphas, model_preds))

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1 for AdaBoost
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the AdaBoost model
    ada = AdaBoostWithTrees(n_estimators=50)
    ada.fit(X_train, y_train)

    # Make predictions and evaluate accuracy
    y_pred = ada.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
