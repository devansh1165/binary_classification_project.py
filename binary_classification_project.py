
"""
Complete Binary Classification Project
=====================================

This project demonstrates various binary classification techniques using Python and Scikit-learn.
It includes data preprocessing, multiple algorithms, evaluation metrics, and visualization.

Author: Machine Learning Practitioner
Date: 2025
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_auc_score, roc_curve)
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

class BinaryClassifier:
    """
    A comprehensive binary classification class that implements multiple algorithms
    and provides extensive evaluation capabilities.
    """

    def __init__(self, random_state=42):
        """Initialize the classifier with default parameters."""
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()

    def create_sample_data(self, n_samples=1000, n_features=10):
        """
        Create a sample binary classification dataset.

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        n_features : int
            Number of features to generate

        Returns:
        --------
        X, y : array-like
            Feature matrix and target vector
        """
        print("Creating sample binary classification dataset...")
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.7),
            n_redundant=int(n_features * 0.2),
            n_clusters_per_class=1,
            random_state=self.random_state
        )

        # Convert to DataFrame for better handling
        feature_names = [f'feature_{i+1}' for i in range(n_features)]
        self.data = pd.DataFrame(X, columns=feature_names)
        self.data['target'] = y

        print(f"Dataset created with {n_samples} samples and {n_features} features")
        print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

        return X, y

    def load_data(self, X, y):
        """
        Load custom data for classification.

        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        """
        self.X = X
        self.y = y
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

    def preprocess_data(self, test_size=0.2):
        """
        Preprocess the data by splitting and scaling.

        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing
        """
        print("Preprocessing data...")

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state, stratify=self.y
        )

        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")

    def initialize_models(self):
        """Initialize all classification models."""
        print("Initializing models...")

        self.models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state),
            'Random Forest': RandomForestClassifier(random_state=self.random_state),
            'SVM': SVC(random_state=self.random_state, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'AdaBoost': AdaBoostClassifier(random_state=self.random_state)
        }

        print(f"Initialized {len(self.models)} models")

    def train_and_evaluate(self):
        """Train all models and evaluate their performance."""
        print("\nTraining and evaluating models...")
        print("-" * 50)

        self.results = {}

        for name, model in self.models.items():
            print(f"Training {name}...")

            # Train the model
            if name in ['SVM', 'Logistic Regression', 'K-Nearest Neighbors']:
                # Use scaled data for algorithms sensitive to feature scaling
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                # Use original data for tree-based algorithms
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)

            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

            print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

        # Find the best model based on F1 score
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['f1_score'])
        self.best_model = {
            'name': best_model_name,
            'model': self.results[best_model_name]['model'],
            'metrics': self.results[best_model_name]
        }

        print(f"\nBest model: {best_model_name} (F1 Score: {self.results[best_model_name]['f1_score']:.4f})")

    def plot_results(self):
        """Create visualizations of the results."""
        print("\nCreating visualizations...")

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        model_names = list(self.results.keys())

        for i, metric in enumerate(metrics):
            if i < 4:  # Only plot first 4 metrics in subplots
                row, col = i // 2, i % 2
                values = [self.results[name][metric] for name in model_names]

                axes[row, col].bar(model_names, values)
                axes[row, col].set_title(f'{metric.replace("_", " ").title()} Comparison')
                axes[row, col].set_ylim(0, 1)
                axes[row, col].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # ROC Curves
        plt.figure(figsize=(10, 8))
        for name in model_names:
            fpr, tpr, _ = roc_curve(self.y_test, self.results[name]['probabilities'])
            auc_score = self.results[name]['roc_auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Confusion Matrix for best model
        best_name = self.best_model['name']
        cm = confusion_matrix(self.y_test, self.results[best_name]['predictions'])

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
        plt.title(f'Confusion Matrix - {best_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self):
        """Generate a comprehensive classification report."""
        print("\n" + "="*70)
        print("BINARY CLASSIFICATION PROJECT REPORT")
        print("="*70)

        # Dataset info
        print(f"\nDataset Information:")
        print(f"- Total samples: {len(self.X)}")
        print(f"- Features: {self.X.shape[1]}")
        print(f"- Training samples: {len(self.X_train)}")
        print(f"- Test samples: {len(self.X_test)}")

        # Model performance summary
        print(f"\nModel Performance Summary:")
        print("-" * 50)

        # Create results DataFrame
        results_df = pd.DataFrame({
            name: {
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC AUC': f"{metrics['roc_auc']:.4f}"
            }
            for name, metrics in self.results.items()
        }).T

        print(results_df)

        # Best model details
        print(f"\nBest Model: {self.best_model['name']}")
        print("-" * 30)
        best_metrics = self.best_model['metrics']
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            print(f"{metric.replace('_', ' ').title()}: {best_metrics[metric]:.4f}")

        # Detailed classification report for best model
        print(f"\nDetailed Classification Report - {self.best_model['name']}:")
        print("-" * 50)
        best_predictions = self.results[self.best_model['name']]['predictions']
        report = classification_report(self.y_test, best_predictions)
        print(report)

        # Save results to CSV
        results_df.to_csv('model_results.csv')
        print("\nResults saved to 'model_results.csv'")

    def hyperparameter_tuning(self, model_name='Random Forest'):
        """
        Perform hyperparameter tuning for a specific model.

        Parameters:
        -----------
        model_name : str
            Name of the model to tune
        """
        print(f"\nPerforming hyperparameter tuning for {model_name}...")

        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestClassifier(random_state=self.random_state)
            X_train_use = self.X_train

        elif model_name == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1],
                'kernel': ['rbf', 'linear']
            }
            model = SVC(random_state=self.random_state, probability=True)
            X_train_use = self.X_train_scaled

        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return

        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1', n_jobs=-1
        )
        grid_search.fit(X_train_use, self.y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def predict_new_data(self, X_new):
        """
        Make predictions on new data using the best model.

        Parameters:
        -----------
        X_new : array-like
            New data to predict

        Returns:
        --------
        predictions : array
            Predicted classes
        probabilities : array
            Prediction probabilities
        """
        if self.best_model is None:
            print("No trained model available. Please train models first.")
            return None, None

        # Scale the new data if needed
        model_name = self.best_model['name']
        if model_name in ['SVM', 'Logistic Regression', 'K-Nearest Neighbors']:
            X_new_processed = self.scaler.transform(X_new)
        else:
            X_new_processed = X_new

        # Make predictions
        model = self.best_model['model']
        predictions = model.predict(X_new_processed)
        probabilities = model.predict_proba(X_new_processed)[:, 1]

        return predictions, probabilities

def main():
    """Main function to demonstrate the binary classification project."""
    print("Binary Classification Project")
    print("="*50)

    # Initialize classifier
    classifier = BinaryClassifier(random_state=42)

    # Create sample data
    X, y = classifier.create_sample_data(n_samples=1000, n_features=10)
    classifier.load_data(X, y)

    # Preprocess data
    classifier.preprocess_data(test_size=0.2)

    # Initialize and train models
    classifier.initialize_models()
    classifier.train_and_evaluate()

    # Create visualizations
    classifier.plot_results()

    # Generate comprehensive report
    classifier.generate_report()

    # Example of hyperparameter tuning
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING EXAMPLE")
    print("="*50)
    best_rf = classifier.hyperparameter_tuning('Random Forest')

    # Example of making predictions on new data
    print("\n" + "="*50)
    print("PREDICTION EXAMPLE")
    print("="*50)

    # Generate some sample new data
    X_new, _ = make_classification(n_samples=5, n_features=10, random_state=99)
    predictions, probabilities = classifier.predict_new_data(X_new)

    print("Sample predictions on new data:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"Sample {i+1}: Class {pred} (Probability: {prob:.3f})")

if __name__ == "__main__":
    main()
