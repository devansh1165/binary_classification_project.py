"""
Simple Example: Binary Classification with Real Dataset
=====================================================

This script demonstrates how to use the binary classification project
with a real dataset (Breast Cancer dataset from scikit-learn).
"""

from binary_classification_project import BinaryClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd

def run_breast_cancer_example():
    """Run binary classification on breast cancer dataset."""
    print("Loading Breast Cancer Dataset...")

    # Load the dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Create feature names DataFrame for better understanding
    feature_names = data.feature_names
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {', '.join(feature_names[:5])}...")  # Show first 5 features
    print(f"Classes: {data.target_names}")

    # Initialize classifier
    classifier = BinaryClassifier(random_state=42)

    # Load and preprocess data
    classifier.load_data(X, y)
    classifier.preprocess_data(test_size=0.2)

    # Train models
    classifier.initialize_models()
    classifier.train_and_evaluate()

    # Create visualizations
    classifier.plot_results()

    # Generate report
    classifier.generate_report()

    # Example of hyperparameter tuning
    print("\nTuning Random Forest...")
    best_rf = classifier.hyperparameter_tuning('Random Forest')

    print("\nExample completed successfully!")
    print("Check the generated files:")
    print("- model_results.csv")
    print("- model_comparison.png")
    print("- roc_curves.png") 
    print("- confusion_matrix.png")

if __name__ == "__main__":
    run_breast_cancer_example()
