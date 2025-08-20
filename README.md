# Binary Classification Project

A comprehensive Python project demonstrating various binary classification techniques using scikit-learn and other popular machine learning libraries.

## 🎯 Project Overview

This project implements a complete binary classification pipeline that includes:
- Data preprocessing and exploration
- Multiple classification algorithms
- Comprehensive evaluation metrics
- Visualization of results
- Hyperparameter tuning
- Model comparison and selection

## 🚀 Features

### Algorithms Implemented
- **Logistic Regression**: Linear model for binary classification
- **Decision Tree**: Tree-based classifier with interpretable rules
- **Random Forest**: Ensemble method combining multiple decision trees
- **Support Vector Machine (SVM)**: Kernel-based classifier for complex boundaries
- **K-Nearest Neighbors (KNN)**: Instance-based learning algorithm
- **Naive Bayes**: Probabilistic classifier based on Bayes' theorem
- **AdaBoost**: Adaptive boosting ensemble method

### Evaluation Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: Quality of positive predictions
- **Recall (Sensitivity)**: Ability to find all positive instances
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed breakdown of prediction results

### Visualizations
- Model performance comparison charts
- ROC curves for all models
- Confusion matrix for the best model
- Feature importance plots (where applicable)

## 📋 Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## 🔧 Installation

1. Clone or download this project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python binary_classification_project.py
   ```

## 📖 Usage

### Basic Usage

```python
from binary_classification_project import BinaryClassifier

# Initialize classifier
classifier = BinaryClassifier(random_state=42)

# Create sample data or load your own
X, y = classifier.create_sample_data(n_samples=1000, n_features=10)
classifier.load_data(X, y)

# Preprocess and train
classifier.preprocess_data()
classifier.initialize_models()
classifier.train_and_evaluate()

# Generate visualizations and report
classifier.plot_results()
classifier.generate_report()
```

### Using Custom Data

```python
# Load your own dataset
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Example with breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

classifier = BinaryClassifier()
classifier.load_data(X, y)
classifier.preprocess_data()
classifier.initialize_models()
classifier.train_and_evaluate()
```

### Hyperparameter Tuning

```python
# Tune hyperparameters for specific models
best_rf = classifier.hyperparameter_tuning('Random Forest')
best_svm = classifier.hyperparameter_tuning('SVM')
```

### Making Predictions

```python
# Predict on new data
predictions, probabilities = classifier.predict_new_data(X_new)
```

## 📊 Output Files

The project generates several output files:
- `model_results.csv`: Detailed performance metrics for all models
- `model_comparison.png`: Visual comparison of model performance
- `roc_curves.png`: ROC curves for all models
- `confusion_matrix.png`: Confusion matrix for the best model

## 🔍 Project Structure

```
binary-classification-project/
├── binary_classification_project.py    # Main project file
├── requirements.txt                     # Dependencies
├── README.md                           # Project documentation
├── data/                               # Data directory (optional)
├── models/                             # Saved models (optional)
└── results/                            # Output files
```

## 🧠 Algorithm Details

### Logistic Regression
- Best for: Linear relationships, interpretable results
- Assumptions: Linear relationship between features and log-odds
- Advantages: Fast, interpretable, no tuning required

### Decision Tree
- Best for: Non-linear relationships, categorical features
- Advantages: Highly interpretable, handles mixed data types
- Disadvantages: Prone to overfitting

### Random Forest
- Best for: Most scenarios, good baseline model
- Advantages: Reduces overfitting, handles missing values
- Disadvantages: Less interpretable than single trees

### Support Vector Machine
- Best for: High-dimensional data, complex boundaries
- Advantages: Effective in high dimensions, memory efficient
- Disadvantages: Slow on large datasets, requires feature scaling

### K-Nearest Neighbors
- Best for: Small datasets, local patterns
- Advantages: Simple, no assumptions about data
- Disadvantages: Computationally expensive, sensitive to irrelevant features

### Naive Bayes
- Best for: Text classification, small datasets
- Advantages: Fast, works well with small data
- Assumptions: Feature independence (often violated)

### AdaBoost
- Best for: Weak learners, reducing bias
- Advantages: Reduces bias and variance
- Disadvantages: Sensitive to noise and outliers

## 📈 Performance Metrics Explained

### Accuracy
- Formula: (TP + TN) / (TP + TN + FP + FN)
- Use when: Classes are balanced
- Range: 0-1 (higher is better)

### Precision
- Formula: TP / (TP + FP)
- Use when: False positives are costly
- Range: 0-1 (higher is better)

### Recall (Sensitivity)
- Formula: TP / (TP + FN)
- Use when: False negatives are costly
- Range: 0-1 (higher is better)

### F1-Score
- Formula: 2 × (Precision × Recall) / (Precision + Recall)
- Use when: Need balance between precision and recall
- Range: 0-1 (higher is better)

### ROC AUC
- Measures: Ability to distinguish between classes
- Use when: Comparing models across all thresholds
- Range: 0-1 (0.5 is random, 1.0 is perfect)

## 🎨 Customization

### Adding New Models
```python
# Add to the initialize_models method
self.models['Your Model'] = YourClassifier(parameters)
```

### Custom Evaluation Metrics
```python
# Add to the train_and_evaluate method
custom_metric = your_metric_function(y_true, y_pred)
self.results[name]['custom_metric'] = custom_metric
```

### Custom Visualizations
```python
# Extend the plot_results method
def plot_custom_viz(self):
    # Your custom visualization code
    pass
```

## 🤝 Contributing

Feel free to contribute to this project by:
1. Adding new algorithms
2. Implementing additional evaluation metrics
3. Creating new visualization functions
4. Improving documentation
5. Fixing bugs or optimizing performance

## 📝 License

This project is open source and available under the MIT License.

## 🔗 References

1. Scikit-learn Documentation: https://scikit-learn.org/
2. "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
3. "Pattern Recognition and Machine Learning" by Christopher Bishop
4. "Hands-On Machine Learning" by Aurélien Géron

## 📞 Support

If you encounter any issues or have questions, please:
1. Check the documentation
2. Review the example usage
3. Ensure all dependencies are installed
4. Verify your data format matches the expected input

---

**Happy Classification!** 🎉
