# Iris Flower Classification using Random Forest

## Project Overview
This project implements a machine learning model to classify Iris flowers into their respective species based on their measurements. It uses the famous Iris dataset and Random Forest Classifier to create a predictive model.

## Features
- Loads and processes the Iris dataset
- Implements Random Forest Classification
- Saves and loads the trained model
- Provides prediction capabilities for new data
- Includes model accuracy evaluation

## Technical Details
The project uses the following Python libraries:
- scikit-learn (for machine learning algorithms and dataset)
- numpy (for numerical operations)
- pickle (for model serialization)

## Project Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setting up a Virtual Environment
1. Open terminal/command prompt
2. Navigate to your project directory: `cd path/to/project`
3. Create a virtual environment: On Windows `python -m venv venv`
4. Activate the virtual environment: On Windows `venv\Scripts\activate` On macOS/Linux `source venv/bin/activate`
5. Install required packages: `pip install -r requirements.txt`

## Random Forest Classifier Explained

Random Forest is an ensemble learning method that:
1. Creates multiple decision trees
2. Uses bootstrap aggregating (bagging)
3. Makes predictions by averaging/voting from all trees

### How it Works:
1. **Bootstrap Sampling**: Creates random samples from the training data
2. **Decision Tree Creation**: Builds a decision tree for each sample
3. **Feature Selection**: Randomly selects features at each split
4. **Voting**: Combines predictions from all trees
   - For classification: Uses majority voting
   - For regression: Uses average of predictions

### Advantages:
- Reduces overfitting
- Handles non-linear relationships
- Provides feature importance
- Works well with both numerical and categorical data

## Dataset Information

The Iris dataset contains 150 samples with:

### Features:
1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm)

### Target Classes:
- Setosa
- Versicolor
- Virginica

## Usage

1. Ensure your virtual environment is activated
2. Run the script: `python main.py`
