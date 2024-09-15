
# AI Final Project

## Author: Taha Faghani 


This repository contains the final project for the AI course, focusing on applying various machine learning models to classify and analyze a given dataset (e.g., churn prediction). The project includes multiple algorithms implemented in Python using popular libraries such as `scikit-learn`, `matplotlib`, and `pandas`.

## Requirements

Ensure you have the following dependencies installed before running the project:

```bash
pip install numpy pandas matplotlib scikit-learn seaborn PyQt5
```

## Project Structure

- `main.py`: The main script that runs the entire application using a GUI created with `PyQt5`. This file links the models and manages the application's flow.
- `ann.py`: Script implementing Artificial Neural Networks (ANN) using the `MLPClassifier` from `scikit-learn`.
- `svm.py`: Script implementing Support Vector Machine (SVM) using the `SVC` class from `scikit-learn`.
- `dt.py`: Script implementing Decision Trees using `DecisionTreeClassifier` from `scikit-learn`.
- `knn.py`: Script implementing the K-Nearest Neighbors (KNN) algorithm using `KNeighborsClassifier` from `scikit-learn`.
- `kmeans.py`: Script for K-Means Clustering.
- `logistic_regression.py`: Script implementing Logistic Regression using `LogisticRegression` from `scikit-learn`.
- `naive_base.py`: Script implementing Naive Bayes using `GaussianNB` from `scikit-learn`.
- `best.py`: A script that compares various models and determines the best performing one.
- `icons/`: Folder containing the necessary icons for the GUI.

## Running the Project

1. Ensure all dependencies are installed.
2. Run the `main.py` file:

   ```bash
   python main.py
   ```

3. You will be greeted with a GUI where you can select a machine learning algorithm to train and test on the dataset.

## Dataset

The project uses a churn dataset. By default, the dataset is expected to be named `churn.csv` and should be placed in the same directory as the scripts. Alternatively, you can upload a custom dataset through the GUI.

### Dataset Preprocessing

The dataset undergoes basic preprocessing steps, including:
- Replacing missing values.
- Converting categorical variables into numerical values.
- Standardizing the feature set using `StandardScaler`.

## Models Implemented

### Supervised Learning Models

1. **Artificial Neural Networks (ANN)** (`ann.py`)
2. **Support Vector Machine (SVM)** (`svm.py`)
3. **Decision Tree** (`dt.py`)
4. **Logistic Regression** (`logistic_regression.py`)
5. **Naive Bayes** (`naive_base.py`)
6. **K-Nearest Neighbors (KNN)** (`knn.py`)

### Unsupervised Learning Models

1. **K-Means Clustering** (`kmeans.py`)

### Model Comparison

The `best.py` script compares the models based on accuracy and outputs the best performing model along with a plot of the results.

## Graphical User Interface (GUI)

The project includes a custom-built GUI using `PyQt5`. The interface allows users to:
- Select a supervised or unsupervised learning algorithm.
- Upload a dataset or use the default one.
- View model performance and output through a dialog window.

### Important Note

This project is designed for educational purposes and should be run in a controlled environment. Ensure that the dataset is properly formatted and follows the expected structure.

## License

This project is open source and free to use under the MIT License.
