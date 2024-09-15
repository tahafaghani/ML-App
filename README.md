
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

## Graphical User Interface (GUI) Using PyQt5

The project includes a custom Graphical User Interface (GUI) built with `PyQt5`, making it easy for users to interact with the machine learning models without directly using the command line. The GUI allows users to select different algorithms, upload datasets, and view model results in a user-friendly manner.

### Key Features of the GUI:

- **Model Selection**: Users can choose between supervised learning models (e.g., ANN, SVM, Decision Trees) or unsupervised learning models (e.g., K-Means Clustering) via intuitive buttons.
- **Dataset Upload**: Users can upload their own CSV files, or they can choose to work with the default dataset (`churn.csv`).
- **Results Display**: The output of each model is shown in a separate window, providing accuracy scores, confusion matrices, and other metrics.
- **Background Images and Icons**: The interface includes icons and background images to enhance the visual experience.
  
### Components of the GUI:

1. **Main Window**: The main interface from which users select the model they wish to run. It has buttons for each model, along with an option to upload a dataset.
2. **Custom Buttons**: Each model is represented by a button, which has an icon and text. When clicked, the corresponding model runs and displays the results in a new window.
3. **Output Window**: A separate window that pops up when a model is executed. It displays the model’s output, including accuracy, recall, precision, and a confusion matrix plot.

### File Upload Mechanism:

- The `QFileDialog` class is used to allow users to select and upload a CSV file from their system.
- Users can also opt to continue with the default dataset by checking a box (`QCheckBox`), which disables the file upload option.

### Layouts and Widgets:

- The GUI uses a combination of `QVBoxLayout` and `QGridLayout` to arrange the buttons and other elements. 
- Each button is represented as an instance of the `IconButton` class, which is a custom widget combining an icon and text.
- The results are displayed in a `QTextEdit` widget within a `QDialog` window for easy reading.

### Running Models from the GUI:

- Each model is imported as a module and executed when its corresponding button is clicked.
- The `run_model()` function handles the process of executing the model's `runscript()` function and displaying the results in an output window.

### Example of GUI Workflow:

1. **Launch the GUI** by running `main.py`.
2. **Choose a Model**: Click on one of the model buttons (e.g., ANN, SVM).
3. **View Results**: A new window will pop up showing the performance of the selected model (e.g., accuracy, confusion matrix).

## Creating an Executable File (.exe) from Python Code

To make it easier for others to run your Python project without needing to install Python and its dependencies, you can create an executable file using `PyInstaller`. Follow these steps:


<p align="center">
  <img src="https://github.com/tahafaghani/ML-App/blob/main/Exe-GUI.png
" width="45%" alt="GUI-Exe"/>
</p>


### Step 1: Install PyInstaller

If you haven't already installed `PyInstaller`, you can do so by running:

```bash
pip install pyinstaller
```

### Step 2: Generate the Executable

Once `PyInstaller` is installed, navigate to the directory where your `main.py` file is located. Then, use the following command to create a standalone `.exe` file:

```bash
pyinstaller --onefile --windowed main.py
```

Explanation of the options:
- `--onefile`: Combines everything into a single executable file.
- `--windowed`: Ensures that no terminal window is opened (useful for GUI applications).

### Step 3: Locate the Executable

After running the command, you will find the executable in the `dist/` folder inside your project directory. You can share this `.exe` file with others, and they will be able to run the program without needing to install Python or any additional dependencies.

#### Example

```bash
dist/
├── main.exe
```

## License

This project is open source and free to use under the MIT License.
