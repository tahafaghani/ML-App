def runscript():
    # Import necessary libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score

    # Load the dataset
    df = pd.read_csv("./churn.csv")

    # Replace specific string values with 'No' for consistency
    df['MultipleLines'] = df['MultipleLines'].str.replace('No phone service', 'No')
    df['OnlineSecurity'] = df['OnlineSecurity'].str.replace('No internet service', 'No')
    df['OnlineBackup'] = df['OnlineBackup'].str.replace('No internet service', 'No')
    df['DeviceProtection'] = df['DeviceProtection'].str.replace('No internet service', 'No')
    df['TechSupport'] = df['TechSupport'].str.replace('No internet service', 'No')
    df['StreamingTV'] = df['StreamingTV'].str.replace('No internet service', 'No')
    df['StreamingMovies'] = df['StreamingMovies'].str.replace('No internet service', 'No')

    # Drop the 'customerID' column as it's not needed for the model
    df = df.drop('customerID', axis=1)

    # Limit the dataset to the first 4000 rows and drop any rows with missing values
    df = df.head(4000)
    df = df.dropna()

    # Function to convert strings to integers
    def str2int(data):
        for col in data.columns:
            for i, item in enumerate(data[col]):
                if isinstance(item, str):
                    try:
                        data.at[i, col] = int(item)
                        data.at[i, col] = float(item)
                    except ValueError:
                        if data[col].dtype == 'object':
                            labels, _ = pd.factorize(data[col])
                            data[col] = labels
        return data

    # Convert string columns to numeric values
    str2int(df)

    # Separate the features (X) and the target variable (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    # Initialize and train an Artificial Neural Network (ANN) model
    best_hidden_layer_sizes = (2, 8, 10)
    best_activation = 'logistic'
    best_solver = 'lbfgs'
    best_learning_rate = 'constant'
    model_mlp = MLPClassifier(
        activation=best_activation, 
        solver=best_solver, 
        learning_rate=best_learning_rate, 
        max_iter=2000, 
        random_state=42, 
        hidden_layer_sizes=best_hidden_layer_sizes
    )
    model_mlp.fit(X_train, y_train)
    y_pred_ann = model_mlp.predict(X_test)
    acc_ann = accuracy_score(y_test, y_pred_ann)

    # Initialize and train a Decision Tree model
    model_dt = DecisionTreeClassifier(max_depth=5, min_samples_split=4, min_samples_leaf=6)
    model_dt.fit(X_train, y_train)
    y_pred_dt = model_dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)

    # Initialize and train a K-Nearest Neighbors (KNN) model
    model_knn = KNeighborsClassifier(n_neighbors=8)
    model_knn.fit(X_train, y_train)
    y_pred_knn = model_knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)

    # Initialize and train a Logistic Regression model
    from sklearn.linear_model import LogisticRegression
    model_lr = LogisticRegression()
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    # Initialize and train a Naive Bayes model
    model_naive_base = GaussianNB()
    model_naive_base.fit(X_train, y_train)
    y_pred_nb = model_naive_base.predict(X_test)
    acc_nb = accuracy_score(y_test, y_pred_nb)

    # Initialize and train a Support Vector Machine (SVM) model
    from sklearn import svm
    model_svm = svm.SVC(kernel="rbf", C=10)  # Using radial basis function kernel
    model_svm.fit(X_train, y_train)
    y_pred_svm = model_svm.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)

    # Store model accuracies in a dictionary
    model_accuracies = {
        "SVM": acc_svm,
        "ANN": acc_ann,
        "Decision Tree": acc_dt,
        "Logistic Regression": acc_lr,
        "Naive Bayes": acc_nb,
        "KNN": acc_knn
    }

    # Find the model with the highest accuracy
    best_model = max(model_accuracies, key=model_accuracies.get)
    best_accuracy = model_accuracies[best_model]

    # Print the best model and its accuracy
    print(f"The best model is {best_model} \n Max-Accuracy : {best_accuracy:.3f}%")

    # Plotting the accuracies of different models
    models = list(model_accuracies.keys())
    accuracies = list(model_accuracies.values())

    plt.figure(figsize=(10, 5))
    plt.bar(models, accuracies, color='purple')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracies')
    plt.ylim([0, 1])
    plt.axhline(y=max(accuracies), color='r', linestyle='--', label=f'Best Model: {best_model} ({best_accuracy:.2f}%)')
    plt.legend()

    plt.show()
