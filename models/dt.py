def runscript():
    # Import necessary libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score
    import seaborn as sns  # Heat map for Confusion Matrix
    from sklearn import tree

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

    # Initialize and train a Decision Tree model
    model_dt = DecisionTreeClassifier(max_depth=5, min_samples_split=4, min_samples_leaf=6)
    model_dt.fit(X_train, y_train)

    # Predict the test set results
    y_pred = model_dt.predict(X_test)

    # Calculate performance metrics
    acc = accuracy_score(y_test, y_pred)          # Accuracy
    r = recall_score(y_test, y_pred, average="micro")  # Recall
    pr = precision_score(y_test, y_pred, average="micro")  # Precision
    f1 = f1_score(y_test, y_pred, average="micro")  # F1 Score

    # Print performance metrics
    print(f" Accuracy: {acc:.4} \n Recall : {r:.4} \n Precision : {pr:.4} \n F1 : {f1:.4}")

    # Check for overfitting by comparing training and test accuracy
    train_acc = model_dt.score(X_train, y_train)
    test_acc = model_dt.score(X_test, y_test)
    print(f" Training Accuracy: {train_acc:.4f} \n Test Accuracy: {test_acc:.2}")

    # Plot the decision tree
    plt.figure(figsize=(18, 18))
    tree.plot_tree(model_dt, filled=True, node_ids=True, impurity=False, max_depth=None, rounded=True, fontsize=5, label="all")
    plt.show()

    # Display the confusion matrix
    plt.figure()
    conf_mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_mat, cmap="Blues", cbar=True, annot=True)
    plt.xticks(np.arange(3) + 0.5, labels=['Class 0', 'Class 1', 'Class 2'])  # Adjust according to your classes
    plt.yticks(np.arange(3) + 0.5, labels=['Class 0', 'Class 1', 'Class 2'])  # Adjust according to your classes
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.show()
