def runscript():
    # Import necessary libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, classification_report
    import seaborn as sns  # Heat map for Confusion Matrix

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

    # Apply PCA for dimensionality reduction to 2 components
    pca = PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)

    # Determine the optimal number of clusters using the elbow method
    wcss = []  # List to store within-cluster sum of squares for each k
    range_of_k = range(1, 11)  # Range of k values to try (from 1 to 10 clusters)

    # Loop through each k value
    for k in range_of_k:
        kmeans = KMeans(n_clusters=k, random_state=42)  # Create KMeans instance with current k
        kmeans.fit(X)  # Fit the KMeans model to the data
        wcss.append(kmeans.inertia_)  # Append WCSS value to the list

    # Plot the WCSS values to visualize the elbow method
    plt.figure(figsize=(10, 6))
    plt.plot(range_of_k, wcss, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.show()

    # Based on the elbow method, choose the optimal number of clusters (k)
    best_k = 7
    model_kmeans = KMeans(n_clusters=best_k, random_state=42)
    model_kmeans.fit(X)
    y_pred = model_kmeans.predict(X)
    print(f"Best K from elbow method: {best_k}\n")

    """
    # Uncomment this section if you want to calculate accuracy for clustering
    acc_tr = accuracy_score(y, y_pred)
    print(f'Accuracy for KMEANS is :{acc_tr:0.3f}')
    """

    # Visualize the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=model_kmeans.labels_)
    plt.scatter(model_kmeans.cluster_centers_[:, 0], model_kmeans.cluster_centers_[:, 1], marker='x', c='r', s=150)
    plt.title('K-Means clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
