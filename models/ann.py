def runscript():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split,GridSearchCV
    import networkx as nx
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, recall_score, precision_score,confusion_matrix,classification_report,f1_score,ConfusionMatrixDisplay
    df = pd.read_csv("./churn.csv")
    df['MultipleLines'] = df['MultipleLines'].str.replace('No phone service', 'No')
    df['OnlineSecurity'] = df['OnlineSecurity'].str.replace('No internet service', 'No')
    df['OnlineBackup'] = df['OnlineBackup'].str.replace('No internet service', 'No')
    df['DeviceProtection'] = df['DeviceProtection'].str.replace('No internet service', 'No')
    df['TechSupport'] = df['TechSupport'].str.replace('No internet service', 'No')
    df['StreamingTV'] = df['StreamingTV'].str.replace('No internet service', 'No')
    df['StreamingMovies'] = df['StreamingMovies'].str.replace('No internet service', 'No')
    df=df.drop('customerID',axis=1)
    df=df.head(4000)
    df=df.dropna()
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
                
    str2int(df)
    #seprate Labels from the other datasets

    """ x:remaining datasets
        y:labels of the remaining datasets """
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    scaler=StandardScaler()
    X=scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=True)

    #model:
    # We get these parameters from Grid Search:
    best_hidden_layer_sizes= (2, 8, 10)
    best_activation = 'logistic'
    best_solver = 'lbfgs'
    best_learning_rate = 'constant'
    model_mlp = MLPClassifier(activation=best_activation,solver=best_solver,learning_rate=best_learning_rate,max_iter=2000, random_state=42,hidden_layer_sizes=best_hidden_layer_sizes)
    model_mlp.fit(X_train, y_train)
    # Just for the tests
    y_pred=model_mlp.predict(X_test)
    # Accuracy
    acc = accuracy_score(y_test,y_pred)
    # Recall
    r=recall_score(y_test,y_pred,average="micro")
    #Precision
    pr= precision_score(y_test,y_pred,average="micro")
    #f1 
    f1= f1_score(y_test,y_pred,average="micro")

    print(f" Accuracy: {acc:.4} \n Recall : {r:.4} \n precision : {pr:.4} \n F1 : {f1:.4}")
    # Check for overfitting (Train & Test Data)
    train_acc = model_mlp.score(X_train, y_train)
    test_acc = model_mlp.score(X_test, y_test)
    print(f" Training Accuracy: {train_acc:.4f} \n Test Accuracy: {test_acc:.2}")
    # Function to plot the MLPClassifier structure
    def plot_mlp_structure(mlp,input_size):
        layers = [input_size] + list(mlp.hidden_layer_sizes) + [1]  # Add input and output layers

        G = nx.DiGraph()
        node_count = 0
        layer_nodes = []

        # Create nodes for each layer
        for i, layer_size in enumerate(layers):
            layer_nodes.append([])
            for _ in range(layer_size):
                G.add_node(node_count, layer=i)
                layer_nodes[-1].append(node_count)
                node_count += 1

        # Create edges between nodes of subsequent layers
        for i in range(len(layer_nodes) - 1):
            for node in layer_nodes[i]:
                for next_node in layer_nodes[i + 1]:
                    G.add_edge(node, next_node)

        pos = {}
        for i, layer in enumerate(layer_nodes):
            for j, node in enumerate(layer):
                pos[node] = (i, j - len(layer) / 2)

        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=False, node_size=700, node_color="purple", edge_color="gray")
        labels = {i: f"Input {i+1}" for i in range(input_size)}
        hidden_layers = sum([[f"H{j+1}" for j in range(size)] for size in mlp.hidden_layer_sizes], [])
        labels.update({i + input_size: hidden_layers[i] for i in range(len(hidden_layers))})
        labels.update({node_count - 1: "Output"})
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        plt.title("MLPClassifier Structure")
        plt.grid(True)
        plt.show()

    # Plot the MLPClassifier structure
    plot_mlp_structure(model_mlp,X.shape[1])


