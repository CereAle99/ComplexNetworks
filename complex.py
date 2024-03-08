import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os
from sklearn.metrics import accuracy_score, classification_report



def model(train, test, label_train, label_test):
    # Creare e addestrare un classificatore SVM
    classifier = LogisticRegression()
    classifier.fit(train, label_train)

    # Effettuare delle predizioni sul set di test
    predictions = classifier.predict(test)

    accuracy = accuracy_score(label_test, predictions)
    return accuracy


# Function to calculate performance metric using cross-validation
def calculate_performance(feature_set, labels):
    kfcv = KFold(n_splits=5, shuffle=True, random_state=43)
    scores = []
    for idx_train, idx_test in  kfcv.split(X, y):
        score = model(feature_set.iloc[idx_train], feature_set.iloc[idx_test], labels.iloc[idx_train], labels.iloc[idx_test])
        scores.append(score)
    return np.mean(scores)

# Create a fully connected symmetric weighted network
def create_network(features, labels):
    G = nx.Graph()
    print(len(features.columns))
    print(len(features.columns) * (len(features.columns) - 1))
    for i in range(len(features.columns)):
        for j in range(i + 1, len(features.columns)):
            performance = calculate_performance(features.iloc[:,[i,j]], labels)
            print(i, j, performance)
            G.add_node(features.iloc[:,[i]].columns[0], label=features.iloc[:,[i]].columns[0])
            G.add_node(features.iloc[:,[j]].columns[0], label=features.iloc[:,[j]].columns[0])
            G.add_edge(features.iloc[:,[i]].columns[0], features.iloc[:,[j]].columns[0], weight=performance)

    return G

# Thresholding to select top-performing pairs
def apply_threshold(G, threshold):
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < threshold]
    G.remove_edges_from(edges_to_remove)
    return G

# Apply the entire process
def signature_extraction(features, labels, threshold):
    # Create the fully connected network
    network = create_network(features, labels)

    # Apply thresholding
    network_thresholded = apply_threshold(network, threshold)

    # Remove pendant nodes
    network_core = nx.k_core(network_thresholded)

    return network_core

if __name__ == "__main__":

    # Getting data directory
    current_path = os.getcwd()
    data_path = os.path.join(current_path, "data")
    results_path = os.path.join(current_path, "results")
    X_path = os.path.join(data_path, "MM_CT_d.csv")
    y_path = os.path.join(data_path, "surv.csv")

    # Data loading
    X = pd.read_csv(X_path, index_col="patient_id")
    y = pd.read_csv(y_path, index_col="MPC")
    y = y.loc[:,'PFS_I_EVENT']

    threshold_value = 0.7
    # X_ = X.iloc[:,:806]
    X_ = X
    final_network = signature_extraction(X_, y, threshold_value)



    feature_selected = list(final_network.nodes())
    print(feature_selected)
    file_path = os.path.join(results_path, "feature_selected.txt")

    # Open the file in 'w' mode (write mode)
    with open(file_path, 'w') as file:
        # Write each element of the list to a new line in the file
        for item in feature_selected:
            file.write(f"{item}\n")


    X_reduced = X.iloc[:,feature_selected]
    X_reduced.to_csv(os.path.join(results_path, "dataset_reduced.csv"), index=True)


    # Visualization 
    pos = nx.spring_layout(final_network)
    nx.draw(final_network, pos, with_labels=True)
    labels = nx.get_edge_attributes(final_network, 'weight')
    nx.draw_networkx_edge_labels(final_network, pos, edge_labels=labels)
    plt.show()
