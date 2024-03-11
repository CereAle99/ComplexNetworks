import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split, cross_val_score
import os
from sklearn.metrics import accuracy_score
import statistics
from sklearn.preprocessing import StandardScaler



def model(train, test, label_train, label_test):
    # Create and train a Logistic Regression model
    classifier = LogisticRegression()
    classifier.fit(train, label_train)

    # Execute prediction with trained model
    predictions = classifier.predict(test)

    accuracy = accuracy_score(label_test, predictions)
    return accuracy


# Function to calculate performance metric using cross-validation
def calculate_performance(feature_set, labels):
    kfcv = KFold(n_splits=5, shuffle=True, random_state=43)
    scores = []
    for idx_train, idx_test in  kfcv.split(feature_set, labels):
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

def apply_top_features(G, num_features):
    # Get all edges with their weights
    edges_with_weights = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]

    # Sort edges by weight in descending order
    sorted_edges = sorted(edges_with_weights, key=lambda x: x[2], reverse=True)

    # Extract the top features based on weights
    top_features = set()
    
    for edge in sorted_edges[:num_features]:
        top_features.add(edge[0])
        top_features.add(edge[1])

    # Create a new graph with only the top edges
    top_edges = [(u, v) for u, v, _ in sorted_edges[:num_features]]
    subgraph = G.edge_subgraph(top_edges)

    # Create a subgraph with only the top features
    # subgraph = G.subgraph(top_features)

    return subgraph


# Apply the entire process
def signature_extraction(features, labels, num_edges):
    # Create the fully connected network
    network = create_network(features, labels)

    # Apply thresholding
    network_thresholded = apply_top_features(network, num_edges)

    # Remove pendant nodes
    network_core = nx.k_core(network_thresholded)

    return network_core


def LR_evaluation(sample, label, penalization=True, show=False):
    # Cross-Validation splits
    kfcv = KFold(n_splits=5, shuffle=True, random_state=15)

    # Scores list
    scores = []

    # Cross validation for LR training and evaluation with complete dataset
    for train_idx, test_idx in kfcv.split(sample, label):

        X_train, X_test, y_train, y_test = sample.iloc[train_idx], sample.iloc[test_idx], label.iloc[train_idx], label.iloc[test_idx]

        # Create a logistic regression
        if penalization == True:
            model = LogisticRegression(penalty='l1', solver='liblinear', C=1.5)
        else:
            model = LogisticRegression()

        # Fit the model to your training data
        model.fit(X_train, y_train)

        # Make predictions
        scores.append(accuracy_score(y_test, model.predict(X_test)))

    mean = statistics.mean(scores)
    error = statistics.stdev(scores)/np.sqrt(5)
    if show == True:
        print(f'Complete model results:  {mean:.2f} \u00B1 {error:.2f}')

    return scores, mean, error


def single_feature_method(features, labels, num_features):
    
    kfcv = KFold(n_splits=5, shuffle=True, random_state=43)
    feature_scores = dict()
    for i in range(len(features.columns)):
        scores = []
        for idx_train, idx_test in  kfcv.split(features, labels):
            model = LogisticRegression()
            model.fit(features.iloc[idx_train, [i]], labels.iloc[idx_train])
            performance = accuracy_score(model.predict(features.iloc[idx_test, [i]]), labels.iloc[idx_test])
            scores.append(performance)
        print(i, np.mean(scores))
        feature_scores[features.columns[i]] = np.mean(scores)
    sorted_dict = dict(sorted(feature_scores.items(), key=lambda item: item[1]))
    features_selected = list(sorted_dict.keys())[:num_features]
    return features_selected


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
    #X = X.loc[:,feats]
    

    y = y.loc[:,'PFS_I_EVENT']


    # Create a StandardScaler
    scaler = StandardScaler()

    # Fit and transform the feature dataset
    X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Split: a dataset for feature selecting and one for model evaluation
    X_selection, X_evaluation, y_selection, y_evaluation = train_test_split(X_normalized, y, test_size=0.33, random_state=42)


    num_features = 15
    trad_feature_selection = single_feature_method(X_selection, y_selection, num_features)
    print('\nFeature selected single evaluation:\n', trad_feature_selection)
    num_edges = 10
    final_network = signature_extraction(X_selection, y_selection, num_edges)
    feature_selected = list(final_network.nodes())
    print('\nFeature selected DNetPRO:\n', feature_selected)


    # Open the file in 'w' mode (write mode)
    file_path = os.path.join(results_path, "fixed_selection.txt")
    with open(file_path, 'w') as file:
        # Write each element of the list to a new line in the file
        for item in feature_selected:
            file.write(f"{item}\n")
    
    print("\nLR total feature:")
    LR_evaluation(X_evaluation, y_evaluation, penalization=True, show=True)


    X_trad = X.loc[:,trad_feature_selection]
    print('\nSingle feature method for feature selection:')
    best_mean = 0
    best_error = 0
    for i in range(20):
        _, mean, error = LR_evaluation(X_evaluation.loc[:,trad_feature_selection[:i+1]], y_evaluation, penalization=False, show=True)
        if mean > best_mean:
            best_mean = mean
            best_error = error
    print(f'Complete model results:  {best_mean:.2f} \u00B1 {best_error:.2f}')

    X_reduced = X.loc[:,feature_selected]
    X_reduced.to_csv(os.path.join(results_path, "fixed_reduced_dataset.csv"), index=True)
    print('\nDNetPRO feature selection:')
    best_mean = 0   
    best_error = 0
    for i in range(20):
        _, mean, error = LR_evaluation(X_evaluation.loc[:,feature_selected[:i+1]], y_evaluation, penalization=False, show=True)
        if mean > best_mean:
            best_mean = mean
            best_error = error
    print(f'Complete model results:  {best_mean:.2f} \u00B1 {best_error:.2f}')

    # Visualization 
    pos = nx.spring_layout(final_network)
    nx.draw(final_network, pos, with_labels=True)
    labels = nx.get_edge_attributes(final_network, 'weight')
    nx.draw_networkx_edge_labels(final_network, pos, edge_labels=labels)
    plt.show()