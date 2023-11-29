import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import model.linear_classifier as mLC
import model.knn_classifier as mKNN
import model.naive_decision_tree_classifer as mDT
import model.pruned_decision_tree_classifer as mPDT
import config as cfg

def save_dict(filename, dict):
    with open(filename, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=dict.keys())
        writer.writeheader()
        writer.writerow(dict)

def save_plot(filename, train_list, valid_list, plotname='accuracy'):
    fig, ax = plt.subplots(figsize=(8,4))
    plt.title(plotname)
    plt.plot(train_list, label='train'+plotname)
    plt.plot(valid_list, label='valid'+plotname, linestyle='--')
    plt.legend()
    plt.savefig(filename)
    plt.show()

def save_bar_chart(filename, label_list, data_list, plotname='Feature Importance'):
    plt.title(plotname)
    plt.bar(label_list, data_list)
    plt.savefig(filename)
    plt.show()

def gen_train_test_data(filename, train_test_split_ratio):
    X = pd.read_csv(filename, usecols = cfg.feature_label_list)
    Y = pd.read_csv(filename, usecols = ['is_claim'])

    # Reformat as numpy and set label as {1, -1}
    X = X.to_numpy()
    Y = Y.to_numpy()
    Y[Y==0] = -1

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=train_test_split_ratio[1], random_state=83)
    Y_train = np.squeeze(Y_train)
    Y_test = np.squeeze(Y_test)
    return X_train, X_test, Y_train, Y_test

def do_minMaxNormalization(X):
    min_X = np.min(X, axis=0)
    max_X = np.max(X, axis=0)
    normalized_X = (X-min_X)/(max_X-min_X)
    return normalized_X

def cal_pertubation_importance(clf, X, Y, clf_name = "LC", num_repetition=2):
    # ref: https://scikit-learn.org/stable/modules/permutation_importance.html
    if clf_name == "LC":
        score, _ = clf.pred(X, Y)
    else:
        score = clf.pred(X, Y)
    print(f"Calculate pertubation importance:")
    print(f"reference accuracy = {score}")

    feature_score_list = np.zeros(X.shape[1])
    for feature_id in range(X.shape[1]):
        sum_scores = 0
        for i in range(num_repetition):
            np.random.seed(i)
            np.random.shuffle(X[:, feature_id])
            if clf_name == "LC":
                score_i, _ = clf.pred(X, Y)
            else:
                score_i = clf.pred(X, Y)
            feature_score_list[feature_id] += score_i
        feature_score_list = feature_score_list/num_repetition

    importance_list = score - feature_score_list
    print(f"importance_list = {importance_list}")
    return importance_list


if __name__ == '__main__':

    X_train, X_test, Y_train, Y_test = gen_train_test_data(cfg.datapath, cfg.train_test_split_ratio)
    X_train_normalized = do_minMaxNormalization(X_train)
    X_test_normalized = do_minMaxNormalization(X_test)
    #print(X_test_normalized.shape)

    clf = mLC.LinearClassifier(num_input_feature=X_train.shape[1])
    acc, _ = clf.fit(X_test_normalized, Y_test)
    importanence_list = cal_pertubation_importance(clf, X_test_normalized, Y_test, 1)
    print(importanence_list)