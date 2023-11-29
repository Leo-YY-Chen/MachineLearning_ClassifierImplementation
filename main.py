import config as cfg
import utils
import model.linear_classifier as mLC
import model.knn_classifier as mKNN
import model.naive_decision_tree_classifer as mDT
import model.pruned_decision_tree_classifer as mPDT



'''
STAGE1: Setup data and classifier
'''
# Generate the data.
X_train, X_test, Y_train, Y_test = utils.gen_train_test_data(cfg.datapath, cfg.train_test_split_ratio)
# Preprocess the data.
X_train_normalized = utils.do_minMaxNormalization(X_train)
X_test_normalized = utils.do_minMaxNormalization(X_test)
# Declare a classifier. 
clf = mLC.LinearClassifier(num_features = X_train.shape[1])
### clf = mKNN.KNNClassifier()
### clf = mDT.DTClassifier()
### clf = mPDT.PDTClassifier()




'''
STAGE2: Feature engineering
'''
# Fit the classifier.
clf.fit(X_train_normalized, Y_train)


# Implement permutation importance algorithm (PIA)
feature_importance_list = utils.cal_pertubation_importance(clf, X_test_normalized, Y_test, cfg.clf_name)
feature_dict = {"feature_importance_list":feature_importance_list}
filename = './data/feature_importance/'+  f'{cfg.clf_name}_PermutImport_{clf.time}'
utils.save_dict(filename + '.csv', feature_dict)
utils.save_bar_chart(filename +'.png', cfg.feature_label_list, feature_importance_list, 'importance')


# Implement SHAP




'''
STAGE3: Train classifers w/(o) k-fold cross validation
'''
# generate data

