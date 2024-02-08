import data.config as cfg
import model.utils as utils
import model.linear_classifier as mLC
import model.knn_classifier as mKNN
import model.naive_decision_tree_classifer as mDT
import model.pruned_decision_tree_classifer as mPDT

dataset = utils.Dataset()
dataset.load(cfg.datapath)
dataset.preprocess()
train_dataset, test_dataset = dataset.split_in_ratio()

clf = mLC.LinearClassifier()
clf.train_then_show_and_save_results(train_dataset)
clf.set_show_and_save_feature_importance(train_dataset)

'''clf.test_then_show_and_save_results(test_dataset)
clf.k_fold_cross_validation(dataset)
'''