import numpy as np
from classifier import Classifier
import calculator
import presenter






'''dataset = utils.Dataset()
dataset.load(cfg.datapath)
dataset.preprocess()
train_dataset, test_dataset = dataset.split_in_ratio()

clf = mLC.LinearClassifier()
clf.train_then_show_and_save_results(train_dataset)
clf.set_show_and_save_feature_importance(train_dataset)'''

'''clf.test_then_show_and_save_results(test_dataset)
clf.k_fold_cross_validation(dataset)
'''

#######################
# TEST classifier performance
#######################
'''def test_show_performance():
    classifier = Classifier(calculator.Performance_Calculator(), presenter.Performance_Presenter())
    classifier.information.type="TESTING"
    classifier.information.epoch_quantity = 1000
    classifier.information.epoch_number = 666
    classifier.information.fold_quantity = 3
    classifier.information.fold_number = 0
    classifier.metrics.accuracy = 100
    classifier.metrics.loss = None
    classifier.metrics.train_loss = [i*np.pi for i in range(1000)]
    classifier.metrics.valid_loss = [i for i in range(1000)]
    classifier.metrics.train_accuracy = [i*np.e for i in range(1000)]
    classifier.metrics.valid_accuracy = [i*np.pi for i in range(1000)]
    
    classifier.show_performance()
test_show_performance()


def test_save_performance():
    classifier = Classifier(calculator.Performance_Calculator(), presenter.Performance_Presenter())
    classifier.information.type="TESTING"
    classifier.information.epoch_quantity = 1000
    classifier.information.epoch_number = 666
    classifier.information.fold_quantity = 3
    classifier.information.fold_number = 0
    classifier.metrics.accuracy = 100
    classifier.metrics.loss = None
    classifier.metrics.train_loss = [i*np.pi for i in range(1000)]
    classifier.metrics.valid_loss = [i for i in range(1000)]
    classifier.metrics.train_accuracy = [i*np.e for i in range(1000)]
    classifier.metrics.valid_accuracy = [i*np.pi for i in range(1000)]
    
    classifier.save_performance()
test_save_performance()


def test_calculate_metrics():
    classifier = Classifier(calculator.Performance_Calculator(), presenter.Performance_Presenter())
    classifier.set_metrics(np.array([0,0,0,0,0]), np.array([0,0,0,1,1]))

    if classifier.metrics.accuracy == 0.6:
        print("passing")
    else:
        print("fail")
test_calculate_metrics()'''