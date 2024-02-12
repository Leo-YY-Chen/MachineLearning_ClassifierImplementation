import numpy as np
import data_processor
import classifier
import calculator
import presenter
from model.knn_classifier import KNNClassifier






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
    classifier = classifier.Classifier(calculator.Performance_Calculator(), presenter.Performance_Presenter())
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
    classifier = classifier.Classifier(calculator.Performance_Calculator(), presenter.Performance_Presenter())
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
    classifier = classifier.Classifier(calculator.Performance_Calculator(), presenter.Performance_Presenter())
    classifier.set_metrics(np.array([0,0,0,0,0]), np.array([0,0,0,1,1]))

    if classifier.metrics.accuracy == 0.6:
        print("passing")
    else:
        print("fail")
test_calculate_metrics()'''







#######################
# TEST KNNClassifier
#######################
'''def test_calcualte_distances():
    knn = KNNClassifier(calculator = calculator.Performance_Calculator(), 
                        presenter = presenter.Performance_Presenter(), 
                        p_norm = 2)
    knn.attributes['parameters']['train_features'] = np.array([[0,0],[1,1],[2,2]])
    
    distance = knn.calcualte_distances(np.array([1,0]))
    if (distance == np.array([1,1,np.sqrt(5)])).all():
        print("passing")
    else:
        print("fail")
test_calcualte_distances()



def test_get_neighbors_indices():
    knn = KNNClassifier(calculator = calculator.Performance_Calculator(), 
                        presenter = presenter.Performance_Presenter(), 
                        p_norm = 2)
    
    indices = knn.get_neighbors_indices(np.array([1,0,3,5,0.1]))
    if (indices == [1,4,0]).all():
        print("passing")
    else:
        print("fail")
test_get_neighbors_indices()



def test_get_major_label_of_neighbors():
    knn = KNNClassifier(calculator = calculator.Performance_Calculator(), 
                        presenter = presenter.Performance_Presenter(), 
                        p_norm = 2)
    knn.attributes['parameters']['train_labels'] = np.array([1, -1, -1, -1, 1])
    
    major_label = knn.get_major_label_of_neighbors([1,4,0])
    if (major_label == 1):
        print("passing")
    else:
        print("fail")
test_get_major_label_of_neighbors()




def test_get_predictions():
    knn = KNNClassifier(calculator = calculator.Performance_Calculator(), 
                        presenter = presenter.Performance_Presenter(), 
                        p_norm = 2)
    knn.attributes['parameters']['train_features'] = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]])
    knn.attributes['parameters']['train_labels'] = np.array([1, -1, 1, -1, 1, -1])
    
    major_label = knn.get_predictions(np.array([[1,0],[1,1],[6,6]]))
    #print(major_label)
    if (major_label == np.array([1, -1, -1])).all():
        print("passing")
    else:
        print("fail")
test_get_predictions()



def test_update_parameters():
    knn = KNNClassifier(calculator = calculator.Performance_Calculator(), 
                        presenter = presenter.Performance_Presenter(), 
                        p_norm = 2)
    features = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]])
    labels = np.array([1, -1, 1, -1, 1, -1])
    features1 = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]])
    labels1 = np.array([1, -1, 1, -1, 1, -1])

    knn.update_parameters(features, labels)
    if (knn.attributes["parameters"]["train_features"] == features1).all() and (knn.attributes["parameters"]["train_labels"]==labels1).all():
        print("passing")
    else:
        print("fail")
test_update_parameters()



def test_train():
    knn = KNNClassifier(calculator = calculator.Performance_Calculator(), 
                        presenter = presenter.Performance_Presenter(), 
                        p_norm = 2)
    features = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]])
    labels = np.array([1, -1, 1, -1, 1, -1])

    knn.train(features, labels)
test_train()'''



'''def test_test():
    knn = KNNClassifier(calculator = calculator.Performance_Calculator(), 
                        presenter = presenter.Performance_Presenter(), 
                        p_norm = 2)
    train_features = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]])
    train_labels = np.array([1, -1, 1, -1, 1, -1])
    test_features = np.array([[1,0],[1,1],[6,6]])
    test_labels = np.array([-1,-1,-1])
    
    knn.train(train_features, train_labels)
    knn.test(test_features, test_labels)
test_test()


def test_k_fold_cross_validation():
    knn = KNNClassifier(calculator = calculator.Performance_Calculator(), 
                        presenter = presenter.Performance_Presenter(), 
                        p_norm = 2)
    train_features = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]])
    train_labels = np.array([1, -1, 1, -1, 1, -1])
    test_features = np.array([[1,0],[1,1],[6,6]])
    test_labels = np.array([-1,-1,-1])
    knn.k_fold_cross_validation(train_features, train_labels, test_features, test_labels, data_processor.Data_Processor())
test_k_fold_cross_validation()'''




#######################
# TEST LinearClassifier
#######################
def test_2D_1D_array_broadcast_multiply():
    twoDArray = np.array([[1,2],[3,4],[5,6]])
    oneDArray = np.expand_dims(np.array([1,-1,3]), axis=1)
    
    if (twoDArray*oneDArray == np.array([[1,2],[-3,-4],[15,18]])).all():
        print("passing")
    else:
        print("fail")
test_2D_1D_array_broadcast_multiply()