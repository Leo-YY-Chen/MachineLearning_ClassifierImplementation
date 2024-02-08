from datetime import datetime
import sys
sys.path.append('..')
import utils
import numpy as np
import config as cfg
import os
import pickle

    

class Classifier_v2:
    def __init__(self):
        self.timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.hyper_parameters = {}
        self.parameters = {}
        self.accuracy = np.nan

    # Assume that   (train_, test_)features:    2D array (feature_type, feature_value)
    #               (train_, test_)labels:      1D array (label)

    def k_fold_cross_validation(self, train_features, train_labels, test_features, test_labels, k = 3):
        for i in range(k):
            self.train(self.get_data_without_ith_fold(k, i, train_features, train_labels))
            self.valid(self.get_ith_fold_data(k, i, train_features, train_labels))
        self.test(test_features, test_labels)

    def train(self, features, labels):
        self.update_parameters(features, labels)
        self.compute_performance_metrics(labels, self.get_prediction(features))
        self.show_performance()
        self.save_performance()
        self.save_classifier()
    
    def test(self, features, labels):
        self.compute_performance_metrics(labels, self.get_prediction(features))
        self.show_performance()
        self.save_performance()
    
    def valid(self, features, labels):
        self.test(features, labels)





    def update_parameters(self, features, labels):
        return None

    def get_prediction(self, features):
        # return labels
        return None
    
    def compute_performance_metrics(self, labels, prediction):
        self.compute_accuracy(labels, prediction)
        return None 
    
    def compute_accuracy(self, labels, prediction):
        return np.sum(prediction == labels) / len(labels)
    




    
    def show_performance(self):
        return None
    
    def save_performance(self):
        return None
    
    def save_classifier(self):
        with open(self.get_file_name(), 'wb') as fp:
            pickle.dump({'hyper_parameters':self.hyper_parameters, 'parameters': self.parameters}, fp)
    
    def load_classifier(self, pkl_file_name):
        with open(pkl_file_name, 'rb') as fp:
            dictionary = pickle.load(fp)
        self.hyper_parameters = dictionary["hyper_parameters"]
        self.parameters = dictionary["parameters"]

    def get_data_without_ith_fold(self, number_folds, i, features, labels):
        the_rest_range = self.get_range_without_ith_fold(number_folds, i, labels)
        return features[:, the_rest_range], labels[the_rest_range]
    
    def get_ith_fold_data(self, number_folds, i, features, labels):
        ith_fold_range = self.get_ith_fold_range(number_folds, i, labels)
        return features[:, ith_fold_range], labels[ith_fold_range]
        
    def get_feature_importances(self, features, labels, number_repetition=10):
        # ref: https://scikit-learn.org/stable/modules/permutation_importance.html
        return [self.get_feature_importance_by_index(index, features, labels, number_repetition) for index in features.shape[1]]
    
        
    



    def get_file_name(self):
        return os.path.join(os.getcwd(), 'checkpoint', self.timestamp) + '.pkl'
    
    def get_ith_fold_range(self, number_folds, fold_index, labels):
        fold_size = int(len(labels) / number_folds)
        start_address = fold_index*fold_size
        end_address = len(labels) if fold_index == number_folds-1 else (start_address + fold_size)
        return [i for i in range(start_address, end_address)]
    
    def get_range_without_ith_fold(self, number_folds, i, labels):
        ith_fold_range = self.get_ith_fold_range(number_folds, i, labels)
        return [i for i in range(len(labels)) if i not in ith_fold_range]
    
    def get_feature_importance_by_index(self, index, features, labels, number_repetition):
        result = 0
        for i in range(number_repetition):
                shuffled_features = self.get_feature_column_shuffled_by_index(self, index, features)
                result[index] += self.compute_accuracy(labels, self.get_prediction(shuffled_features))
        return result/number_repetition

    def get_feature_column_shuffled_by_index(self, index, features):
        features_copy = features.copy()
        np.random.shuffle(features_copy[:, index])
        return features_copy
    
    
    
    







        

    
        
    

if __name__ == "__main__":
    #######################
    # TEST getting data
    #######################
    '''def test_get_ith_fold_data():
        clf = Classifier_v2()
        fea = np.array(np.resize(range(2*7), (2,7)))
        lab = np.array(range(7))
        #print(fea, lab)
        features, labels = clf.get_ith_fold_data(3, 0, fea, lab)
        features1, labels1 = clf.get_ith_fold_data(3, 2, fea, lab)
        if (features == np.array([[0,1],[7,8]])).all() and (labels == np.array([0,1])).all():
            if (features1 == np.array([[4,5,6],[11,12,13]])).all() and (labels1 == np.array([4,5,6])).all():
                print("passing")
            else:
                print("fail") 
        else:
            print("fail")
    test_get_ith_fold_data()
    


    def test_get_ith_fold_range():
        clf = Classifier_v2()
        labels = np.array([i for i in range(7)])
        rg = clf.get_ith_fold_range(3, 0, labels)
        rg1 = clf.get_ith_fold_range(3, 2, labels)
        if (rg == [0,1]) and (rg1 == [4,5,6]):
            print("passing")
        else:
            print("fail")
    test_get_ith_fold_range()



    def test_get_data_without_ith_fold():
        clf = Classifier_v2()
        fea = np.array(np.resize(range(2*7), (2,7)))
        lab = np.array(range(7))
        #print(fea, lab)
        features, labels = clf.get_data_without_ith_fold(3, 0, fea, lab)
        features1, labels1 = clf.get_data_without_ith_fold(3, 2, fea, lab)
        if (features == np.array([[2,3,4,5,6],[9,10,11,12,13]])).all() and (labels == np.array([2,3,4,5,6])).all():
            if (features1 == np.array([[0,1,2,3],[7,8,9,10]])).all() and (labels1 == np.array([0,1,2,3])).all():
                print("passing")
            else:
                print("fail") 
        else:
            print("fail")
    test_get_data_without_ith_fold()'''




    #######################
    # TEST getting feature importance
    #######################
    '''def test_get_feature_column_shuffled_by_index():
        clf = Classifier_v2()
        origin = np.array(np.resize(range(10*7), (10,7)))
        result = clf.get_feature_column_shuffled_by_index(6, origin)
        #print(origin,result)
        if (result[:,0:5] == origin[:,0:5]).all() and (result[:,6] != origin[:,6]).any():
            print("passing")
        else:
            print("fail")
    test_get_feature_column_shuffled_by_index()'''




    #######################
    # TEST save and load classifier
    #######################
    '''def test_save_and_load_classifier():
        clf = Classifier_v2()
        clf.hyper_parameters = {'k': 10, 'lr': 0.001}
        clf.parameters = {'weight':list(range(7)) ,'bias':np.array(range(7))}
        filename = clf.get_saving_filename()

        clf.save_classifier()
        clf.hyper_parameters = {}
        clf.parameters = {}
        if len(clf.hyper_parameters) == 0 and len(clf.parameters) == 0:
            clf.load_classifier(filename)
            if (clf.hyper_parameters == {'k': 10, 'lr': 0.001}) and (clf.parameters == {'weight':list(range(7)) ,'bias':np.array(range(7))}).all():
                print("passing")
            else:
                print("loading fail")
        else:
            print("saving fail")
    test_save_and_load_classifier()'''


    