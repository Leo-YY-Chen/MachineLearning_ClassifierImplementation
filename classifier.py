from datetime import datetime
import numpy as np
import os
import pickle
import data_processor as dp

    

class Classifier:
    def __init__(self):
        self.timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.attributes = {'hyper_parameters':{}, 'parameters':{}}
        self.hyper_parameters = {}
        self.parameters = {}
        self.accuracy = np.nan

    # Assume that   (train_, test_)features:    2D array (feature_type, feature_value)
    #               (train_, test_)labels:      1D array (label)

    def k_fold_cross_validation(self, train_features, train_labels, test_features, test_labels, k = 3):
        data_processor = dp.Data_Processor()
        for i in range(k):
            self.train(data_processor.get_ith_fold_data(k, i, train_features, train_labels))
            self.valid(data_processor.remove_ith_fold_data(k, i, train_features, train_labels))
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
            pickle.dump(self.attributes, fp)
    
    def load_classifier(self, pkl_file_name):
        with open(pkl_file_name, 'rb') as fp:
            self.attributes = pickle.load(fp)


    



    def get_file_name(self):
        return os.path.join(os.getcwd(), 'model/checkpoint', self.timestamp) + '.pkl'
    
    
    







        

    
        
    

if __name__ == "__main__":
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
        clf = Classifier()
        clf.attributes = {'hyper_parameters':{'k': 10, 'lr': 0.001}, 'parameters':{'weight':list(range(7)) ,'bias':np.array(range(7))}}
        filename = clf.get_file_name()

        clf.save_classifier()
        clf.attributes = {'hyper_parameters':{}, 'parameters':{}}
        if len(clf.attributes['hyper_parameters']) == 0 and len(clf.attributes['parameters']) == 0:
            clf.load_classifier(filename)
            if (clf.attributes['hyper_parameters'] == {'k': 10, 'lr': 0.001}) and (clf.attributes['parameters']['weight'] == list(range(7))) and (clf.attributes['parameters']['bias']==np.array(range(7))).all():
                print("passing")
            else:
                print("loading fail")
        else:
            print("saving fail")
    test_save_and_load_classifier()'''


    